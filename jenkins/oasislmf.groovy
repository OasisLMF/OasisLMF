
node {
    hasFailed = false
    sh 'sudo /var/lib/jenkins/jenkins-chown'
    deleteDir() // wipe out the workspace

    set_piwind_branch='develop'
    if (BRANCH_NAME.matches("master") || BRANCH_NAME.matches("hotfix/(.*)") || BRANCH_NAME.matches("release/(.*)")){
        set_piwind_branch='master'
    }

    properties([
      parameters([
        [$class: 'StringParameterDefinition',  name: 'BUILD_BRANCH', defaultValue: 'master'],
        [$class: 'StringParameterDefinition',  name: 'SOURCE_BRANCH', defaultValue: BRANCH_NAME],
        [$class: 'StringParameterDefinition',  name: 'PIWIND_BRANCH', defaultValue: set_piwind_branch],
        [$class: 'StringParameterDefinition',  name: 'PUBLISH_VERSION', defaultValue: ''],
        [$class: 'StringParameterDefinition',  name: 'KTOOLS_VERSION', defaultValue: ''],
        [$class: 'StringParameterDefinition',  name: 'GPG_KEY', defaultValue: 'gpg-privatekey'],
        [$class: 'StringParameterDefinition',  name: 'GPG_PASSPHRASE', defaultValue: 'gpg-passphrase'],
        [$class: 'StringParameterDefinition',  name: 'TWINE_ACCOUNT', defaultValue: 'sams_twine_account'],
        [$class: 'BooleanParameterDefinition', name: 'PURGE', value: Boolean.valueOf(false)],
        [$class: 'BooleanParameterDefinition', name: 'PUBLISH', value: Boolean.valueOf(false)],
        [$class: 'BooleanParameterDefinition', name: 'SLACK_MESSAGE', value: Boolean.valueOf(false)]
      ])
    ])



    // Build vars
    String build_repo = 'git@github.com:OasisLMF/build.git'
    String build_branch = params.BUILD_BRANCH
    String build_workspace = 'oasis_build'
    String script_dir = env.WORKSPACE + "/" + build_workspace
    String PIPELINE = script_dir + "/buildscript/pipeline.sh"
    String git_creds = "1335b248-336a-47a9-b0f6-9f7314d6f1f4"

    String vers_pypi        = params.PUBLISH_VERSION
    String vers_ktools      = params.KTOOLS_VERSION
    String gpg_key          = params.GPG_KEY
    String gpg_pass         = params.GPG_PASSPHRASE
    String twine_account    = params.TWINE_ACCOUNT
    String model_branch     = params.PIWIND_BRANCH  // Git repo branch to build from
    String source_branch    = params.SOURCE_BRANCH  // Git repo branch to build from
    String source_name      = 'oasislmf'
    String source_varient   = 'pip' // Platform to build for
    String source_git_url   = "git@github.com:OasisLMF/${source_name}.git"
    String source_workspace = "${source_varient}_workspace"
    String source_sh        = '/buildscript/utils.sh'
    String source_func      = "${source_varient}_${source_name}".toLowerCase()   // function name reference <function>_<model>_<varient>

    String MDK_RUN='ri'
    String MDK_BRANCH = source_branch
    if (source_branch.matches("PR-[0-9]+")){
        MDK_BRANCH = "refs/pull/$CHANGE_ID/merge"
    }    

    //env.PYTHON_ENV_DIR = "${script_dir}/pyth-env"           // Virtualenv location
    env.PIPELINE_LOAD =  script_dir + source_sh             // required for pipeline.sh calls
    sh 'env'

    if (params.PUBLISH && ! ( source_branch.matches("release/(.*)") || source_branch.matches("hotfix/(.*)")) ){
        // fail fast, only branches named `release/*` are valid for publish
        sh "echo `Publish Only allowed on a release/* branch`"
        sh "exit 1"
    }

    try {
        parallel(
            clone_build: {
                stage('Clone: ' + build_workspace) {
                    dir(build_workspace) {
                       git url: build_repo, credentialsId: git_creds, branch: build_branch
                    }
                }
            },
            clone_source: {
                stage('Clone: ' + source_name) {
                    sshagent (credentials: [git_creds]) {
                        dir(source_workspace) {
                            sh "git clone --recursive ${source_git_url} ."
                            if (source_branch.matches("PR-[0-9]+")){
                                // Checkout PR and merge into target branch, test on the result
                                sh "git fetch origin pull/$CHANGE_ID/head:$BRANCH_NAME"
                                sh "git checkout $CHANGE_TARGET"
                                sh "git merge $BRANCH_NAME"
                            } else {
                                // Checkout branch
                                sh "git checkout ${source_branch}"
                            }   
                        }   
                    }
                }
            }
        )

        stage('Set version: ' + source_func) {
            dir(source_workspace) {
                if (vers_pypi?.trim() || vers_ktools?.trim()) {
                    sh "${PIPELINE} set_vers_oasislmf ${vers_pypi} ${vers_ktools}"
                } else {
                    println("Keep current version numbers")

                }
                if (! vers_pypi?.trim() && params.PUBLISH){
                    vers_file = readFile("oasislmf/__init__.py")
                    vers_pypi = vers_file.trim().split("'")[-1]
                    println("Loaded package version from file: $vers_pypi")
                }    
            }
        }

        stage('Run MDK: PiWind 3.6') {
            dir(build_workspace) {
                sh "sed -i 's/FROM.*/FROM python:3.6/g' docker/Dockerfile.mdk-tester"
                sh 'docker build -f docker/Dockerfile.mdk-tester -t mdk-runner:3.6 .'
                sh "docker run mdk-runner:3.6 --model-repo-branch ${model_branch} --mdk-repo-branch ${MDK_BRANCH} --model-run-mode ${MDK_RUN}"
            }
        }

        stage('Build: ' + source_func) {
            dir(source_workspace) {
                sh ' ./runtests.sh'
            }
        }

        // Access to stored GPG key
        // https://jenkins.io/doc/pipeline/steps/credentials-binding/
        //
        // gpg_key  --> Jenkins credentialId  type 'Secret file', GPG key
        // gpg_pass --> Jenkins credentialId  type 'Secret text', passphrase for the above key
        if (params.PUBLISH){
            stage('Sign Package: ' + source_func) {
                String gpg_dir='/var/lib/jenkins/.gnupg/'
                sh "if test -d ${gpg_dir}; then rm -r ${gpg_dir}; fi"
                withCredentials([file(credentialsId: gpg_key, variable: 'FILE')]) {
                    sh 'gpg --import $FILE'
                    sh 'gpg --list-keys'
                    withCredentials([string(credentialsId: gpg_pass, variable: 'PASSPHRASE')]) {
                        dir(source_workspace) {
                            sh PIPELINE + ' sign_oasislmf'
                        }
                    }
                }
                // delete GPG key from jenkins account
                sh "rm -r ${gpg_dir}"
            }
            stage ('Publish: ' + source_func) {
                dir(source_workspace) {
                    // Commit new verion numbers before pushing package
                    sshagent (credentials: [git_creds]) {
                        sh PIPELINE + " commit_vers_oasislmf ${vers_pypi}"
                    }
                    // Publish package
                    withCredentials([usernamePassword(credentialsId: twine_account, usernameVariable: 'TWINE_USERNAME', passwordVariable: 'TWINE_PASSWORD')]) {
                        sh PIPELINE + ' push_oasislmf'
                    }
                }
            }

            // Create GitHub release 
            stage("Create Release: GitHub") { 
                // Tag repo
                sshagent (credentials: [git_creds]) {
                    dir(source_workspace) {
                        sh "git tag ${vers_pypi}"
                        sh "git push origin ${vers_pypi}"
                    }
                }
                // Create GH release
                withCredentials([string(credentialsId: 'github-api-token', variable: 'gh_token')]) {
                    String repo = "OasisLMF/OasisLMF"
                    def json_request = readJSON text: '{}'
                    json_request['tag_name'] = vers_pypi
                    json_request['target_commitish'] = 'master'
                    json_request['name'] = vers_pypi
                    json_request['body'] = ""
                    json_request['draft'] = false
                    json_request['prerelease'] = false
                    writeJSON file: 'gh_request.json', json: json_request
                    sh 'curl -XPOST -H "Authorization:token ' + gh_token + "\" --data @gh_request.json https://api.github.com/repos/$repo/releases > gh_response.json"
                }
            }
        }
    } catch(hudson.AbortException | org.jenkinsci.plugins.workflow.steps.FlowInterruptedException buildException) {
        hasFailed = true
        error('Build Failed')
    } finally {
        if(params.SLACK_MESSAGE && (params.PUBLISH || hasFailed)){
            def slackColor = hasFailed ? '#FF0000' : '#27AE60'
            SLACK_GIT_URL = "https://github.com/OasisLMF/${source_name}/tree/${source_branch}"
            SLACK_MSG = "*${env.JOB_NAME}* - (<${env.BUILD_URL}|${vers_pypi}>): " + (hasFailed ? 'FAILED' : 'PASSED')
            SLACK_MSG += "\nBranch: <${SLACK_GIT_URL}|${source_branch}>"
            SLACK_MSG += "\nMode: " + (params.PUBLISH ? 'Publish' : 'Build Test')
            SLACK_CHAN = (params.PUBLISH ? "#builds-release":"#builds-dev")
            slackSend(channel: SLACK_CHAN, message: SLACK_MSG, color: slackColor)
        }

        //Store reports
        dir(source_workspace) {
            archiveArtifacts artifacts: 'reports/**/*.*'
        }

        // Run merge back if publish
         if (params.PUBLISH){ 
            dir(source_workspace) {
                sshagent (credentials: [git_creds]) {
                    sh "git checkout master && git pull"
                    sh "git merge ${source_branch} && git push"
                    sh "git checkout develop && git pull"
                    sh "git merge master && git push"
                }   
            }   
        }   
    }
}
