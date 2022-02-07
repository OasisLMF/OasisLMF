//SUB JOB TEMPLATE
def createStage(stage_name, stage_params, propagate_flag) {
    return {
        stage("Test: ${stage_name}") {
            build job: "${stage_name}", parameters: stage_params, propagate: propagate_flag
        }
    }
}

node {
    hasFailed = false
    sh 'sudo /var/lib/jenkins/jenkins-chown'
    deleteDir() // wipe out the workspace

    set_piwind_branch='develop'
    if (BRANCH_NAME.matches("master") || BRANCH_NAME.matches("hotfix/(.*)") || BRANCH_NAME.matches("release/(.*)")){
        set_piwind_branch='master'
    }
    if (BRANCH_NAME.matches("backports/(.*)")) {
        set_piwind_branch=BRANCH_NAME
    }

    properties([
      parameters([
        [$class: 'StringParameterDefinition',  description: "Oasis Build scripts branch",        name: 'BUILD_BRANCH', defaultValue: 'master'],
        [$class: 'StringParameterDefinition',  description: "OasisLMF repo branch",              name: 'SOURCE_BRANCH', defaultValue: BRANCH_NAME],
        [$class: 'StringParameterDefinition',  description: "Test against piwind branch",        name: 'PIWIND_BRANCH', defaultValue: set_piwind_branch],
        [$class: 'StringParameterDefinition',  description: "Release Version",                   name: 'PUBLISH_VERSION', defaultValue: ''],
        [$class: 'StringParameterDefinition',  description: "Last released version",             name: 'PREV_VERSION', defaultValue: ''],
        [$class: 'StringParameterDefinition',  description: "Ktools version to install",         name: 'KTOOLS_VERSION', defaultValue: ''],
        [$class: 'StringParameterDefinition',  description: "Jenkins credential for GPG",        name: 'GPG_KEY', defaultValue: 'gpg-privatekey'],
        [$class: 'StringParameterDefinition',  description: "Jenkins credential for passphrase", name: 'GPG_PASSPHRASE', defaultValue: 'gpg-passphrase'],
        [$class: 'StringParameterDefinition',  description: "Jenkins credentials Twine",         name: 'TWINE_ACCOUNT', defaultValue: 'sams_twine_account'],
        [$class: 'BooleanParameterDefinition', description: "Test worker build",                 name: 'TEST_WORKER', defaultValue: Boolean.valueOf(true)],
        [$class: 'BooleanParameterDefinition', description: "Create release if checked",         name: 'PUBLISH', defaultValue: Boolean.valueOf(false)],
        [$class: 'BooleanParameterDefinition', description: "Mark as pre-released software",     name: 'PRE_RELEASE', defaultValue: Boolean.valueOf(false)],
        [$class: 'BooleanParameterDefinition', description: "Perform a gitflow merge",           name: 'AUTO_MERGE', defaultValue: Boolean.valueOf(true)],
        [$class: 'BooleanParameterDefinition', description: "Send build status to slack",        name: 'SLACK_MESSAGE', defaultValue: Boolean.valueOf(false)]
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

    if (! params.PRE_RELEASE) {
        if (params.PUBLISH && ! ( source_branch.matches("release/(.*)") || source_branch.matches("hotfix/(.*)") || source_branch.matches("backports/(.*)")) ){
            // fail fast, only branches named `release/*` are valid for publish
            sh "echo `Publish Only allowed on a release/* branch`"
            sh "exit 1"
        }
    }

    //make sure release candidate versions are tagged correctly
    if (params.PUBLISH && params.PRE_RELEASE && ! vers_pypi.matches('^(\\d+\\.)(\\d+\\.)(\\*|\\d+)rc(\\d+)$')) {
        sh "echo release candidates must be tagged {version}rc{N}, example: 1.0.0rc1"
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
                // UPDATE ktools and package versions
                if (vers_pypi?.trim() && vers_ktools?.trim()) {
                    sh "${PIPELINE} set_vers_oasislmf ${vers_pypi} ${vers_ktools}"
                } else {
                    println("Keep current version numbers")
                }

                // Load versions as set in files
                if (! vers_pypi?.trim() && params.PUBLISH){
                    vers_file = readFile("oasislmf/__init__.py")
                    vers_pypi = vers_file.trim().split("'")[-1]
                    println("Loaded package version from file: $vers_pypi")
                }
            }
        }

        if (params.TEST_WORKER) {
            // Test current branch using a model_worker image and checking expected output
            job_params = [
                 [$class: 'StringParameterValue',  name: 'MDK_BRANCH', value: MDK_BRANCH],
                 [$class: 'StringParameterValue',  name: 'RUN_TESTS', value: 'control_set parquet'],
                 [$class: 'BooleanParameterValue', name: 'BUILD_WORKER', value: true]
            ]

            job_branch_name = model_branch.replace("/", "%2F")
            pipeline = "oasis_PiWind/$job_branch_name"
            createStage(pipeline, job_params, true).call()
        } else {
            // Only check that the MDK runs and creates non-empty files
            stage('Run MDK: PiWind 3.8') {
                dir(build_workspace) {
                    sh "sed -i 's/FROM.*/FROM python:3.8/g' docker/Dockerfile.mdk-tester"
                    sh 'docker build -f docker/Dockerfile.mdk-tester -t mdk-runner:3.8 .'
                    sh "docker run mdk-runner:3.8 --model-repo-branch ${model_branch} --mdk-repo-branch ${MDK_BRANCH} --model-run-mode ${MDK_RUN}"
                }
            }
        }

        stage('Build: ' + source_func) {
            dir(source_workspace) {
                sh ' ./scripts/runtests.sh'
            }
        }


        // Access to stored GPG key
        // https://jenkins.io/doc/pipeline/steps/credentials-binding/
        //
        // gpg_key  --> Jenkins credentialId  type 'Secret file', GPG key
        // gpg_pass --> Jenkins credentialId  type 'Secret text', passphrase for the above key

        if (params.PUBLISH){
            // Build chanagelog image
            stage("Create Changelog builder") {
                dir(build_workspace) {
                    sh "docker build -f docker/Dockerfile.release-notes -t release-builder ."
                }
            }

            // Tag release
            stage('Tag release'){
                dir(source_workspace) {
                    sshagent (credentials: [git_creds]) {
                        sh "git tag ${vers_pypi}"
                        sh "git push origin ${vers_pypi}"
                    }
                }
            }

            // Create release notes
            stage('Create Changelog'){
                dir(source_workspace) {
                    withCredentials([string(credentialsId: 'github-api-token', variable: 'gh_token')]) {
                        sh "docker run -v ${env.WORKSPACE}/${source_workspace}:/tmp release-builder build-changelog --repo OasisLMF --from-tag ${params.PREV_VERSION} --to-tag ${vers_pypi} --github-token ${gh_token} --local-repo-path ./ --output-path ./CHANGELOG.rst --apply-milestone"
                        sh "docker run -v ${env.WORKSPACE}/${source_workspace}:/tmp release-builder build-release --repo OasisLMF --from-tag ${params.PREV_VERSION} --to-tag ${vers_pypi} --github-token ${gh_token} --local-repo-path ./ --output-path ./RELEASE.md"
                    }
                    sshagent (credentials: [git_creds]) {
                        sh "git add ./CHANGELOG.rst"
                        sh "git commit -m 'Update changelog ${vers_pypi}'"
                        sh "git push"
                    }
                }
            }

            // GPG sign pip package
            // Access to stored GPG key
            // https://jenkins.io/doc/pipeline/steps/credentials-binding/
            //
            // gpg_key  --> Jenkins credentialId  type 'Secret file', GPG key
            // gpg_pass --> Jenkins credentialId  type 'Secret text', passphrase for the above key
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
                // Create GH release
                withCredentials([string(credentialsId: 'github-api-token', variable: 'gh_token')]) {
                    String repo = "OasisLMF/OasisLMF"

                    // Create release
                    def json_request = readJSON text: '{}'
                    def release_body = readFile(file: "${env.WORKSPACE}/${source_workspace}/RELEASE.md")

                    json_request['tag_name'] = vers_pypi
                    json_request['target_commitish'] = 'master'
                    json_request['name'] = vers_pypi
                    json_request['body'] = release_body
                    json_request['draft'] = false
                    json_request['prerelease'] = params.PRE_RELEASE
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
         if (params.PUBLISH && params.AUTO_MERGE && ! hasFailed){
            dir(source_workspace) {
                sshagent (credentials: [git_creds]) {
                    if (! params.PRE_RELEASE) {
                        // Release merge back into master
                        sh "git stash"
                        sh "git checkout master && git pull"
                        sh "git merge ${source_branch} && git push"
                        sh "git checkout develop && git pull"
                        sh "git merge master && git push"
                    } else {
                        // pre_pelease merge back into develop
                        sh "git stash"
                        sh "git checkout develop && git pull"
                        sh "git merge ${source_branch} && git push"
                    }
                }
            }
        }
    }
}
