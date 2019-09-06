node {
    hasFailed = false
    sh 'sudo /var/lib/jenkins/jenkins-chown'
    deleteDir() // wipe out the workspace

    properties([
      parameters([
        [$class: 'StringParameterDefinition',  name: 'SOURCE_BRANCH', defaultValue: BRANCH_NAME],
        [$class: 'StringParameterDefinition',  name: 'BENCHMARK_MNT', defaultValue: '/mnt/efs/benchmark/axa_test'],
        [$class: 'StringParameterDefinition',  name: 'BENCHMARK_THRESHOLD', defaultValue: '300'],
        [$class: 'StringParameterDefinition',  name: 'OASISLMF_ARGS', defaultValue: ''],
      ])
    ])

    String source_name      = 'oasislmf'
    String source_workspace = "${source_name}_workspace"
    String source_git_url   = "git@github.com:OasisLMF/${source_name}.git"
    String source_branch    = params.SOURCE_BRANCH
    String git_creds = "1335b248-336a-47a9-b0f6-9f7314d6f1f4"
    String run_workspace = env.WORKSPACE

    String benchmark_branch = source_branch
    String benchmark_mnt = params.BENCHMARK_MNT
    String benchmark_threshold = params.BENCHMARK_THRESHOLD
    String benchmark_args = params.OASISLMF_ARGS 


    if (source_branch.matches("PR-[0-9]+")){
        benchmark_branch = "refs/pull/$CHANGE_ID/merge"
    }
    sh 'env'

    try {
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
        stage('Build: oasislmf runner') {
            dir(source_workspace) {
                sh "docker build --build-arg oasis_ver=$benchmark_branch -f docker/Dockerfile.oasislmf_benchmark -t mdk-bench ."
            }
        }
        stage('Test: Oasis Files Gen') {
            dir(run_workspace) {
                sh "docker run -t -v $run_workspace:/var/report -v $benchmark_mnt:/var/oasis mdk-bench --time-threshold=$benchmark_threshold --extra-oasislmf-args=$benchmark_args"
                archiveArtifacts artifacts: '**/*.log'
            }
        }
    } catch(hudson.AbortException | org.jenkinsci.plugins.workflow.steps.FlowInterruptedException buildException) {
        hasFailed = true
        error('Build Failed')
    } finally {
        //pass
    }
}
