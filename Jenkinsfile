#!/usr/bin/groovy

@Library(['github.com/indigo-dc/jenkins-pipeline-library@1.2.3']) _

def job_result_url = ''

pipeline {
    agent {
        label 'python'
    }

    environment {
        author_name = "V.Kozlov (KIT)"
        author_email = "valentin.kozlov@kit.edu"
        app_name = "dogs_breed_det"
        job_location = "Pipeline-as-code/DEEP-OC-org/DEEP-OC-dogs_breed_det/${env.BRANCH_NAME}"
    }

    stages {
        stage('Code fetching') {
            steps {
                checkout scm
            }
        }

        stage('Style analysis: PEP8') {
            steps {
                ToxEnvRun('pep8')
            }
            post {
                always {
                    warnings canComputeNew: false,
                             canResolveRelativePaths: false,
                             defaultEncoding: '',
                             excludePattern: '',
                             healthy: '',
                             includePattern: '',
                             messagesPattern: '',
                             parserConfigurations: [[parserName: 'PYLint', pattern: '**/flake8.log']],
                             unHealthy: ''
                    //WarningsReport('PYLint') // 'Flake8' fails..., consoleParsers does not produce any report...
                }
            }
        }

        stage('Security scanner') {
            steps {
                ToxEnvRun('bandit-report')
                script {
                    if (currentBuild.result == 'FAILURE') {
                        currentBuild.result = 'UNSTABLE'
                    }
               }
            }
            post {
               always {
                    HTMLReport("/tmp/bandit", 'index.html', 'Bandit report')
                }
            }
        }

        stage("Re-build Docker images") {
            when {
                anyOf {
                   branch 'master'
                   branch 'test'
                   buildingTag()
               }
            }
            steps {
                script {
                    def job_result = JenkinsBuildJob("${env.job_location}")
                    job_result_url = job_result.absoluteUrl
                }
            }
        }

    }

    post {
        failure {
            script {
                currentBuild.result = 'FAILURE'
            }
        }

        always  {
            script { //stage("Email notification")
                def build_status =  currentBuild.result
                build_status =  build_status ?: 'SUCCESS'
                def subject = """
New ${app_name} build in Jenkins@DEEP:\
${build_status}: Job '${env.JOB_NAME}\
[${env.BUILD_NUMBER}]'"""

                def body = """
Dear ${author_name},\n\n
A new build of '${app_name}' DEEP application is available in Jenkins at:\n\n
*  ${env.BUILD_URL}\n\n
terminated with '${build_status}' status.\n\n
Check console output at:\n\n
*  ${env.BUILD_URL}/console\n\n
and resultant Docker images rebuilding jobs at (may be empty in case of FAILURE):\n\n
*  ${job_result_url}\n\n

DEEP Jenkins CI service"""

                EmailSend(subject, body, "${author_email}")
            }
        }
    }
}
