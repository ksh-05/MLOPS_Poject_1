pipeline {
    agent any
    stages{
        stage('Cloning Github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github repo to Jenkins............'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'git_token', url: 'https://github.com/ksh-05/MLOPS_Poject_1.git']])
                }
            }
        }

    }
}