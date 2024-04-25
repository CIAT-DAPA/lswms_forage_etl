// Define an empty map for storing remote SSH connection parameters
def remote = [:]

pipeline {
    agent any

    environment {
        server_name = credentials('forage_etl_name')
        server_host = credentials('forage_etl_host')
        ssh_key = credentials('forage_etl_devops')
    }

    stages {
        stage('Ssh to connect server') {
            steps {
                script {
                    // Set up remote SSH connection parameters
                    remote.allowAnyHosts = true
                    remote.identityFile = ssh_key
                    remote.user = ssh_key_USR
                    remote.name = server_name
                    remote.host = server_host
                }
            }
        }
        stage('Download latest release') {
            steps {
                script {
                    try {
                        echo "Downloading latest release..."
                        sshCommand remote: remote, command: """
                            cd /opt/etlwms
                            rm -rf src/*
                            rm -rf src
                            curl -LOk https://github.com/CIAT-DAPA/lswms_forage_etl/releases/latest/download/releaseForageEtl.zip
                            unzip -o releaseForageEtl.zip
                            rm -fr releaseForageEtl.zip
                        """
                        echo "Download completed."
                    } catch (Exception e) {
                        // Log any errors that occur during download
                        echo "Failed to download latest release: ${e.getMessage()}"
                        error "Failed to download latest release"
                    }
                }
            }
        }

        stage('Copy secret files to remote server') {
            steps {
                script {
                     try {
                        echo "Copying secret files to remote server..."
                    // Define the remote directory where the files will be copied
                    def remoteDirectory = '/opt/etlwms/src'

                    // Retrieve the content of the secret files from Jenkins credentials
                    withCredentials([file(credentialsId: 'forage_etl_data_file', variable: 'DATA_FILE'),
                                     file(credentialsId: 'forage_etl_gee_file', variable: 'GEE_FILE')]) {

                        // Use SCP to copy the secret files to the remote server
                        sh 'scp -o StrictHostKeyChecking=no -i ' + ssh_key + ' $DATA_FILE ' + ssh_key_USR + '@' + server_host + ':' + remoteDirectory + '/data.json'
                        sh 'scp -o StrictHostKeyChecking=no -i ' + ssh_key + ' $GEE_FILE ' + ssh_key_USR + '@' + server_host + ':' + remoteDirectory + '/private_key.json'
                                     }
                        } catch (Exception e) {
                        // Log any errors that occur during download
                        echo "Failed to copy secret files: ${e.getMessage()}"
                        error "Failed to copy secret files"
                    }
                    }
                }
            }
    }

    post {
        failure {
            script {
                echo 'Fail: ETL deployment unsuccessful'
            }
        }

        success {
            script {
                echo 'Success: ETL deployment successful'
            }
        }
    }
}