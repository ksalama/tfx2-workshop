#!/bin/bash

SQ_CONNECTION_NAME=${1}

wget https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 -O cloud_sql_proxy
chmod +x cloud_sql_proxy
./cloud_sql_proxy -instances=${SQ_CONNECTION_NAME}=tcp:3306