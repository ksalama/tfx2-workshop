# TFX 0.2 Workshop

This repository is a workshop for implementing a Machine Learning (ML) training pipeline using TensorFlow Extended (TFX)
and AI Platform Pipelines. The ML pipeline uses Google Cloud Platform (GCP) managed services.

This workshop uses **Python 3.6** requires the following packages:

* TensorFlow 2.2
* TFX  2.1.4
* KFP SDK 0.5.1

## MLOps environment on GCP

The core services in the environment are:

1. ML experimentation and development - AI Platform Notebooks
2. Scalable, serverless model training - AI Platform Training
3. Scalable, serverless model serving - AI Platform Prediction
4. Distributed data processing - Dataflow
5. Analytics data warehouse - BigQuery
6. Artifact store - Google Cloud Storage
7. Machine learning pipelines - TensorFlow Extended (TFX) and Kubeflow Pipelines (KFP)
8. Machine learning metadata management - ML Metadata on Cloud SQL
9. CI/CD tooling - Cloud Build

## Dataset

The dataset used in these labs is the **UCI Adult Dataset**: https://archive.ics.uci.edu/ml/datasets/adult.

It is a classification dataset, where the task is to predict whether income exceeds 50K USD per year based on census data. 
It is also known as "Census Income" dataset.

We load the dataset from [Cloud Stroage](gs://cloud-samples-data/ml-engine/census/data) to BigQuery, 
using the [load-bq.ipynb Notebook](gcp_setup/load-bq.ipynb),
Then we use the data in BigQuery for Exploratory Data Analysis and ML Experimentation.

## Setup


[gcp_setup](gcp_setup/README.md) explains the required steps to create a basic MLOps environment
to run the TFX workshop, which includes:

1. Enable the required GCP APIs
2. Build ML Container Image
3. Create AI Notebook Instance
4. Provision AI Platform Pipelines






