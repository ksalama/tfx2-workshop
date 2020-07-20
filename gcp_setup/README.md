# Environment Setup

The following are the required steps to create a basic MLOps environment
to run the TFX workshop:

1. Enable the required GCP APIs
2. Build ML Container Image
3. Create AI Notebook Instance
4. Provision AI Platform Pipelines

You can use the `gcloud` command line interface in **Cloud Shell** to 
setup the environment

1. Start GCP Cloud Shell
2. Make sure that Cloud Shell is configured to use your project

    ```
    PROJECT_ID=[YOUR_PROJECT_ID]
    gcloud config set project $PROJECT_ID
    ```
   
3. Clone the repository

    ```
    git clone https://github.com/ksalama/tfx2-workshop.git
    cd gcp_setup 
    ```


## Enable GCP service APIs

In addition to [the services enabled by default](https://cloud.google.com/service-usage/docs/enabled-service),
the following additional services must be enabled:

```
gcloud services enable \
    cloudresourcemanager.googleapis.com \
    compute.googleapis.com \
    iam.googleapis.com \
    container.googleapis.com \
    containerregistry.googleapis.com \
    containeranalysis.googleapis.com \
    cloudbuild.googleapis.com 
    dataflow.googleapis.com \
    sqladmin.googleapis.com \
    notebooks.googleapis.com \
    ml.googleapis.com
```

## Build ML Container Image

we create container image with the required libraries to be used for 
experimentation in AI Platform Notebooks, and as a base image for 
AI Platform training jobs. 

This container image is derived from 
an existing [Deep Learning Container Image](gcr.io/deeplearning-platform-release/base-cpu:m42), 
and includes the packages in the [requirements.txt](requirements.txt), like TFX and KFP.

You can add the packages that you need for your ML tasks in the `requirements.txt` file, 
to make sure that the experimentation and training environment have the same runtime. 

```
IMAGE_NAME=ks-tfx-dev
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

gcloud builds submit --timeout 15m --tag ${IMAGE_URI} .
    
```

## Create AI Notebook Instance

You can provision an instance of AI Platform Notebooks using the 
[GCP Console](https://console.cloud.google.com/ai-platform/notebooks/create-instance).
Select **Custom Container** in the environment option, and type the 
container image URI: `gcr.io/[YOUR PROJECT ID]/[YOUR IMAGE NAME]:latest`

You can also use the `gcloud` command to create the AI Notebooks instance, as follows:

```
ZONE=[YOUR_ZONE]
INSTANCE_NAME=[YOUR_INSTANCE_NAME]

IMAGE_FAMILY="common-container"
IMAGE_PROJECT="deeplearning-platform-release"
INSTANCE_TYPE="n1-standard-4"
METADATA="proxy-mode=service_account,container=$IMAGE_URI"

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --image-family=$IMAGE_FAMILY \
    --machine-type=$INSTANCE_TYPE \
    --image-project=$IMAGE_PROJECT \
    --maintenance-policy=TERMINATE \
    --boot-disk-device-name=${INSTANCE_NAME}-disk \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --scopes=cloud-platform,userinfo-email \
    --metadata=$METADATA
```

## Provision AI Platform Pipelines

In order to create a hosted KFP on AI Platform pipelines, 
please follow the instruction in the documentation of 
[Setting up AI Platform Pipelines](https://cloud.google.com/ai-platform/pipelines/docs/setting-up).