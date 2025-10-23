#!/bin/bash

# Google Cloud Run Deployment Script for Food Segmentation App
set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"food-segmentation-0001"}
SERVICE_NAME="food-segmentation"
REGION="africa-south1-a"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Starting deployment to Google Cloud Run..."
echo "Project ID: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
    echo "Please authenticate with gcloud first: gcloud auth login"
    exit 1
fi

# Set the project
echo "Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push docker image
echo "Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 4 \
    --timeout 900 \
    --max-instances 10 \
    --set-env-vars FLASK_ENV=production

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

echo ""
echo "Deployment completed successfully!"
echo "Service URL: ${SERVICE_URL}"
echo "You can view logs with: gcloud logs tail --follow --project=${PROJECT_ID} --resource-type=cloud_run_revision"
echo ""
echo "Test the deployment by visiting: ${SERVICE_URL}/health"
echo "Main app: ${SERVICE_URL}"
