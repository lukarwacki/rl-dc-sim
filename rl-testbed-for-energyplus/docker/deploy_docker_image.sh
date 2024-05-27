#!/bin/bash

# Variables from your previous script
RG_NAME=data-science-dev
RG_LOCATION=westeurope
ACR_NAME=cgdevcontainerregistry
ACI_CONTAINER_NAME=rl-experiments-pepijn-1
DOCKER_IMAGE_NAME=rl-testbed-for-energyplus9-5-0
SSH_PORT=2222
ACI_STORAGE_ACCOUNT_NAME=cgstoragedsdev001
ACI_STORAGE_CONTAINER_NAME=pepijn-research
ACI_MEMORY=16

# Login to Azure
az login

# Login to the Azure Container Registry
az acr login --name $ACR_NAME

# Uncomment the following lines if you want push a local docker image to the Azure Container Registry before
# deploying the Azure Container Instance

# # Give you local docker image a tag that matches the Azure Container Registry
# docker tag $DOCKER_IMAGE_NAME:latest $ACR_NAME.azurecr.io/$DOCKER_IMAGE_NAME:latest
# # Push the image to the Azure Container Registry
# docker push  $ACR_NAME.azurecr.io/$DOCKER_IMAGE_NAME:latest


# Export the connection string as an environment variable. The following 'az storage share create' command
# references this environment variable when creating the Azure file share.
echo "Exporting storage connection string: $ACI_STORAGE_ACCOUNT_NAME"
export AZURE_STORAGE_CONNECTION_STRING=`az storage account show-connection-string --resource-group $RG_NAME --name $ACI_STORAGE_ACCOUNT_NAME --output tsv`

# Environment variable (AZURE_STORAGE_ACCESS_KEY) to be set at client and with Server
# Export the access keyas an environment variable
echo "Exporting storage keys: $ACI_STORAGE_ACCOUNT_NAME"
export AZURE_STORAGE_ACCESS_KEY=$(az storage account keys list --resource-group $RG_NAME --account-name $ACI_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)

# Run the following command to get the username and password of the container registry
az acr credential show --resource-group $RG_NAME --name $ACR_NAME

# Create an Azure Container Instance using the image from the Azure Container Registry
az container create \
    --resource-group $RG_NAME \
    --name $ACI_CONTAINER_NAME \
    --image $ACR_NAME.azurecr.io/$DOCKER_IMAGE_NAME:latest \
    --ports $SSH_PORT\
    --azure-file-volume-account-name $ACI_STORAGE_ACCOUNT_NAME \
    --azure-file-volume-account-key $AZURE_STORAGE_ACCESS_KEY \
    --memory $ACI_MEMORY \
    --restart-policy Always \
    --environment-variables AZURE_STORAGE_ACCESS_KEY=$AZURE_STORAGE_ACCESS_KEY \
        AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING \


