## Model Deployment using AWS Sagemaker BYOM (Bring Your Own Model)

- This repository is an example of how an ML model can be deployed in AWS.
- It contains the source code for training and evaluating the model using Pytorch, and a small FastAPI app that we dockerize and upload to AWS ECR.

Note:

- The standard Torch and Torch vision libraries can be very large. For a smaller docker image, we use the light weight torch-cpu and torch-vision-cpu instead.
- We deploy a serverless endpoint, which means the instance only runs when the endpoint is invoked, this could induce cold starts. If you want to avoid cold starts, please use realtime instances instead. Check the repo `mlops-IaC` for how to deploy the infrastructure.
