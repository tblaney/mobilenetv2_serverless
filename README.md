# AWS Lambda Serverless Implementation of Image Classifier - MobileNetV2 (Pytorch)

Test Steps:
1. Download and Install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
2. Download and Install [SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
3. Download and Install [Docker](https://docs.docker.com/desktop/install/windows-install/)
4. Launch Docker
5. Configure AWS (enter in access keys, follow prompts)
```
aws configure
```
6. Clone this repo and cd to it
8. Edit event.json to specify image(s) URLs to be tested
7. 
```
sam build
```
8. Test from local call
```
sam local invoke PyTorchInferenceFunction --event event.json
```
10. (Optional) Deploy
```
sam deploy --guided
```
9. (Optional - if deployed) Delete 
```
aws cloudformation delete-stack --stack-name pytorch-inference-docker-lambda
```
