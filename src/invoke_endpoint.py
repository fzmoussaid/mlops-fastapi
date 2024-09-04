import boto3
import json

ENDPOINT_NAME="classification-inference-endpoint"

def lambda_handler(event, context):
    input_data = { "input": ["0980" ]}
    client = boto3.client("sagemaker-runtime")
    json_data = json.dumps(input_data)
    response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                                    ContentType='application/json',
                                    Body=json.dumps(input_data))
    predictions = json.loads(response)
    return predictions.predicted_class