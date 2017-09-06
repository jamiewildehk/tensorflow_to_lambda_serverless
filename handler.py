import os
import sys
import json
import ConfigParser
import base64
import boto3
import re

"""
This is needed so that the script running on AWS will pick up the pre-compiled dependencies
from the vendored folder
"""
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendored"))
"""
Now that the script knows where to look, we can safely import our objects
"""
from tf_sdd_box import TensorFlowBoxingModel
"""
Declare here global objects living across requests
"""
# use Pythonic ConfigParser to handle settings
Config = ConfigParser.ConfigParser()
Config.read(HERE + '/settings.ini')
# instantiate the tf_model in "prediction mode"

def extract_s3_path(s3path):
    Result = {}
    s3path = s3path.replace("s3://","")
    tmppath =  s3path.split("/")
    Result["bucket"] = tmppath[0]
    Result["key"] = tmppath[1]
    print Result
    return Result

def s3_download_config(s3_client,config,destination):
    dest_path = destination + config["key"]
    if not os.path.isfile(dest_path):
        s3_client.download_file(config["bucket"], config["key"], dest_path)
        print ('%s loaded!' % config["key"])
    else:
        print ('%s already loaded!' % config["key"])
    return


s3_client = boto3.client('s3')

s3_download_config(s3_client,extract_s3_path(Config.get('remote', 'REMOTE_MODEL_META')),Config.get('model', 'LOCAL_MODEL_FOLDER') )
s3_download_config(s3_client,extract_s3_path(Config.get('remote', 'REMOTE_MODEL_INDEX')),Config.get('model', 'LOCAL_MODEL_FOLDER') ) 
s3_download_config(s3_client,extract_s3_path(Config.get('remote', 'REMOTE_MODEL_DATA')),Config.get('model', 'LOCAL_MODEL_FOLDER') )

tf_model = TensorFlowBoxingModel(Config, is_training=False)
# just print a message so we can verify in AWS the loading of dependencies was correct

print "loaded done!"



def validate_input(input_val):
    """
    Helper function to check if the input is indeed a float

    :param input_val: the value to check
    :return: the floating point number if the input is of the right type, None if it cannot be converted
    """
    try:
        float_input = float(input_val)
        return float_input
    except ValueError:
        return None


def get_image_from_request_body(event):
    """
    Helper function to convert the request body (B64) into an image in memory

    :param event: the event as input in the Lambda function
    :return: the image
    """
    base64_image = event.body
    img_data = base64.b64decode(base64_image)
    img = cv2.imdecode(np.fromstring(img_data, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def return_lambda_gateway_response(code, body):
    """
    This function wraps around the endpoint responses in a uniform and Lambda-friendly way

    :param code: HTTP response code (200 for OK), must be an int
    :param body: the actual content of the response
    """
    return {"statusCode": code, "body": json.dumps(body)}


def predict(event, context):
    """
    This is the function called by AWS Lambda, passing the standard parameters "event" and "context"
    When deployed, you can try it out pointing your browser to

    {LambdaURL}/{stage}/predict?x=2.7

    where {LambdaURL} is Lambda URL as returned by serveless installation and {stage} is set in the
    serverless.yml file.

    """
    try:
        img = get_image_from_request_body(event)
        if img:
            value = tf_model.predict(img)
        else:
            raise "Input request has invalid type: base64 body expected"
    except Exception as ex:
        error_response = {
            'error_message': "Unexpected error",
            'stack_trace': str(ex)
        }
        return return_lambda_gateway_response(503, error_response)

    return return_lambda_gateway_response(200, {'value': value})



