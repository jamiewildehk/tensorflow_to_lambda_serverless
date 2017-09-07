import os
import sys
import json
import ConfigParser
import base64
import boto3
import re
import StringIO
import uuid
import time


"""
This is needed so that the script running on AWS will pick up the pre-compiled dependencies
from the vendored folder
"""
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendored"))
sys.path.insert(0, os.path.join(HERE, "vendored/google"))
sys.path.insert(0, os.path.join(HERE, "vendored/google/protobuf"))
sys.path.insert(0, os.path.join(HERE, "vendored/PIL"))

from PIL import Image
from io import BytesIO
import numpy as np

this_process = None

print sys.path

s3_client = boto3.client('s3')


def extract_s3_path(s3path):
    Result = {}
    s3path = s3path.replace("s3://","")
    tmppath =  s3path.split("/")
    Result["bucket"] = tmppath[0]
    Result["key"] = tmppath[1]
    return Result

def s3_download(s3_client,config,destination,dest_filename=None):
    dest_path = destination + config["key"]
    if dest_filename:
        dest_path = destination + dest_filename
    print config

    if not os.path.isfile(dest_path):
        s3_client.download_file(config["bucket"], config["key"], dest_path)
        print ('%s loaded!' % dest_path)
    else:
        print ('%s already loaded!' % dest_path)
        time.sleep(1)
    return dest_path



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
    
    base64_image = event['body']
    img = Image.open(BytesIO(base64.b64decode(base64_image)))
    imdata = np.asarray(img)    
    img = Image.fromarray(np.roll(imdata, 1, axis=-1))
    return img


def return_lambda_gateway_response(code, body):
    """
    This function wraps around the endpoint responses in a uniform and Lambda-friendly way

    :param code: HTTP response code (200 for OK), must be an int
    :param body: the actual content of the response
    """
    return {"statusCode": code, "body": json.dumps(body)}

def format_tf_to_dict(npyv):
    """
    This function convert the given TF input (numpy value) into a py dict

    :param npyv: the numbpy value 
    """
    result = {"faces":{
        "rect" : []
    }}
    for i, item in enumerate(npyv):
        if i == 1:
            for j, rect in enumerate(item):
                rect = rect.tolist()
                result["faces"]["rect"].append({
                    "ymin" : str(rect[0]),
                    "xmin" : str(rect[1]),
                    "ymax" : str(rect[2]),
                    "xmax" : str(rect[3])
                })
    return result
           
def image_from_s3(event):
    record = event['Records'][0]
    s3_param = {}
    tmppath =  "/tmp/"
    s3_param["bucket"] = record['s3']['bucket']['name']
    s3_param["key"] = record['s3']['object']['key']
    filename =  s3_param["key"].split("/")
    filename = filename[-1]

    tmppath = s3_download(s3_client,s3_param,tmppath,filename)
    
    img = Image.open(tmppath)
    if os.path.exists(tmppath):
        os.remove(tmppath)
    imdata = np.asarray(img)    
    img = Image.fromarray(np.roll(imdata, 1, axis=-1))
    return img


def upload_json(bucket,key,o_json,filename):
    tmp_json_path = '/tmp/{}{}' + filename
    print tmp_json_path
    print key
    print bucket
    print o_json
    
    with open(tmp_json_path, 'w') as outfile:
        json.dump(o_json,outfile)
    s3_client.upload_file(tmp_json_path,bucket,key)
    return



def predict(event, context):
    """
    This is the function called by AWS Lambda, passing the standard parameters "event" and "context"
    When deployed, you can try it out pointing your browser to

    {LambdaURL}/{stage}/predict?x=2.7

    where {LambdaURL} is Lambda URL as returned by serveless installation and {stage} is set in the
    serverless.yml file.

    """


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


    

    s3_download(s3_client,extract_s3_path(Config.get('remote', 'REMOTE_MODEL_META')),Config.get('model', 'LOCAL_MODEL_FOLDER') )
    s3_download(s3_client,extract_s3_path(Config.get('remote', 'REMOTE_MODEL_INDEX')),Config.get('model', 'LOCAL_MODEL_FOLDER') ) 
    s3_download(s3_client,extract_s3_path(Config.get('remote', 'REMOTE_MODEL_DATA')),Config.get('model', 'LOCAL_MODEL_FOLDER') )

    tf_model = TensorFlowBoxingModel(Config, is_training=False)

    this_process = {
        "source" : "",
        "key" : "",
        "bucket" : "",
        "event": event,
        "jsonkey" : ""
    }

    try:
        img = None
        if event['Records']:
            #s3 triggered
            img = image_from_s3(event)
            this_process["source"] = "s3"
            key = event['Records'][0]['s3']['object']['key']
            key = key.split("/")
            this_process["key"] = key[-1]

            bucket = event['Records'][0]['s3']['bucket']['name']
            bucket = bucket.split("/")
            this_process["bucket"] = bucket[-1]

            jsonkey = this_process["key"].split(".")
            this_process["jsonkey"] =  jsonkey[0] + ".json"
        else:
            #direct base64 as body        
            img = get_image_from_request_body(event)
            this_process["source"] = "http"
        if img:
            npy_value = tf_model.predict(img)       
            value = format_tf_to_dict(npy_value)
            #process output to s3 to complete the pipeline
            print this_process
            if this_process["source"] == "s3" :
                upload_json(this_process["bucket"],"json/cnn/" + this_process["jsonkey"],value,this_process["jsonkey"])
                
        else:
            raise "Input request has invalid type: base64 body expected"   
    except Exception as ex:
        error_response = {
            'error_message': "Unexpected error",
            'stack_trace': str(ex)
        }
        print ex
        return return_lambda_gateway_response(503, error_response)

    return return_lambda_gateway_response(200, {'value': value})



