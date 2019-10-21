import json
import cv2
import numpy as np
import requests
import base64
from procMaimai import processImg

'''
Lambda handler function
'''
def ras_cry(event, context):
    # Decode json string payload
    body = event['body']
    url = json.loads(body)['url']

    # Get image from url, write file to /tmp/
    filename = "/tmp/" + url.split('/')[-1]
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)

    # Call processImg
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
    result = processImg(img)

    # Return result as json
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }

