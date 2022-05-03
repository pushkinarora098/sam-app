import joblib
import base64
import numpy as np
import json

from io import BytesIO
from PIL import Image

model_file = '/opt/ml/model'
model = joblib.load(model_file)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def lambda_handler(event, context):
    image_bytes = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    image = image.resize((28, 28))

    x = np.array(image)
    prediction = int(model.predict(x.reshape(1, -1))[0])

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": prediction,
                "array": x
            }
        ,cls = NumpyEncoder)
    }
