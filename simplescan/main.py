# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import os
import sys
import onnxruntime
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection import ObjectDetection
import tempfile
import requests
from requests.auth import HTTPDigestAuth
from PIL import Image
from io import BytesIO
import configparser
from notify import notify
from datetime import date,datetime
import json
from pprint import pprint
from timeit import default_timer as timer
import asyncio

SAVE_DIRECTORY = '/mnt/elements/capture/'

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        super(ONNXRuntimeObjectDetection, self).__init__(labels)
        model = onnx.load(model_filename)
        with tempfile.TemporaryDirectory() as dirpath:
            temp = os.path.join(dirpath, os.path.basename(model_filename))
            model.graph.input[0].type.tensor_type.shape.dim[-1].dim_param = 'dim1'
            model.graph.input[0].type.tensor_type.shape.dim[-2].dim_param = 'dim2'
            onnx.save(model, temp)
            self.session = onnxruntime.InferenceSession(temp)
        self.input_name = self.session.get_inputs()[0].name
        self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
        inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))

        if self.is_fp16:
            inputs = inputs.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: inputs})
        return np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)

def draw_bbox(image, p, color):
    w,h = image.size
    # {'probability': 0.60014141, 'tagId': 1, 'tagName': 'deer', 'boundingBox': {'left': 0.94383056, 'top': 0.82897264, 'width': 0.05527838, 'height': 0.18486874}}
    bbox = p['boundingBox']
    rect_start = (w * bbox['left'], h * bbox['top'])
    rect_end = (w * (bbox['left'] + bbox['width']), h * (bbox['top'] + bbox['height']))
    draw = ImageDraw.Draw(image)
    draw.rectangle((rect_start, rect_end), outline = color, width=4)
    del draw

async def main(options):
    config = configparser.ConfigParser()
    config.read("config.txt")
    detector_config = config['detector']
    colors = config['colors']

    # Load labels
    with open(detector_config['labelfile-path'], 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = ONNXRuntimeObjectDetection(detector_config['onnx-file'], labels)
   
    cams=[]
    i = 0
    while "cam%d" % i in config.sections():
        cams.append(config["cam%d" % i])
        i += 1
    print("Configured %i cams" % i)
    threshold = config['detector'].getfloat('threshold')
    cam_state = {}
    
    while True:
      start_time = timer()
      prediction_time = 0.0
      for cam in cams:
        cam_name = cam['name']
        print("Checking %s" % cam['name'])
        if 'file' in cam:
            raw_image = open(cam['file'], "rb").read()
        elif 'user' in cam:
            r = requests.get(cam['uri'], auth=HTTPDigestAuth(cam['user'], cam['password']))
            raw_image = r.content
        else:
            r = requests.get(cam['uri'])
            raw_image = r.content
        try:
            image = Image.open(BytesIO(raw_image))
            prediction_start = timer()
            predictions = od_model.predict_image(image)
            prediction_time += (timer() - prediction_start)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            pprint(r)
            print(r.status_code)
            print(len(raw_image))
            continue
        predictions = list(filter(lambda p: p['probability'] > threshold, predictions))
        # remove road
        predictions = list(filter(lambda p: not(p['boundingBox']['top'] < .16 and p['tagName'] == 'person'), predictions))
        if len(predictions) == 0:
            if len( cam_state.get( cam_name, []) ) > 0:
                print("  %s left %s" % (",".join(cam_state[cam['name']]), cam['name']))
                cam_state[cam['name']] = []
            continue
        for p in predictions:
            p['camName'] = cam_name
        pprint(predictions)
        detected_objects = list(p['tagName'] for p in predictions)
        current_objects = cam_state.get(cam['name'], "")
        if detected_objects == current_objects:
            print("  %s still near %s" % (",".join(current_objects), cam_name) )
            continue
        save_dir = os.path.join(SAVE_DIRECTORY,date.today().strftime("%Y%m%d"))
        os.makedirs(save_dir,exist_ok=True)
        basename = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + cam_name + "-" + "_".join(detected_objects))
        image.save(basename + '.jpg')
        with open(basename + '.txt', 'w') as file:
            file.write(json.dumps(predictions))
        for p in predictions:
            draw_bbox(image, p, colors.get(p['tagName'], fallback='red'))
        image.save(basename + '-annotated.jpg')
        await notify("%s near %s" % (",".join(detected_objects),cam['name']), image, predictions, config)
        cam_state[cam['name']] = detected_objects
      end_time = timer()
      print('Tour completed in %.2fs, spent %.2fs predicting' % ( (end_time - start_time), prediction_time ) )
    
if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:]))
