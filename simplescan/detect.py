import os
import sys
from io import BytesIO
from object_detection import ObjectDetection
import onnxruntime
import onnx
import numpy as np
import tempfile
from pprint import pprint
import json
from PIL import Image, ImageDraw
from timeit import default_timer as timer
import requests
from requests.auth import HTTPDigestAuth
from notify import notify
from datetime import date,datetime

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

class Camera():
    def __init__(self, config):
        self.name = config['name']
        self.config = config
        self.objects = []
        self.is_file = False
    def capture(self):
          #pprint(r)
          #print(r.status_code)
        if 'file' in self.config:
            self.is_file = True
            return open(self.config['file'], "rb").read()
        elif 'user' in self.config:
            r = requests.get(self.config['uri'], auth=HTTPDigestAuth(self.config['user'], self.config['password']))
            return r.content
        else:
            r = requests.get(self.config['uri'])
            return r.content
    
def draw_bbox(image, p, color):
    w,h = image.size
    # {'probability': 0.60014141, 'tagId': 1, 'tagName': 'deer', 'boundingBox': {'left': 0.94383056, 'top': 0.82897264, 'width': 0.05527838, 'height': 0.18486874}}
    bbox = p['boundingBox']
    rect_start = (w * bbox['left'], h * bbox['top'])
    rect_end = (w * (bbox['left'] + bbox['width']), h * (bbox['top'] + bbox['height']))
    draw = ImageDraw.Draw(image)
    draw.rectangle((rect_start, rect_end), outline = color, width=4)
    del draw

def add_centers(predictions):
    for p in predictions:
        bbox = p['boundingBox']
        p['center'] = {}
        center = p['center']
        center['x'] = bbox['left'] + bbox['width'] / 2.0
        center['y'] = bbox['top'] + bbox['height'] / 2.0

def detect(cam, raw_image, od_model, config):
    threshold = config['detector'].getfloat('threshold')
    try:
        image = Image.open(BytesIO(raw_image))
        prediction_start = timer()
        predictions = od_model.predict_image(image)
        prediction_time = (timer() - prediction_start)
    except:
        print("%s=error:" % cam.name, sys.exc_info()[0])
        return 0
    # filter out lower predictions
    predictions = list(filter(lambda p: p['probability'] > threshold, predictions))
    #  lots of false positives for dog
    predictions = list(filter(lambda p: not(p['probability'] < .9 and p['tagName'] == 'dog'), predictions))
    predictions = list(filter(lambda p: not(p['probability'] < .9 and p['tagName'] == 'cat'), predictions))
    add_centers(predictions)
    if len(predictions) == 0:
        print('%s=[], ' % cam.name, end='', flush=True)
    else:
        print('')
        print(cam.name, end="=")
        pprint(predictions)
    # remove road
    if cam.name in ['west lawn','driveway']:
        predictions = list(filter(lambda p: not(p['center']['y'] < 0.25 and p['tagName'] == 'person'), predictions))
    elif cam.name == 'garage':
        predictions = list(filter(lambda p: not(p['boundingBox']['top'] < .03 and p['tagName'] == 'dog'), predictions))
    if len(predictions) == 0:
        if len( cam.objects ) > 0:
            print("  %s left %s" % (",".join(cam.objects), cam.name))
            cam.objects.clear()
        return prediction_time
    for p in predictions:
        p['camName'] = cam.name
    detected_objects = list(p['tagName'] for p in predictions)
    if detected_objects == cam.objects:
        print("  %s still near %s" % (",".join(cam.objects), cam.name) )
        return prediction_time
    # don't save file if we're reading from a file
    if not cam.is_file:
        save_dir = os.path.join(config['detector']['save-path'],date.today().strftime("%Y%m%d"))
        os.makedirs(save_dir,exist_ok=True)
        basename = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(detected_objects))
        image.save(basename + '.jpg')
        with open(basename + '.txt', 'w') as file:
            file.write(json.dumps(predictions))
        colors = config['colors']
        for p in predictions:
            draw_bbox(image, p, colors.get(p['tagName'], fallback='red'))
        image.save(basename + '-annotated.jpg')
    notify("%s near %s" % (",".join(detected_objects),cam.name), image, predictions, config)
    cam.objects = detected_objects
    return prediction_time
