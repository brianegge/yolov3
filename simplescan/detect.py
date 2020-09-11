import os
import sys
from io import BytesIO
from object_detection import ObjectDetection
from utils import bb_intersection_over_union
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
from notify import notify, get_st_mode
from datetime import date,datetime,timedelta
from PIL import UnidentifiedImageError
import traceback

class ONNXRuntimeObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels, prob_threshold=0.10, scale=4):
        super(ONNXRuntimeObjectDetection, self).__init__(labels, prob_threshold=prob_threshold, scale=scale)
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
    def __init__(self, config, excludes):
        self.name = config['name']
        self.config = config
        self.objects = []
        self.prev_predictions = {}
        self.is_file = False
        self.last_st_mode = None
        self.vehicle_check = config.getboolean('vehicle_check', False)
        self.excludes = excludes

    def capture(self, session):
        if 'file' in self.config:
            self.is_file = True
            return open(self.config['file'], "rb").read()
        elif 'user' in self.config:
            r = session.get(self.config['uri'], auth=HTTPDigestAuth(self.config['user'], self.config['password']))
            return r.content
        else:
            r = session.get(self.config['uri'])
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

def detect(cam, raw_image, od_model, vehicle_model, config):
    threshold = config['detector'].getfloat('threshold')
    smartthings = config['smartthings']
    image = None
    for _ in range(2):
        try:
            image = Image.open(BytesIO(raw_image))
        except UnidentifiedImageError as e:
            print("%s=error:" % cam.name, sys.exc_info()[0])
            #track = traceback.format_exc()
            #print(e)
            #print(track)
            continue
        else:
            break
    if image == None:
        print("Can't capture %s camera" % cam.name)
        return 0
    prediction_start = timer()
    predictions = od_model.predict_image(image)
    vehicle_predictions = []
    if cam.vehicle_check:
        vehicle_predictions = vehicle_model.predict_image(image)
        vehicle_predictions = list(filter(lambda p: p['probability'] > 0.5, vehicle_predictions))
    prediction_time = (timer() - prediction_start)
    # filter out lower predictions
    predictions = list(filter(lambda p: p['probability'] > threshold, predictions))
    # include all vehicle predictions for now
    predictions += vehicle_predictions
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
    if cam.name in ['front lawn','driveway']:
        predictions = list(filter(lambda p: not(p['center']['y'] < 0.25 and p['tagName'] == 'person'), predictions))
    #elif cam.name == 'shed':
    #    # flag pole on far right
    #    predictions = list(filter(lambda p: not(p['center']['x'] > 0.93 and p['center']['y'] < 0.396021495), predictions))
    #elif cam.name == 'deck':
    #    # flag pole 
    #    predictions = list(filter(lambda p: not(p['center']['x'] < 0.3031482 and p['center']['y'] < 0.28), predictions))
    for p in predictions:
        if p['tagName'] in cam.excludes:
            for e in cam.excludes[p['tagName']]:
                iou = bb_intersection_over_union(e,p['boundingBox'])
                if iou > 0.5:
                    print("Static exclude {} at {}".format(p['tagName'],p['boundingBox']))
                    p['ignore'] = True
    predictions = list(filter(lambda p: not('ignore' in p), predictions))

    save_dir = os.path.join(config['detector']['save-path'],date.today().strftime("%Y%m%d"))
    os.makedirs(save_dir,exist_ok=True)
    if len(predictions) == 0:
        if len( cam.objects ) > 0:
            print("  %s departed %s" % (",".join(cam.objects), cam.name))
            basename = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(cam.objects) + "-departed")
            image.save(basename + '.jpg')
            cam.objects.clear()
        return prediction_time
    for p in predictions:
        p['camName'] = cam.name
    detected_objects = list(p['tagName'] for p in predictions)
    if detected_objects == cam.objects:
        print("  %s still near %s" % (",".join(cam.objects), cam.name) )
    # Always open garage door, we can call this many times
    if smartthings:
        if 'cat' in detected_objects:
            print("Cracking open garage door for cat")
            r = requests.get(smartthings['crack_garage'])
            if r.json()['result'] != "OK":
                print("Failed to crack open garage door for cat")
                print(r.text)

    #    return prediction_time
    colors = config['colors']
    # check if this class has been reported for this location today
    st_mode = get_st_mode(config)
    # clear out any prior predictions when mode changes
    if cam.last_st_mode != st_mode:
        cam.last_st_mode = st_mode
        if bool(cam.prev_predictions):
            print("Clearing out previous predictons for cam %s now that we are in mode %s" % (cam.name, st_mode))
        cam.prev_predictions = {}
    uniq_predictions = []
    for p in predictions:
        prev_class = cam.prev_predictions.setdefault(p['tagName'],[])
        prev_time = cam.prev_predictions.get(p['tagName'] + "-time", datetime.utcfromtimestamp(0))
        if datetime.now() - prev_time > timedelta(minutes=15) and len(prev_class) > 0: # clear if we haven't seen this class in 15 minutes on this camera
            cam.prev_predictions[p['tagName']] = []
            prev_class.clear()
            print('Cleared previous predictions for %s on %s' % (p['tagName'], cam.name))
        cam.prev_predictions[p['tagName'] + '-time'] = datetime.now()
        skip = False
        for prev_box in prev_class:
            iou = bb_intersection_over_union(prev_box,p['boundingBox'])
            if iou > 0:
                print("IOU %s=%.3f" % (p['tagName'],iou), prev_box)
            if iou > 0.5:
                print("Already notified %s with iou %f" % (p['tagName'], iou))
                skip = True
                break
        if not skip:
            prev_class.append(p['boundingBox'])
            uniq_predictions.append(p)

    if len(uniq_predictions):
        # don't save file if we're reading from a file
        if not cam.is_file:
            basename = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(detected_objects))
            image.save(basename + '.jpg')
            with open(basename + '.txt', 'w') as file:
                file.write(json.dumps(predictions))
        for p in predictions:
            draw_bbox(image, p, colors.get(p['tagName'], fallback='red'))
        if not cam.is_file:
            image.save(basename + '-annotated.jpg')

    predictions = uniq_predictions
    detected_objects = list(p['tagName'] for p in predictions)

    # Only notify deer if not seen
    if smartthings:
        if 'deer' in detected_objects:
            print("Deer alert!")
            r = requests.get(smartthings['deer_alert'])
            if r.json()['result'] != "OK":
                print("Failed to crack open garage door for cat")
                print(r.text)

    if len(detected_objects):
        notify("%s near %s" % (",".join(detected_objects),cam.name), image, predictions, config)

    cam.objects = detected_objects
    return prediction_time
