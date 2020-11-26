import os
import sys
from io import BytesIO
from object_detection import ObjectDetection
from utils import bb_intersection_over_union,draw_bbox
import numpy as np
from pprint import pprint
import json
from timeit import default_timer as timer
import requests
from requests.auth import HTTPDigestAuth
from notify import notify
from datetime import date,datetime,timedelta
from PIL import UnidentifiedImageError
import traceback

class Camera():
    def __init__(self, config, excludes):
        self.name = config['name']
        self.config = config
        self.objects = []
        self.prev_predictions = {}
        self.is_file = False
        self.vehicle_check = config.getboolean('vehicle_check', False)
        self.excludes = excludes
        self.session = requests.Session()
        if 'user' in self.config:
            self.session.auth = HTTPDigestAuth(self.config['user'], self.config['password'])
        self.capture_async = self.config.getboolean('async', False)
        self.prior_image = None
        self.prior_time = None

    def capture(self):
        self.image = None
        for _ in range(0,2):
            if not self.image is None:
                break
            if 'file' in self.config:
                self.is_file = True
                raw_image = open(self.config['file'], "rb").read()
            else:
                r = self.session.get(self.config['uri'])
                raw_image = r.content
            try:
                self.image = Image.open(BytesIO(raw_image))
            except UnidentifiedImageError as e:
                self.image = None
                #print("%s=error:" % self.name, sys.exc_info()[0], end=" ")
        return self
    
def add_centers(predictions):
    for p in predictions:
        bbox = p['boundingBox']
        p['center'] = {}
        center = p['center']
        center['x'] = bbox['left'] + bbox['width'] / 2.0
        center['y'] = bbox['top'] + bbox['height'] / 2.0

def detect(cam, od_model, vehicle_model, config, st):
    threshold = config['detector'].getfloat('threshold')
    smartthings = config['smartthings']
    image = cam.image
    if image == None:
        print("{}=[X]".format(cam.name), end=", ")
        return 0
    prediction_start = timer()
    try:
        predictions = od_model.predict_image(image)
    except OSError as e:
        print("%s=error:" % cam.name, sys.exc_info()[0])
        return 0
    vehicle_predictions = []
    if cam.vehicle_check:
        vehicle_predictions = vehicle_model.predict_image(image)
        # include all vehicle predictions for now
        predictions += vehicle_predictions
    prediction_time = (timer() - prediction_start)
    # filter out lower predictions
    predictions = list(filter(lambda p: p['probability'] > config['thresholds'].getfloat(p['tagName'], threshold) or (p['tagName'] in cam.objects and p['probability'] > 0.4), predictions))
    if len(predictions) == 0:
        print('%s=[], ' % cam.name, end='')
    else:
        print(cam.name, end="=")
        print("[" + ",".join( "{}:{:.2f}".format(p['tagName'],p['probability']) for p in predictions ), end=', ')
    #  lots of false positives for cat and dog
    #for label in ['dog','cat']:
    #    if not label in cam.objects:
    #        predictions = list(filter(lambda p: not(p['probability'] < .9 and p['tagName'] == label), predictions))
    add_centers(predictions)
    # remove road
    if cam.name in ['front lawn','driveway']:
        predictions = list(filter(lambda p: not(p['center']['y'] < 0.25 and (p['tagName'] in ['vehicle','person'])), predictions))
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
        cam.prior_image = image
        cam.prior_time = datetime.now()
        return prediction_time
    for p in predictions:
        p['camName'] = cam.name
    detected_objects = list(p['tagName'] for p in predictions)
    # Always open garage door, we can call this many times
    if smartthings:
        if 'cat' in detected_objects and cam.name != 'garage':
            print("Cracking open garage door for cat")
            r = requests.get(smartthings['crack_garage'])
            if r.json()['result'] != "OK":
                print("Failed to crack open garage door for cat")
                print(r.text)

    colors = config['colors']
    uniq_predictions = []
    for p in predictions:
        prev_class = cam.prev_predictions.setdefault(p['tagName'],[])
        prev_time = cam.prev_predictions.get(p['tagName'] + "-time", datetime.utcfromtimestamp(0))
        if datetime.now() - prev_time > timedelta(minutes=60) and len(prev_class) > 0: # clear if we haven't seen this class in 15 minutes on this camera
            prev_class.clear()
            print('Cleared previous predictions for %s on %s' % (p['tagName'], cam.name))
        cam.prev_predictions[p['tagName'] + '-time'] = datetime.now()
        skip = False
        for prev_box in prev_class:
            iou = bb_intersection_over_union(prev_box,p['boundingBox'])
            if iou > 0.5:
                print("%s iou=%f" % (p['tagName'], iou), end=", ")
                skip = True
                break
        if skip:
            p['uniq'] = False
        else:
            p['uniq'] = True
            prev_class.append(p['boundingBox'])
            uniq_predictions.append(p)

    if len(uniq_predictions):
        # don't save file if we're reading from a file
        if not cam.is_file:
            if cam.prior_image:
                basename = os.path.join(save_dir,cam.prior_time.strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(detected_objects) + "-prior")
                cam.prior_image.save(basename + '.jpg')
                cam.prior_image = None
            basename = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(detected_objects))
            image.save(basename + '.jpg')
            with open(basename + '.txt', 'w') as file:
                file.write(json.dumps(predictions))
        for p in predictions:
            draw_bbox(image, p, colors.get(p['tagName'], fallback='red'))
        if not cam.is_file:
            image.save(basename + '-annotated.jpg')

    uniq_objects = list(p['tagName'] for p in uniq_predictions)

    # Only notify deer if not seen
    if 'deer' in uniq_objects:
        st.deer_alert()

    print("],", end=' ', flush=True)
    if len(uniq_objects):
        if cam.name == 'driveway':
            message = "%s in %s" % (",".join(detected_objects),cam.name)
        elif cam.name == 'shed':
            message = "%s in front of garage" % ",".join(detected_objects)
        else:
            message = "%s near %s" % (",".join(detected_objects),cam.name)
        notify(message, image, predictions, config, st)
        cam.objects = detected_objects

    return prediction_time
