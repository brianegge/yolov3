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
from PIL import Image
import cv2
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
        self.age = 0

    def capture(self):
        self.image = None
        for _ in range(0,2):
            if not self.image is None:
                break
            if 'file' in self.config:
                self.is_file = True
                self.image = cv2.imread(self.config['file'])
            else:
                resp = self.session.get(self.config['uri'], stream=True).raw
                try:
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    if len(image):
                        self.image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    else:
                        self.image = None
                    # self.image = Image.open(BytesIO(raw_image))
                except (cv2.error, UnidentifiedImageError) as e:
                    self.image = None
                    print("%s=error:" % self.name, sys.exc_info()[0], end=" ")
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
    if image is None:
        print("{}=[X]".format(cam.name), end=", ")
        return 0
    prediction_start = timer()
    try:
        predictions = od_model.predict_image(image)
    except OSError as e:
        print("%s=error:" % cam.name, sys.exc_info()[0])
        return 0
    cam.age = cam.age + 1
    vehicle_predictions = []
    if cam.vehicle_check and vehicle_model is not None:
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
    add_centers(predictions)
    # remove road
    road = []
    if cam.name in ['front lawn','driveway']:
        not_road = []
        for p in predictions:
            road_y = 0.31 + 0.17 * p['center']['x']
            if p['center']['y'] < road_y and (p['tagName'] in ['vehicle','person']):
                p['road_y'] = road_y
                road.append(p)
            else:
                not_road.append(p)
        predictions = not_road
        #if len(road) > 0:
        #    print("objects on road: ", end="")
        #    pprint(road)
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
            cv2.imwrite(basename + '.jpg', image)
            cam.objects.clear()
        cam.prior_image = image
        cam.prior_time = datetime.now()
        return prediction_time
    for p in predictions:
        p['camName'] = cam.name
    detected_objects = list(p['tagName'] for p in predictions)
    # Always open garage door, we can call this many times
    if smartthings:
        if 'cat' in detected_objects and cam.name != 'garage' and 'crack_garage' in smartthings:
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

    im_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im_pil)
    if len(uniq_predictions):
        # don't save file if we're reading from a file
        if not cam.is_file:
            if cam.prior_image is not None:
                basename = os.path.join(save_dir,cam.prior_time.strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(detected_objects) + "-prior")
                cv2.imwrite(basename + '.jpg', cam.prior_image)
                cam.prior_image = None
            basename = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(detected_objects))
            cv2.imwrite(basename + '.jpg', image)
            with open(basename + '.txt', 'w') as file:
                file.write(json.dumps(predictions))
        for p in predictions:
            draw_bbox(im_pil, p, colors.get(p['tagName'], fallback='red'))
        if not cam.is_file:
            im_pil.save(basename + '-annotated.jpg')

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
        if cam.age > 2:
            notify(message, im_pil, predictions, config, st)
        else:
            print("Skipping notifications until after warm up")
        cam.objects = detected_objects

    return prediction_time
