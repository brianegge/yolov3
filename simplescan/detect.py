import os
import time
import sys
from io import BytesIO,StringIO
from object_detection import ObjectDetection
from utils import bb_intersection_over_union,draw_bbox,draw_road
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
        self.fails = 0
        self.skip = 0

    def capture(self):
        self.image = None
        self.resized = None
        self.resized2 = None
        if self.skip > 0:
            self.error = 'skip'
            self.skip -= 1
            return self
        if 'file' in self.config:
            self.is_file = True
            self.image = cv2.imread(self.config['file'])
        else:
            try:
                resp = self.session.get(self.config['uri'], timeout=15, stream=True).raw
                bytes = np.asarray(bytearray(resp.read()), dtype="uint8")
                if len(bytes) == 0:
                    self.error = 'empty'
                    return self
                self.image = cv2.imdecode(bytes, cv2.IMREAD_UNCHANGED)
                if self.vehicle_check:
                    self.resized2 = cv2.resize(self.image, (608, 608))
                hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                sum = np.sum(hsv[:,:,0])
                if sum == 0:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                self.resized = cv2.resize(self.image, (1344, 768))
                self.error = None
                self.fails = 0
            except:
                self.image = None
                self.resized = None
                self.skip = 2 ** self.fails
                self.fails += 1
                self.error = sys.exc_info()[0]
        return self
    
def add_centers(predictions):
    for p in predictions:
        bbox = p['boundingBox']
        p['center'] = {}
        center = p['center']
        center['x'] = bbox['left'] + bbox['width'] / 2.0
        center['y'] = bbox['top'] + bbox['height'] / 2.0

def detect(cam, color_model, grey_model, vehicle_model, config, st):
    threshold = config['detector'].getfloat('threshold')
    smartthings = config['smartthings']
    image = cam.image
    if image is None:
        return 0,"{}=[{}]".format(cam.name, cam.error)
    prediction_start = timer()
    try:
        if len(cam.resized.shape) == 3:
            predictions = color_model.predict_image(cam.resized)
        elif len(cam.resized.shape) == 2:
            predictions = grey_model.predict_image(cam.resized)
        else:
            return 0,"Unknown image shape {}".format(cam.resized.shape)
    except OSError as e:
        return 0,"{}=error:{}".format(cam.name, sys.exc_info()[0])
    cam.age = cam.age + 1
    vehicle_predictions = []
    if cam.vehicle_check and vehicle_model is not None:
        vehicle_predictions = vehicle_model.predict_image(cam.resized2)
        # include all vehicle predictions for now
        predictions += vehicle_predictions
    prediction_time = (timer() - prediction_start)
    # filter out lower predictions
    predictions = list(filter(lambda p: p['probability'] > config['thresholds'].getfloat(p['tagName'], threshold) or (p['tagName'] in cam.objects and p['probability'] > 0.4), predictions))
    for p in predictions:
        p['camName'] = cam.name
    add_centers(predictions)
    # remove road
    if cam.name in ['peach tree','driveway']:
        for p in predictions:
            x = p['center']['x']
            if x < 0.651:
                road_y = 0.31 + 0.038 * x
            else:
                road_y = 0.348 + 0.131 * (x - 0.651) / (1.0 - 0.651)
            p['road_y'] = road_y
            if p['center']['y'] < road_y and (p['tagName'] in ['vehicle','person']):
                p['ignore'] = 'road'
    elif cam.name == 'garage-l':
        for p in predictions:
            if p['boundingBox']['top'] + p['boundingBox']['height'] < .18 and (p['tagName'] in ['vehicle','person']):
                p['ignore'] = 'road'
    elif cam.name == 'garage-r':
        for p in predictions:
            if p['tagName'] in ['vehicle']:
                p['ignore'] = 'package cam'
    elif cam.name in ['front entry']:
        for p in filter(lambda p: p['tagName'] == 'package', predictions):
            if p['center']['x'] < 0.178125:
                p['ignore'] = 'in grass'
    for p in filter(lambda p: 'ignore' not in p, predictions):
        if "*" in cam.excludes:
            for e in cam.excludes["*"]:
                iou = bb_intersection_over_union(e,p['boundingBox'])
                if iou > 0.5:
                    p['ignore'] = e.get('comment', 'static')
                    break
        if p['tagName'] in cam.excludes:
            for i,e in enumerate(cam.excludes[p['tagName']]):
                iou = bb_intersection_over_union(e,p['boundingBox'])
                if iou > 0.5:
                    p['ignore'] = e.get('comment', 'static iou {}'.format(iou))
                    break
        if p['tagName'] == 'package' and cam.name == 'shed':
            p['ignore'] = 'class for cam'

    valid_predictions = list(filter(lambda p: not('ignore' in p), predictions))

    save_dir = os.path.join(config['detector']['save-path'],date.today().strftime("%Y%m%d"))
    os.makedirs(save_dir,exist_ok=True)
    if len(valid_predictions) == 0:
        if len(list(filter(lambda p: p != 'dog', cam.objects))):
            print(" %s departed" % (",".join(cam.objects)), end="")
            basename = os.path.join(save_dir, datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(cam.objects) + "-departed")
            if isinstance(image, Image.Image):
                image.save(basename + '.jpg')
            else:
                cv2.imwrite(basename + '.jpg', image)
            cam.objects.clear()
        cam.prior_image = image
        cam.prior_time = datetime.now()
    detected_objects = list(p['tagName'] for p in valid_predictions)
    # Always open garage door, we can call this many times
    if smartthings:
        if 'cat' in detected_objects and cam.name in ['shed','garage-l', 'garage-r'] and 'crack_garage' in smartthings:
            print("Cracking open garage door for cat")
            r = requests.get(smartthings['crack_garage'])
            if r.json()['result'] != "OK":
                print("Failed to crack open garage door for cat")
                print(r.text)

    colors = config['colors']
    uniq_predictions = []
    for p in valid_predictions:
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
                p['ignore'] = "prev-iou={:.2f}".format(iou)
                skip = True
                break
        if not skip:
            prev_class.append(p['boundingBox'])
            uniq_predictions.append(p)

    if len(valid_predictions) >= 0:
        if isinstance(image, Image.Image):
            im_pil = image.copy() # for drawing on
        else:
            im_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(im_pil)
        for p in predictions:
            if 'ignore' in p:
                width = 2
            else:
                width = 4
            color = colors.get(p['tagName'], fallback='red')
            draw_bbox(im_pil, p, color, width=width)
        if cam.name in ['peach tree','driveway']:
            draw_road(im_pil, [(0, 0.31), (0.651, 0.348), (1.0, 0.348 + 0.131)])
        elif cam.name in ['garage-l']:
            draw_road(im_pil, [(0, 0.18), (1.0, 0.18)])

    uniq_objects = list(p['tagName'] for p in uniq_predictions)

    # Only notify deer if not seen
    if 'deer' in uniq_objects:
        st.deer_alert()

    if len(uniq_objects):
        if cam.name == 'driveway':
            message = "%s in %s" % (",".join(detected_objects),cam.name)
        elif cam.name == 'shed':
            message = "%s in front of garage" % ",".join(detected_objects)
        elif cam.name == 'garage-r':
            message = "%s in front of left garage" % ",".join(detected_objects)
        elif cam.name == 'garage-l':
            message = "%s in front of right garage" % ",".join(detected_objects)
        else:
            message = "%s near %s" % (",".join(detected_objects),cam.name)
        if cam.age > 2:
            priority = notify(cam, message, im_pil, valid_predictions, config, st)
        else:
            print("Skipping notifications until after warm up")
            priority = 0
        cam.objects = detected_objects
    else:
        priority = -4

    if priority > -3 and not cam.is_file:
        # don't save file if we're reading from a file
        if cam.prior_image is not None:
            basename = os.path.join(save_dir,cam.prior_time.strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(detected_objects) + "-prior")
            if isinstance(cam.prior_image, Image.Image):
                cam.prior_image.save(basename + '.jpg')
            else:
                cv2.imwrite(basename + '.jpg', cam.prior_image)
            cam.prior_image = None
        basename = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(detected_objects))
        if isinstance(image, Image.Image):
            image.save(basename + '.jpg')
        else:
            cv2.imwrite(basename + '.jpg', image)
        with open(basename + '.txt', 'w') as file:
            file.write(json.dumps(predictions))
        im_pil.save(basename + '-annotated.jpg')

    def format_prediction(p):
        if 'ignore' in p:
            return "{}:{:.2f}:{}".format(p['tagName'], p['probability'], p['ignore'])
        else:
            return "{}:{:.2f}:p{}".format(p['tagName'], p['probability'], p.get('priority','?'))
    return prediction_time, '{}=['.format(cam.name) + ",".join( format_prediction(p) for p in predictions ) + ']'
