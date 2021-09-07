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
from pathlib import Path

class Camera():
    def __init__(self, config, excludes):
        self.name = config['name']
        self.config = config
        self.objects = set()
        self.prev_predictions = {}
        self.is_file = False
        self.vehicle_check = config.getboolean('vehicle_check', False)
        self.excludes = excludes
        self.capture_async = self.config.getboolean('async', False)
        self.error = None
        self.image = None
        self.prior_image = None
        self.prior_time = datetime.fromtimestamp(0)
        self.prior_priority = -4
        self.age = 0
        self.fails = 0
        self.skip = 0
        self.ftp_path = config.get('ftp-path', None)
        self.globber = None
        self.session = None

    def poll(self):
        # print('polling {}'.format(self.name))
        if self.ftp_path:
            if self.globber is None:
                globber = Path(self.ftp_path).glob('**/*.jpg')
            try:
                f = next(globber)
            except StopIteration:
                self.globber = None
                return None
            print('found {}'.format(f))
            img = cv2.imread(str(f))
            os.remove(f)
            if img is not None and len(img) > 0:
                self.image = img
                self.resize()
                return self
            else:
                self.error='bad file'
        return None

    def capture(self):
        self.image = None
        self.resized = None
        self.resized2 = None
        if self.skip > 0:
            self.error = 'skip={}'.format(self.skip)
            self.skip -= 1
            return self
        if 'file' in self.config:
            self.is_file = True
            self.image = cv2.imread(self.config['file'])
            self.resize()
        else:
            if self.session == None:
                self.session = requests.Session()
                if 'user' in self.config:
                    self.session.auth = HTTPDigestAuth(self.config['user'], self.config['password'])
            try:
                with self.session.get(self.config['uri'], timeout=20, stream=False) as resp:
                    resp.raise_for_status()
                    bytes = np.asarray(bytearray(resp.raw.read()), dtype="uint8")
                    if len(bytes) == 0:
                        self.error = 'empty'
                        return self
                    self.image = cv2.imdecode(bytes, cv2.IMREAD_UNCHANGED)
                    self.resize()
                    self.error = None
                    self.fails = 0
            except:
                self.image = None
                self.resized = None
                self.skip = 2 ** self.fails
                self.fails += 1
                self.session = None
                self.error = sys.exc_info()[0]
        return self

    def resize(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        sum = np.sum(hsv[:,:,0])
        if sum == 0:
            self.resized2 = cv2.resize(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), (608, 608))
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.resized = cv2.resize(self.image, (608, 608))
        else:
            resized = cv2.resize(self.image, (608, 608))
            self.resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.resized2 = self.resized

def add_centers(predictions):
    for p in predictions:
        bbox = p['boundingBox']
        p['center'] = {}
        center = p['center']
        center['x'] = bbox['left'] + bbox['width'] / 2.0
        center['y'] = bbox['top'] + bbox['height'] / 2.0

def detect(cam, color_model, grey_model, vehicle_model, config, st, ha):
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
    if cam.name in ['driveway']: #'peach tree'
        for p in predictions:
            x = p['center']['x']
            if x < 0.651:
                road_y = 0.31 + 0.038 * x
            else:
                road_y = 0.348 + 0.131 * (x - 0.651) / (1.0 - 0.651)
            p['road_y'] = road_y
            if p['center']['y'] < road_y and (p['tagName'] in ['vehicle','person','package']):
                p['ignore'] = 'road'
    elif cam.name == 'garage-l':
        for p in predictions:
            if p['boundingBox']['top'] + p['boundingBox']['height'] < .21 and (p['tagName'] in ['vehicle','person']):
                p['ignore'] = 'neighbor'
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

    valid_predictions = list(filter(lambda p: not('ignore' in p), predictions))
    valid_objects = set(p['tagName'] for p in valid_predictions)
    departed_objects = cam.objects - valid_objects

    yyyymmdd = date.today().strftime("%Y%m%d")
    save_dir = os.path.join(config['detector']['save-path'],yyyymmdd)
    today_dir = os.path.join(config['detector']['save-path'],'today')
    if not os.path.exists(today_dir):
        os.symlink(yyyymmdd, today_dir)
    if os.readlink(today_dir) != yyyymmdd:
        os.unlink(today_dir)
        os.symlink(yyyymmdd, today_dir)

    os.makedirs(save_dir,exist_ok=True)
    if len(departed_objects) > 0 and cam.prior_priority > -3:
        print("\n{} current={}, prior={}, departed={}".format( cam.name, ",".join(valid_objects), ",".join(cam.objects), ",".join(departed_objects)) )
        basename = os.path.join(save_dir, datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(departed_objects) + "-departed")
        if isinstance(image, Image.Image):
            image.save(basename + '.jpg')
        else:
            cv2.imwrite(basename + '.jpg', image)
    # Always open garage door, we can call this many times
    if smartthings:
        if 'cat' in valid_objects and cam.name in ['shed','garage-l', 'garage-r'] and 'crack_garage' in smartthings:
            print("Letting cat in")
            r = requests.get(smartthings['crack_garage'])
            if r.json()['result'] != "OK":
                print("Failed to crack open garage door for cat")
                print(r.text)
        elif 'cat' in valid_objects and cam.name in ['garage']:
            print("Letting cat out")
            st.crack_garage_door()

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
                priority = cam.prior_priority
                break
        if not skip:
            prev_class.append(p['boundingBox'])
            uniq_predictions.append(p)

    if len(valid_predictions) >= 0:
        if isinstance(image, Image.Image):
            im_pil = image.copy() # for drawing on
        else:
            im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
            draw_road(im_pil, [(0, 0.21), (1.0, 0.21)])

    uniq_objects = set(p['tagName'] for p in uniq_predictions)

    # Only notify deer if not seen
    if 'deer' in uniq_objects:
        st.deer_alert()

    if len(uniq_objects):
        if cam.name in ['driveway','garage']:
            message = "%s in %s" % (",".join(valid_objects),cam.name)
        elif cam.name == 'shed':
            message = "%s in front of garage" % ",".join(valid_objects)
        elif cam.name == 'garage-r':
            message = "%s in front of left garage" % ",".join(valid_objects)
        elif cam.name == 'garage-l':
            message = "%s in front of right garage" % ",".join(valid_objects)
        else:
            message = "%s near %s" % (",".join(valid_objects),cam.name)
        if cam.age > 2:
            priority = notify(cam, message, im_pil, valid_predictions, config, st, ha)
        else:
            print("Skipping notifications until after warm up")
            priority = 0
    else:
        priority = -4
    # Notify may also mark objects as ignore
    valid_predictions = list(filter(lambda p: not('ignore' in p), predictions))
    cam.objects = set(p['tagName'] for p in valid_predictions)

    if priority > -3 and not cam.is_file:
        # don't save file if we're reading from a file
        if cam.prior_image is not None:
            basename = os.path.join(save_dir,cam.prior_time.strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(valid_objects) + "-prior")
            if isinstance(cam.prior_image, Image.Image):
                cam.prior_image.save(basename + '.jpg')
            else:
                cv2.imwrite(basename + '.jpg', cam.prior_image)
            cam.prior_image = None
        basename = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + cam.name + "-" + "_".join(valid_objects))
        if isinstance(image, Image.Image):
            image.save(basename + '.jpg')
        else:
            cv2.imwrite(basename + '.jpg', image)
        with open(basename + '.txt', 'w') as file:
            file.write(json.dumps(predictions))
        im_pil.save(basename + '-annotated.jpg')
    else:
        cam.prior_image = image
        cam.prior_time = datetime.now()
    cam.prior_priority = priority

    def format_prediction(p):
        if 'ignore' in p:
            return "{}:{:.2f}:{}".format(p['tagName'], p['probability'], p['ignore'])
        else:
            return "{}:{:.2f}:p{}".format(p['tagName'], p['probability'], p.get('priority','?'))
    return prediction_time, '{}=['.format(cam.name) + ",".join( format_prediction(p) for p in predictions ) + ']'
