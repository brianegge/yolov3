import hashlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from pprint import pformat
from timeit import default_timer as timer
from urllib.parse import urlparse

import cv2
import humanize
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from requests.auth import HTTPDigestAuth

from notify import notify
from object_detection import ObjectDetection
from utils import bb_intersection_over_union, cleanup, draw_bbox, draw_road

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, config, excludes):
        self.name = config["name"]
        self.ha_name = self.name.replace(" ", "_")
        self.config = config
        self.objects = set()
        self.prev_predictions = {}
        self.is_file = False
        self.counts = {}
        self.vehicle_check = config.getboolean("vehicle_check", False)
        self.excludes = excludes
        self.capture_async = config.getboolean("async", False)
        self.error = None
        self.image = None
        self.image_hash = 0
        self.source = None
        self.prior_image = None
        self.prior_time = datetime.fromtimestamp(0)
        self.prior_priority = -4
        self.age = 0
        self.fails = 0
        self.skip = 0
        self.ftp_path = config.get("ftp-path", None)
        self.interval = config.getint("interval", 60)
        self.globber = None
        self.session = None

    def poll(self):
        # logger.debug('polling {}'.format(self.name))
        if self.ftp_path:
            img = None
            while img == None:
                if self.globber is None:
                    globber = Path(self.ftp_path).glob("**/*.jpg")
                try:
                    f = next(globber)
                except OSError as e:
                    logger.error(f"Error scanning {self.ftp_path}: {e}\n{e.args}")
                    self.globber = None
                    return None
                except StopIteration:
                    self.globber = None
                    cleanup(self.ftp_path)
                    return None
                if datetime.fromtimestamp(
                    os.path.getmtime(f)
                ) < datetime.now() - timedelta(minutes=10):
                    logger.warning(f"Skipping old file {f}")
                    os.remove(f)
                    continue
                img = cv2.imread(str(f))
                os.remove(f)
                if img is not None and len(img) > 0:
                    h = hashlib.md5(img.view(np.uint8)).hexdigest()
                    if self.image_hash == h:
                        self.error = "dup"
                        return None
                    self.image = img
                    self.image_hash = h
                    self.source = f
                    self.resize()
                    return self
                else:
                    self.error = "bad file"
        return None

    def capture(self):
        self.image = None
        self.resized = None
        self.resized2 = None
        if self.skip > 0:
            self.error = "skip={}".format(self.skip)
            self.skip -= 1
            return self
        if "file" in self.config:
            self.is_file = True
            self.image = cv2.imread(self.config["file"])
            self.source = self.config["file"]
            self.resize()
        else:
            if self.session == None:
                self.session = requests.Session()
                if "user" in self.config:
                    self.session.auth = HTTPDigestAuth(
                        self.config["user"], self.config["password"]
                    )
            try:
                with self.session.get(
                    self.config["uri"], timeout=20, stream=True
                ) as resp:
                    resp.raise_for_status()
                    bytes = np.asarray(bytearray(resp.raw.read()), dtype="uint8")
                    if len(bytes) == 0:
                        self.error = "empty"
                        return self
                    self.image = cv2.imdecode(bytes, cv2.IMREAD_UNCHANGED)
                    self.image_hash = hashlib.md5(self.image.view(np.uint8)).hexdigest()
                    self.source = self.config["uri"]
                    self.resize()
                    self.error = None
                    self.fails = 0
            except Exception as e:
                self.image = None
                self.image_hash = 0
                self.source = None
                self.resized = None
                self.skip = 2 ** self.fails
                self.fails += 1
                self.session = None
                self.error = sys.exc_info()[0]
                logger.exception(f"Error with {self.name}:{self.error}")
                if self.skip > 12:
                    self.reboot()
        return self

    def reboot(self):
        # "http://treeline-cam.home/cgi-bin/magicBox.cgi?action=reboot"
        url = (
            urlparse(self.config["uri"])
            ._replace(path="/cgi-bin/magicBox.cgi", query="action=reboot")
            .geturl()
        )
        logger.info(f"Rebooting {self.name}: {url}")
        try:
            r = requests.get(
                url, auth=HTTPDigestAuth(self.config["user"], self.config["password"])
            )
            r.raise_for_status()
        except:
            logger.exception("Failed to reboot %s", self.name)

    def resize(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        sum = np.sum(hsv[:, :, 0])
        if sum == 0:
            self.resized2 = cv2.resize(
                cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), (608, 608)
            )
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.resized = cv2.resize(self.image, (608, 608))
        else:
            resized = cv2.resize(self.image, (608, 608))
            self.resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.resized2 = self.resized


def add_centers(predictions):
    for p in predictions:
        bbox = p["boundingBox"]
        p["center"] = {}
        center = p["center"]
        center["x"] = bbox["left"] + bbox["width"] / 2.0
        center["y"] = bbox["top"] + bbox["height"] / 2.0


def detect(cam, color_model, grey_model, vehicle_model, config, ha, mqtt_client):
    threshold = config["detector"].getfloat("threshold")
    image = cam.image
    if image is None:
        return 0, "{}=[{}]".format(cam.name, cam.error)
    prediction_start = timer()
    try:
        if len(cam.resized.shape) == 3:
            predictions = color_model.predict_image(cam.resized)
        elif len(cam.resized.shape) == 2:
            predictions = grey_model.predict_image(cam.resized)
        else:
            return 0, "Unknown image shape {}".format(cam.resized.shape)
    except OSError as e:
        return 0, "{}=error:{}".format(cam.name, sys.exc_info()[0])
    cam.age = cam.age + 1
    vehicle_predictions = []
    if cam.vehicle_check and vehicle_model is not None:
        vehicle_predictions = vehicle_model.predict_image(cam.resized2)
        # include all vehicle predictions for now
        predictions += vehicle_predictions
    prediction_time = timer() - prediction_start
    # filter out lower predictions
    predictions = list(
        filter(
            lambda p: p["probability"]
            > config["thresholds"].getfloat(p["tagName"], threshold)
            or (p["tagName"] in cam.objects and p["probability"] > 0.4),
            predictions,
        )
    )
    for p in predictions:
        p["camName"] = cam.name
    add_centers(predictions)
    # remove road
    if cam.name in ["driveway", "peach tree"]:
        for p in predictions:
            x = p["center"]["x"]
            if x < 0.651:
                road_y = 0.31 + 0.038 * x
            else:
                road_y = 0.348 + 0.131 * (x - 0.651) / (1.0 - 0.651)
            p["road_y"] = road_y
            if p["center"]["y"] < road_y and (
                p["tagName"] in ["vehicle", "person", "package", "dog"]
            ):
                p["ignore"] = "road"
    elif cam.name == "garage-l":
        for p in predictions:
            if p["boundingBox"]["top"] + p["boundingBox"]["height"] < 0.22 and (
                p["tagName"] in ["vehicle", "person"]
            ):
                p["ignore"] = "neighbor"
    elif cam.name in ["front entry"]:
        for p in filter(lambda p: p["tagName"] == "package", predictions):
            if p["center"]["x"] < 0.178125:
                p["ignore"] = "in grass"
    for p in filter(lambda p: "ignore" not in p, predictions):
        if "*" in cam.excludes:
            for e in cam.excludes["*"]:
                iou = bb_intersection_over_union(e, p["boundingBox"])
                if iou > 0.5:
                    p["ignore"] = e.get("comment", "static")
                    break
        if p["tagName"] in cam.excludes:
            for i, e in enumerate(cam.excludes[p["tagName"]]):
                iou = bb_intersection_over_union(e, p["boundingBox"])
                if iou > 0.5:
                    p["ignore"] = e.get("comment", "static iou {}".format(iou))
                    break

    valid_predictions = list(filter(lambda p: not ("ignore" in p), predictions))
    valid_objects = set(p["tagName"] for p in valid_predictions)
    departed_objects = cam.objects - valid_objects

    yyyymmdd = date.today().strftime("%Y%m%d")
    save_dir = os.path.join(config["detector"]["save-path"], yyyymmdd)
    today_dir = os.path.join(config["detector"]["save-path"], "today")
    if not os.path.exists(today_dir):
        os.symlink(yyyymmdd, today_dir)
    if os.readlink(today_dir) != yyyymmdd:
        os.unlink(today_dir)
        os.symlink(yyyymmdd, today_dir)

    os.makedirs(save_dir, exist_ok=True)
    if len(departed_objects) > 0 and cam.prior_priority > -3:
        logger.info(
            "{} current={}, prior={}, departed={}".format(
                cam.name,
                ",".join(valid_objects),
                ",".join(cam.objects),
                ",".join(departed_objects),
            )
        )
        basename = os.path.join(
            save_dir,
            datetime.now().strftime("%H%M%S")
            + "-"
            + cam.name.replace(" ", "_")
            + "-"
            + "_".join(departed_objects)
            + "-departed",
        )
        if isinstance(image, Image.Image):
            image.save(basename + ".jpg")
        else:
            cv2.imwrite(basename + ".jpg", image)
        cam.objects = valid_objects
    # Always open garage door, we can call this many times
    if "cat" in valid_objects and cam.name in ["shed", "garage-l", "garage-r"]:
        logger.info("Letting cat in")
        ha.crack_garage_door()
    elif "cat" in valid_objects and cam.name in ["garage"]:
        logger.info("Letting cat out")
        ha.let_cat_out()

    colors = config["colors"]
    new_predictions = []
    for p in valid_predictions:
        this_box = p["boundingBox"]
        this_name = p["tagName"]
        prev_class = cam.prev_predictions.setdefault(p["tagName"], [])
        for prev in prev_class:
            prev_box = prev["boundingBox"]
            iou = bb_intersection_over_union(prev_box, this_box)
            logger.debug(f"iou {cam.name}:{this_name} = prev_box & this_box = {iou}")
            if iou > 0.5:
                p["iou"] = iou
                prev["boundingBox"] = this_box  # move the box to current
                prev["last_time"] = datetime.now()
                prev["age"] = prev["age"] + 1
                for t in ["age", "ignore", "priority", "priority_type"]:
                    if t in prev:
                        p[t] = prev[t]
        if not "iou" in p:
            p["start_time"] = datetime.now()
            p["last_time"] = datetime.now()
            p["age"] = 0
            prev_class.append(p)
            new_predictions.append(p)
    expired = []
    for prev_tag, prev_class in cam.prev_predictions.items():
        expired += [
            x
            for x in prev_class
            if x["last_time"] < datetime.now() - timedelta(minutes=1)
        ]
        prev_class[:] = [x for x in prev_class if x not in expired]

    if len(valid_predictions) >= 0:
        if isinstance(image, Image.Image):
            im_pil = image.copy()  # for drawing on
        else:
            im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for p in predictions:
            if "ignore" in p:
                width = 2
            else:
                width = 4
            color = colors.get(p["tagName"], fallback="red")
            draw_bbox(im_pil, p, color, width=width)
        if cam.name in ["peach tree", "driveway"]:
            draw_road(im_pil, [(0, 0.31), (0.651, 0.348), (1.0, 0.348 + 0.131)])
        elif cam.name in ["garage-l"]:
            draw_road(im_pil, [(0, 0.22), (1.0, 0.22)])
    notify_expired = []
    for e in expired:
        draw_bbox(im_pil, e, "grey", width=4)
        t = (
            humanize.naturaltime(datetime.now() - e["start_time"])
            .replace(" ago", "")
            .replace("a minute", "minute")
        )
        e[
            "msg"
        ] = f"{e['tagName']} departed from {cam.name} after being seen {e['age']} times over the past {t}"
        e["departed"] = True
        logger.info(e["msg"])
        if datetime.now() - e["start_time"] > timedelta(minutes=2) and e["age"] > 4:
            notify_expired.append(e)
    if len(notify_expired):
        logger.debug(pformat(notify_expired))
        msg = ", ".join([x["msg"] for x in notify_expired])
        notify(
            cam,
            msg,
            im_pil,
            notify_expired,
            config,
            ha,
        )

    new_objects = set(p["tagName"] for p in new_predictions)

    # Only notify deer if not seen
    if "deer" in new_objects:
        ha.deer_alert(cam.name)
    # mosquitto_pub -h mqtt.home -t "homeassistant/sensor/deck-dog/config" -r -m '{"name": "deck dog count", "state_topic": "deck/dog/count", "state_class": "measurement", "uniq_id": "deck-dog", "availability_topic": "aicam/status"}'
    for o in ["vehicle", "dog", "person", "package"]:
        count = len(
            list(
                filter(
                    lambda p: p["tagName"] == o,
                    valid_predictions,
                )
            )
        )
        if cam.counts.get(o, -1) != count:
            logger.info(f"Publishing count {cam.name}/{o}/count={count}")
            mqtt_client.publish(f"{cam.name}/{o}/count", count, retain=True)
            cam.counts[o] = count

    if len(new_objects):
        if cam.name in ["driveway", "garage"]:
            message = "%s in %s" % (",".join(valid_objects), cam.name)
        elif cam.name == "shed":
            message = "%s in front of garage" % ",".join(valid_objects)
        elif cam.name == "garage-r":
            message = "%s in front of left garage" % ",".join(valid_objects)
        elif cam.name == "garage-l":
            message = "%s in front of right garage" % ",".join(valid_objects)
        else:
            message = "%s near %s" % (",".join(valid_objects), cam.name)
        if cam.age > 2:
            priority = notify(cam, message, im_pil, valid_predictions, config, ha)
        else:
            logger.info("Skipping notifications until after warm up")
            priority = -4
    elif len(valid_predictions) > 0:
        priority = cam.prior_priority
    else:
        priority = -4

    # Notify may also mark objects as ignore
    valid_predictions = list(filter(lambda p: not ("ignore" in p), predictions))
    cam.objects = set(p["tagName"] for p in valid_predictions)

    if priority > -3 and not cam.is_file:
        # don't save file if we're reading from a file
        if cam.prior_image is not None:
            priorname = (
                os.path.join(
                    save_dir,
                    cam.prior_time.strftime("%H%M%S")
                    + "-"
                    + cam.name.replace(" ", "_")
                    + "-"
                    + "_".join(valid_objects)
                    + "-prior",
                )
                + ".jpg"
            )
            if isinstance(cam.prior_image, Image.Image):
                cam.prior_image.save(priorname)
            else:
                cv2.imwrite(priorname, cam.prior_image)
            cam.prior_image = None
            utime = time.mktime(cam.prior_time.timetuple())
            os.utime(priorname, (utime, utime))
        basename = os.path.join(
            save_dir,
            datetime.now().strftime("%H%M%S")
            + "-"
            + cam.name.replace(" ", "_")
            + "-"
            + "_".join(valid_objects),
        )
        if isinstance(image, Image.Image):
            image.save(basename + ".jpg")
        else:
            cv2.imwrite(basename + ".jpg", image)
        with open(basename + ".txt", "w") as file:
            j = {
                "source": str(cam.source),
                "time": str(datetime.now()),
                "predictions": predictions,
            }
            file.write(json.dumps(j, indent=4, default=str))
        im_pil.save(basename + "-annotated.jpg")
    else:
        cam.prior_image = image
    cam.prior_time = datetime.now()
    cam.prior_priority = priority

    def format_prediction(p):
        o = "{}:{:.2f}".format(p["tagName"], p["probability"])
        if "iou" in p:
            o += ":iou={:.2f}".format(p["iou"])
        if "ignore" in p:
            o += ":ignore={}".format(p["ignore"])
        if "priority" in p:
            o += ":p={}".format(p["priority"])
        if "priority_type" in p:
            o += ":pt={}".format(p["priority_type"])
        if "age" in p:
            o += ":age={}".format(p["age"])
        return o

    return (
        prediction_time,
        "{}=[".format(cam.name)
        + ",".join(format_prediction(p) for p in predictions)
        + "]",
    )
