import hashlib
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
import paho.mqtt.client as paho
import requests
from PIL import Image, UnidentifiedImageError
from requests.auth import HTTPDigestAuth

from utils import cleanup

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
        self.last_show_count = -1
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
        self.interval = config.getint("interval", 30)
        self.globber = None
        self.session = None
        self.mqtt = set(config.get("mqtt", "").split(","))
        self.mqtt_client = paho.Client(f"aicam-{self.ha_name}")
        self.mqtt_client.connect("mqtt.home", 1883)
        self.mqtt_client.loop_start()

    def __del__(self):
        self.mqtt_client.disconnect()  # disconnect gracefully
        self.mqtt_client.loop_stop()  # disconnect gracefully

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
