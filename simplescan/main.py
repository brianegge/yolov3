# The steps implemented in the object detection sample code:
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
# scale is a multiplier of the default model size
import argparse
import asyncio
import concurrent.futures
import configparser
import faulthandler
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from timeit import default_timer as timer

import paho.mqtt.client as paho
import requests
import sdnotify

from detect import Camera, detect
from homeassistant import HomeAssistant
from object_detection_rt import ONNXTensorRTObjectDetection
from object_detection_rtv4 import ONNXTensorRTv4ObjectDetection
from onnx_object_detection import ONNXRuntimeObjectDetection
from utils import cleanup

log = logging.getLogger("aicam")
mlog = logging.getLogger("mqtt")
kill_now = False


def on_publish(client, userdata, mid):
    mlog.debug("on_publish({},{})".format(userdata, mid))


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    mlog.info("mqtt connected")
    client.publish("aicam/status", "online", retain=True)


def on_disconnect(client, userdata, rc):
    mlog.info("mqtt disconnected reason  " + str(rc))
    global kill_now
    kill_now = True


def on_message(self, mqtt_client, obj, msg):
    mlog.info("on_message()")


class GracefulKiller:
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        global kill_now
        kill_now = True


async def main(options):
    config = configparser.ConfigParser()
    config.read(options.config_file)
    ha = HomeAssistant(config["homeassistant"])
    detector_config = config["detector"]
    color_model_config = config["color-model"]
    grey_model_config = config["grey-model"]
    lwt = "aicam/status"
    mqtt_client = paho.Client("aicam")
    mqtt_client.enable_logger(logger=mlog)
    mqtt_client.on_publish = on_publish
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message = on_message
    mqtt_client.will_set(lwt, payload="offline", qos=0, retain=True)
    mqtt_client.connect("mqtt.home", 1883)
    mqtt_client.subscribe("test")  # get on connect messages
    mqtt_client.loop_start()

    # Load labels
    with open(detector_config["labelfile-path"], "r") as f:
        labels = [l.strip() for l in f.readlines()]
    if "vehicle-labelfile-path" in detector_config:
        with open(detector_config["vehicle-labelfile-path"], "r") as f:
            vehicle_labels = [l.strip() for l in f.readlines()]
    # open static exclusion
    excludes = {}
    if "excludes-file" in detector_config:
        with open(detector_config["excludes-file"]) as f:
            excludes = json.load(f)

    sd = sdnotify.SystemdNotifier()
    sd.notify("STATUS=Loading color model")
    color_model = ONNXTensorRTv4ObjectDetection(color_model_config, labels)
    sd.notify("STATUS=Loading grey model")
    grey_model = ONNXTensorRTv4ObjectDetection(grey_model_config, labels)
    sd.notify("STATUS=Loading vehicle/packages model")
    vehicle_model = ONNXTensorRTv4ObjectDetection(
        config["vehicle-model"], vehicle_labels
    )
    sd.notify("STATUS=Loaded models")

    cams = []
    i = 0
    while "cam%d" % i in config.sections():
        cams.append(
            Camera(config["cam%d" % i], excludes.get(config["cam%d" % i]["name"], {}))
        )
        i += 1
    log.info("Configured %i cams" % i)
    async_cameras = len(list(filter(lambda cam: cam.capture_async, cams)))
    # async_cameras = 4
    log.info("%i async workers" % async_cameras)
    async_pool = concurrent.futures.ThreadPoolExecutor(max_workers=async_cameras)
    sync_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    for cam in filter(lambda c: c.vehicle_check, cams):
        j = {
            "name": f"{cam.name} Vehicle Count",
            "state_topic": "{}/vehicle/count".format(cam.ha_name),
            "state_class": "measurement",
            "uniq_id": "{}-vehicle".format(cam.ha_name),
            "availability_topic": lwt,
        }
        mqtt_client.publish(
            "homeassistant/sensor/{}-vehicle/config".format(cam.ha_name),
            json.dumps(j),
            retain=True,
        )

    sd.notify("READY=1")
    sd.notify("STATUS=Running")
    cleanup_time = datetime(1970, 1, 1, 0, 0, 0)
    killer = GracefulKiller()
    global kill_now
    while not kill_now:
        sd.notify("WATCHDOG=1")
        start_time = timer()
        prediction_time = 0.0
        capture_futures = []
        messages = []
        log_line = ""
        if options.sync:
            for cam in cams:
                if cam.poll() is None:
                    cam.capture()
                p, m = detect(
                    cam,
                    color_model,
                    grey_model,
                    vehicle_model,
                    config,
                    st,
                    ha,
                    mqtt_client,
                )
                prediction_time += p
                messages.append(m)
        else:
            for cam in filter(lambda cam: cam.ftp_path, cams):
                try:
                    capture_futures.append(async_pool.submit(cam.poll))
                except KeyboardInterrupt:
                    return
                except requests.exceptions.ConnectionError:
                    log.warning("cam:%s poll:" % cam.name, sys.exc_info()[0])

            count = 0
            for f in concurrent.futures.as_completed(capture_futures, timeout=180):
                try:
                    cam = f.result()
                    if cam:
                        p, m = detect(
                            cam,
                            color_model,
                            grey_model,
                            vehicle_model,
                            config,
                            ha,
                            mqtt_client,
                        )
                        prediction_time += p
                        messages.append(m)
                        count += 1
                except KeyboardInterrupt:
                    return

            if count == 0:
                # scan each camera
                for cam in filter(
                    lambda cam: (datetime.now() - cam.prior_time).total_seconds()
                    > cam.interval,
                    cams,
                ):
                    try:
                        if cam.capture_async and not options.sync:
                            capture_futures.append(async_pool.submit(cam.capture))
                        else:
                            capture_futures.append(sync_pool.submit(cam.capture))
                        count += 1
                    except KeyboardInterrupt:
                        return
                    except requests.exceptions.ConnectionError:
                        log.warning(
                            "cam:%s requests.exceptions.ConnectionError:" % cam.name,
                            sys.exc_info()[0],
                        )
                if count > 0:
                    log_line = "Snapshoting "

                for f in concurrent.futures.as_completed(capture_futures, timeout=180):
                    try:
                        cam = f.result()
                        if cam:
                            p, m = detect(
                                cam,
                                color_model,
                                grey_model,
                                vehicle_model,
                                config,
                                ha,
                                mqtt_client,
                            )
                            prediction_time += p
                            messages.append(m)
                    except KeyboardInterrupt:
                        return
            else:
                log_line = "Reading "

        end_time = timer()
        if count > 0:
            log_line += ",".join(sorted(messages))
            log_line += ".. completed in %.2fs, spent %.2fs predicting" % (
                (end_time - start_time),
                prediction_time,
            )
        if len(log_line) > 0:
            log.debug(log_line)
        if prediction_time < 0.1:
            if datetime.now() - cleanup_time > timedelta(minutes=15):
                log.debug("Cleaning up")
                for cam in filter(lambda cam: cam.ftp_path, cams):
                    cleanup(cam.ftp_path)
                    cam.globber = None
                cleanup_time = datetime.now()
            else:
                time.sleep(1.0)
        if "once" in detector_config:
            break
    # graceful shutdown
    log.info("Graceful shutdown initiated")
    mqtt_client.disconnect()  # disconnect gracefully
    mqtt_client.loop_stop()  # stops network loop
    del color_model
    del grey_model
    del vehicle_model


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)
    # python 3.7 is asyncio.run()
    parser = argparse.ArgumentParser(description="Process cameras")
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("config_file", nargs="?", default="config.txt")

    handlers = [
        logging.StreamHandler(),
        # logging.FileHandler("/var/log/aicam.log"),
    ]
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)-5s %(levelname)-4s %(message)s",
        datefmt="%b-%d %H:%M",
        handlers=handlers,
    )
    logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)
    logging.getLogger("detect").setLevel(logging.INFO)
    mlog.setLevel(logging.INFO)
    log.info("Starting")
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(main(args))
    log.info("Graceful shutdown complete")
