# The steps implemented in the object detection sample code:
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
# scale is a multiplier of the default model size
import os
import subprocess
import sys
import configparser
from timeit import default_timer as timer
from detect import detect, Camera
from onnx_object_detection import ONNXRuntimeObjectDetection
from object_detection_rt import ONNXTensorRTObjectDetection
from object_detection_rtv4 import ONNXTensorRTv4ObjectDetection
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
import requests
import faulthandler, signal
import argparse
import json
import time
import sdnotify
from smartthings import SmartThings
from homeassistant import HomeAssistant
from utils import cleanup


async def main(options):
    config = configparser.ConfigParser()
    config.read(options.config_file)
    st = SmartThings(config)
    ha = HomeAssistant(config["homeassistant"])
    detector_config = config["detector"]
    color_model_config = config["color-model"]
    grey_model_config = config["grey-model"]

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
    print("Configured %i cams" % i)
    async_cameras = len(list(filter(lambda cam: cam.capture_async, cams)))
    # async_cameras = 4
    print("%i async workers" % async_cameras)
    async_pool = concurrent.futures.ThreadPoolExecutor(max_workers=async_cameras)
    sync_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    sd.notify("READY=1")
    sd.notify("STATUS=Running")
    cleanup_time = datetime(1970, 1, 1, 0, 0, 0)
    while True:
        sd.notify("WATCHDOG=1")
        start_time = timer()
        prediction_time = 0.0
        capture_futures = []
        messages = []
        if options.sync:
            for cam in cams:
                if cam.poll() is None:
                    cam.capture()
                p, m = detect(
                    cam, color_model, grey_model, vehicle_model, config, st, ha
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
                    print("cam:%s poll:" % cam.name, sys.exc_info()[0])

            count = 0
            for f in concurrent.futures.as_completed(capture_futures, timeout=180):
                try:
                    cam = f.result()
                    if cam:
                        p, m = detect(
                            cam, color_model, grey_model, vehicle_model, config, st, ha
                        )
                        prediction_time += p
                        messages.append(m)
                        count += 1
                except KeyboardInterrupt:
                    return

            if count == 0:
                print("Scanning ", end="")
                # scan each camera
                for cam in filter(
                    lambda cam: (datetime.now() - cam.prior_time).total_seconds() > 10,
                    cams,
                ):
                    try:
                        if cam.capture_async and not options.sync:
                            capture_futures.append(async_pool.submit(cam.capture))
                        else:
                            capture_futures.append(sync_pool.submit(cam.capture))
                    except KeyboardInterrupt:
                        return
                    except requests.exceptions.ConnectionError:
                        print(
                            "cam:%s requests.exceptions.ConnectionError:" % cam.name,
                            sys.exc_info()[0],
                        )

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
                                st,
                                ha,
                            )
                            prediction_time += p
                            messages.append(m)
                    except KeyboardInterrupt:
                        return
            else:
                print("Reading ", end="")

        end_time = timer()
        print(",".join(sorted(messages)), end="")
        print(
            ".. completed in %.2fs, spent %.2fs predicting"
            % ((end_time - start_time), prediction_time),
            flush=True,
        )
        # if prediction_time < 0.01:
        #    print("Cameras appear down, waiting 30 seconds")
        #    time.sleep(30)
        if datetime.now() - cleanup_time > timedelta(minutes=60):
            # script = os.path.join(os.path.dirname(__file__), 'cleanup.sh')
            print("Cleaning up")
            # subprocess.run(script, check=True)
            for cam in filter(lambda cam: cam.ftp_path, cams):
                cleanup(cam.ftp_path)
                cam.globber = None
            cleanup_time = datetime.now()
        if "once" in detector_config:
            break


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)
    # python 3.7 is asyncio.run()
    parser = argparse.ArgumentParser(description="Process cameras")
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("config_file", nargs="?", default="config.txt")
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(main(args))
