# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
# scale is a multiplier of the default model size
import os
import sys
import configparser
from timeit import default_timer as timer
from detect import detect,Camera
from onnx_object_detection import ONNXRuntimeObjectDetection
from object_detection_rt import ONNXTensorRTObjectDetection
from object_detection_rtv4 import ONNXTensorRTv4ObjectDetection
import asyncio
import concurrent.futures
from datetime import datetime
import requests
import faulthandler, signal
import argparse
import json
import time
import sdnotify
from smartthings import SmartThings

async def main(options):
    config = configparser.ConfigParser()
    config.read(options.config_file)
    st = SmartThings(config)
    detector_config = config['detector']

    # Load labels
    with open(detector_config['labelfile-path'], 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    if 'vehicle-labelfile-path' in detector_config:
        with open(detector_config['vehicle-labelfile-path'], 'r') as f:
            vehicle_labels = [l.strip() for l in f.readlines()]
    # open static exclusion
    excludes={}
    if 'excludes-file' in detector_config:
        with open(detector_config['excludes-file']) as f:
            excludes = json.load(f)

    sd = sdnotify.SystemdNotifier()
    sd.notify("STATUS=Loading main model")
    vehicle_model = None
    if 'yolov4' in detector_config['color-model']:
        color_model = ONNXTensorRTv4ObjectDetection(detector_config['color-model'], labels, detector_config.getfloat('prob_threshold', 0.10))
        grey_model = ONNXTensorRTv4ObjectDetection(detector_config['grey-model'], labels, detector_config.getfloat('prob_threshold', 0.10), channels=1)
    elif options.trt:
        color_model = ONNXTensorRTObjectDetection(detector_config['color-model'], labels, detector_config.getfloat('prob_threshold', 0.10), scale=4)
    else:
        color_model = ONNXRuntimeObjectDetection(detector_config['color-model'], labels, detector_config.getfloat('prob_threshold', 0.10), scale=4)
    if 'vehicle-model' in detector_config:
        sd.notify("STATUS=Loading vehicle/packages model")
        if 'yolov4' in detector_config['vehicle-model']:
            vehicle_model = ONNXTensorRTv4ObjectDetection(detector_config['vehicle-model'], vehicle_labels, detector_config.getfloat('prob_threshold', 0.10), model_width=608, model_height=608)
        elif options.trt:
            vehicle_model = ONNXTensorRTObjectDetection(detector_config['vehicle-model'], vehicle_labels, detector_config.getfloat('prob_threshold', 0.10), scale=1)
        else:
            vehicle_model = ONNXRuntimeObjectDetection(detector_config['vehicle-model'], vehicle_labels, detector_config.getfloat('prob_threshold', 0.10), scale=1)
    sd.notify("STATUS=Loaded models")
   
    cams=[]
    i = 0
    while "cam%d" % i in config.sections():
        cams.append(Camera(config["cam%d" % i],excludes.get(config["cam%d" % i]['name'], {})))
        i += 1
    print("Configured %i cams" % i)
    async_cameras = len(list(filter(lambda cam: cam.capture_async, cams)))
    #async_cameras = 4
    print("%i async workers" % async_cameras)
    async_pool = concurrent.futures.ThreadPoolExecutor(max_workers=async_cameras)
    sync_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    sd.notify("READY=1")
    sd.notify("STATUS=Running")
    while True:
      sd.notify("WATCHDOG=1")
      start_time = timer()
      prediction_time = 0.0
      capture_futures = []
      messages=[]
      print("Checking ", end="")
      if options.sync:
          for cam in cams:
              cam.capture()
              p,m = detect(cam, color_model, grey_model, vehicle_model, config, st)
              prediction_time += p
              messages.append(m)
      else:
          for cam in cams:
              try:
                  if cam.capture_async and not options.sync:
                      capture_futures.append(async_pool.submit(cam.capture))
                  else:
                      capture_futures.append(sync_pool.submit(cam.capture))
              except KeyboardInterrupt:
                  return
              except requests.exceptions.ConnectionError:
                  print("cam:%s requests.exceptions.ConnectionError:" % cam.name, sys.exc_info()[0] )

          for f in concurrent.futures.as_completed(capture_futures, timeout=180):
              try:
                  cam = f.result()
                  if cam:
                      p,m = detect(cam, color_model, grey_model, vehicle_model, config, st)
                      prediction_time += p
                      messages.append(m)
              except KeyboardInterrupt:
                  return

      end_time = timer()
      print(",".join(sorted(messages)), end="")
      print('.. completed in %.2fs, spent %.2fs predicting' % ( (end_time - start_time), prediction_time ), flush=True)
      if prediction_time < 0.1 * len(cams):
          print("Cameras appear down, waiting 30 seconds")
          time.sleep(30)
      if 'once' in detector_config:
          break
    
if __name__ == '__main__':
    faulthandler.register(signal.SIGUSR1)
    # python 3.7 is asyncio.run()
    parser = argparse.ArgumentParser(description='Process cameras')
    parser.add_argument('--trt', action='store_true')
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('config_file', nargs='?', default='config.txt')
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(main(args))
