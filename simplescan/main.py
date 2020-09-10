# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import os
import sys
import configparser
from timeit import default_timer as timer
from detect import ONNXRuntimeObjectDetection,detect,Camera
from object_detection_rt import ONNXTensorRTObjectDetection
import asyncio
import concurrent.futures
from datetime import datetime
import requests
import faulthandler, signal
import argparse

async def main(options):
    config = configparser.ConfigParser()
    config.read(options.config_file)
    detector_config = config['detector']

    # Load labels
    with open(detector_config['labelfile-path'], 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    if 'vehicle-labelfile-path' in detector_config:
        with open(detector_config['vehicle-labelfile-path'], 'r') as f:
            vehicle_labels = [l.strip() for l in f.readlines()]

    vehicle_model = None
    if options.trt:
        od_model = ONNXTensorRTObjectDetection(detector_config['onnx-file'], labels, detector_config.getfloat('prob_threshold', 0.10))
        if 'vehicle-model' in detector_config:
            vehicle_model = ONNXTensorRTObjectDetection(detector_config['vehicle-model'], vehicle_labels, detector_config.getfloat('prob_threshold', 0.10))
    else:
        od_model = ONNXRuntimeObjectDetection(detector_config['onnx-file'], labels, detector_config.getfloat('prob_threshold', 0.10))
        if 'vehicle-model' in detector_config:
            vehicle_model = ONNXRuntimeObjectDetection(detector_config['vehicle-model'], vehicle_labels, detector_config.getfloat('prob_threshold', 0.10))
    print("Loaded models")
   
    cams=[]
    i = 0
    while "cam%d" % i in config.sections():
        cams.append(Camera(config["cam%d" % i]))
        i += 1
    print("Configured %i cams" % i)
    pool = concurrent.futures.ThreadPoolExecutor()
    session = requests.Session()
    
    while True:
      start_time = timer()
      prediction_time = 0.0
      futures = []
      print("Checking ", end="")
      for cam in cams:
          try:
              raw_image = cam.capture(session)
              if options.sync:
                  prediction_time += detect(cam, raw_image, od_model, vehicle_model, config)
              else:
                  futures.append(pool.submit(detect, cam, raw_image, od_model, vehicle_model, config))
          except requests.exceptions.ConnectionError:
              print("cam:%s requests.exceptions.ConnectionError:" % cam.name, sys.exc_info()[0] )

      for f in futures:
          try:
              prediction_time += f.result(timeout=90)
          except KeyboardInterrupt:
              return
          #except:
          #  print("Unexpected error:", sys.exc_info()[0])
      end_time = timer()
      print('.. completed in %.2fs, spent %.2fs predicting' % ( (end_time - start_time), prediction_time ) )
    
if __name__ == '__main__':
    faulthandler.register(signal.SIGUSR1)
    # python 3.7 is asyncio.run()
    parser = argparse.ArgumentParser(description='Process cameras')
    parser.add_argument('--trt', action='store_true')
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('config_file', nargs='?', default='config.txt')
    args = parser.parse_args()
    asyncio.get_event_loop().run_until_complete(main(args))
