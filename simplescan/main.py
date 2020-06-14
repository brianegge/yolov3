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
import asyncio
import concurrent.futures

async def main(options):
    config = configparser.ConfigParser()
    config.read("config.txt")
    detector_config = config['detector']

    # Load labels
    with open(detector_config['labelfile-path'], 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = ONNXRuntimeObjectDetection(detector_config['onnx-file'], labels)
   
    cams=[]
    i = 0
    while "cam%d" % i in config.sections():
        cams.append(Camera(config["cam%d" % i]))
        i += 1
    print("Configured %i cams" % i)
    pool = concurrent.futures.ThreadPoolExecutor()
    
    while True:
      start_time = timer()
      prediction_time = 0.0
      futures = []
      print("Checking ", end="")
      for cam in cams:
          #if cam.name != 'garage':
          #    continue
          raw_image = cam.capture()
          futures.append(pool.submit(detect, cam, raw_image, od_model, config))

      for f in futures:
          try:
              prediction_time += f.result()
          except KeyboardInterrupt:
              return
          #except:
          #  print("Unexpected error:", sys.exc_info()[0])
      end_time = timer()
      print('.. completed in %.2fs, spent %.2fs predicting' % ( (end_time - start_time), prediction_time ) )
    
if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:]))
