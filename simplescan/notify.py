import requests
from pprint import pprint
from io import BytesIO
import aiohttp
import json
import jsonpickle
import sys
sys.path.insert(0,'/home/egge/detector/simplescan/pysmartthings/pysmartthings')
import pysmartthings
#import logging

#logging.basicConfig(level=logging.DEBUG)


def get_st_mode(config):
    st_config = config['smartthings']
    token = st_config['token']
    r = requests.get('http://raspberrypi-zerow:8282/mode')
    return r.content.decode("utf-8").lower()

def notify(message, image, predictions, config):
    mode = get_st_mode(config)
    mode_key = 'priority-%s' % mode
    if mode_key in config:
        mode_priorities = config[mode_key]
    else:
        mode_priorities = {}
    priorities = config['priority']
    priority = None
    for p in predictions:
        if p['tagName'] in mode_priorities:
            i = mode_priorities.getint(p['tagName'])
            print('mode priority %s=%d' % (mode_key,i))
        elif p['tagName'] in priorities:
            i = priorities.getint(p['tagName'])
            print('config priority=%d' % i)
        else:
            print('default priority for %s' % p['tagName'])
            i = None
        if i is not None:
            if p['tagName'] == 'dog' and p['camName'] == 'garage':
                priority = 1
            if priority is None:
                priority = i
            else:
                priority = max(i, priority)

    # crop to area of interest
    width, height = image.size
    left = min(p['boundingBox']['left'] for p in predictions) * width
    right = max(p['boundingBox']['left'] + p['boundingBox']['width'] for p in predictions) * width
    top = min(p['boundingBox']['top'] for p in predictions) * height
    bottom = max(p['boundingBox']['top'] + p['boundingBox']['height'] for p in predictions) * height
    center_x = left + (right - left) / 2
    center_y = top + (bottom - top) / 2
    left = min(max(0,center_x - width / 4), width / 2)
    top = min(max(0,center_y - height / 4), height / 2)
    crop_rectangle = (left, top, max(left + width / 2, right), max(top + height / 2, bottom))
    cropped_image = image.crop(crop_rectangle)

    if priority is None:
        priority = 0
    if priority <= -3:
        print('Ignoring "%s" with priority %d, mode %s' % (message,priority, mode) )
        return
    else:
        print('Notifying "%s" with priority %d, mode %s' % (message,priority, mode) )
    # prepare post
    output_bytes = BytesIO()
    cropped_image.save(output_bytes, 'jpeg')
    output_bytes.seek(0)
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
      "token": "ahyf2ozzhdb6a8ie95bdvvfwenzuox",
      "user": "uzziquh6d7a4vyouise2ti482gc1pq",
      "message": message,
      "priority": priority
    },
    files = {
      "attachment": ("image.jpg", output_bytes, "image/jpeg")
    })
    if r.status_code != 200:
        pprint(r)
