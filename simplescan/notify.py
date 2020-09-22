import requests
from pprint import pprint
from io import BytesIO
import aiohttp
import json
import traceback
from datetime import date,datetime
import sys
import os
sys.path.insert(0,'/home/egge/detector/simplescan/pysmartthings/pysmartthings')
import pysmartthings
import sighthound
from urllib.parse import quote
#import logging

#logging.basicConfig(level=logging.DEBUG)


def get_st_mode(config):
    st_config = config['smartthings']
    token = st_config['token']
    r = requests.get('http://raspberrypi-zerow.local:8282/mode')
    return r.content.decode("utf-8").lower()

def echo_speaks(config, message):
    st_config = config['smartthings']
    if 'echo_speaks' in st_config:
        url = st_config['echo_speaks']
        requests.get(url + quote(message))

def notify(message, image, predictions, config):
    mode = get_st_mode(config)
    mode_key = 'priority-%s' % mode
    if mode_key in config:
        mode_priorities = config[mode_key]
    else:
        mode_priorities = {}
    priorities = config['priority']
    priority = None
    has_package = False
    has_dog = False
    sound = 'pushover'
    for p in predictions:
        tagName = p['tagName']
        if tagName == 'dog':
            has_dog = True
        elif tagName == 'package':
            has_package = True
        if tagName in mode_priorities:
            i = mode_priorities.getint(p['tagName'])
            print('mode priority %s=%d' % (mode_key,i))
        elif tagName in priorities:
            i = priorities.getint(p['tagName'])
            print('config priority=%d' % i)
        else:
            print('default priority for %s' % p['tagName'])
            i = None
        if tagName in config['sounds']:
            sound = config['sounds'][tagName]
        if i is not None:
            if p['tagName'] == 'dog' and p['camName'] == 'garage':
                priority = 1
            if priority is None:
                priority = i
            else:
                priority = max(i, priority)
    # raise priority if dog is near package
    if has_package and has_dog:
        priority = max(1, priority)

    # crop to area of interest
    width, height = image.size
    left = min(p['boundingBox']['left'] - 0.05 for p in predictions) * width
    right = max(p['boundingBox']['left'] + p['boundingBox']['width'] + 0.05 for p in predictions) * width
    top = min(p['boundingBox']['top'] - 0.05 for p in predictions) * height
    bottom = max(p['boundingBox']['top'] + p['boundingBox']['height'] + 0.05 for p in predictions) * height
    center_x = left + (right - left) / 2
    center_y = top + (bottom - top) / 2
    show_width = max(width / 4, right - left)
    show_height = max(height / 4, bottom - top)
    left = min(left, center_x - show_width / 2)
    left = max(0,left)
    top = max(0,top)
    if left + show_width > width:
        show_width = width - left
    if top + show_height > height:
        show_height = height - top
    top = min(top, center_y - show_height / 2)
    top = max(0,top)
    crop_rectangle = (left, top, left + show_width, top + show_height)
    print("Cropping to %d,%d,%d,%d" % crop_rectangle)
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
    vehicles = list(filter(lambda p: p['tagName'] == 'vehicle', predictions))
    if len(vehicles):
        if len(notify.license_plates) == 0:
            with open(config['detector']['license-plates']) as f:
                notify.license_plates = json.load(f)
        try:
            save_dir = os.path.join(config['detector']['save-path'],date.today().strftime("%Y%m%d"))
            save_json = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + vehicles[0]['camName'] + "-" + "sighthound.txt")
            enrichments = sighthound.enrich(output_bytes.read(), save_json)
            owner = None
            make = None
            plate_name = None
            for plate in enrichments['plates']:
                if plate['name'] in notify.license_plates:
                    owner = notify.license_plates[plate['name']].get('owner')
                    make = notify.license_plates[plate['name']].get('make')
                    if notify.license_plates[plate['name']].get("announce", True) == False:
                        print('Ignoring {}\'s vehicle with plate {}'.format(owner, plate))
                        return
                elif plate['confidence'] > 0.05:
                    plate_name = plate['name']
            if len(enrichments['message']) > 0:
                message = enrichments['message']
                if owner is not None:
                    message = owner + "'s " + message
                elif make is not None:
                    message = make
                echo_speaks(config, 'Vehicle in driveway: ' + message)
                # don't announce plate
                if plate_name is not None:
                    message += ' ' + plate_name
        except:
            traceback.print_exc(file=sys.stdout)
            print('Failed to enrich via sighthound')
        output_bytes.seek(0)

    packages = list(filter(lambda p: p['tagName'] == 'package', predictions))
    for package in packages:
        echo_speaks(config, 'Package delivered near {}'.format(package['camName']))

    print("Sending Pushover message '{}'".format(message))
    r = requests.post("https://api.pushover.net/1/messages.json", data = {
      "token": "ahyf2ozzhdb6a8ie95bdvvfwenzuox",
      "user": "uzziquh6d7a4vyouise2ti482gc1pq",
      "message": message,
      "priority": priority,
      "sound": sound
      },
      files = {
          "attachment": ("image.jpg", output_bytes, "image/jpeg")
    })
    if r.status_code != 200:
        pprint(r)

notify.license_plates = {}
