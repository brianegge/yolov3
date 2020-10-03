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
    r = requests.get('http://raspberrypi-zerow.local:8282/mode')
    return r.content.decode("utf-8").lower()

def get_garage_lights(config):
    r = requests.get('http://raspberrypi-zerow.local:8282/device/Garage Lights')
    return r.content.decode("utf-8").lower()

def echo_speaks(config, message):
    st_config = config['smartthings']
    if 'echo_speaks' in st_config:
        url = st_config['echo_speaks']
        requests.get(url + quote(message))

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz0123456789'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    #transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

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
    vehicles = list(filter(lambda p: p['tagName'] == 'vehicle', predictions))
    has_vehicles = len(vehicles) > 0
    if has_vehicles:
        notify_vehicle = get_garage_lights(config) != "on"
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
        if has_vehicles and not notify_vehicle:
            print("Garage light on - ignore own vehicle")
            priority = -2
        else:
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
    if has_vehicles:
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
                guesses = [plate['name']] + list(edits1(plate['name'])) + list(edits2(plate['name']))
                for guess in guesses:
                    if guess in notify.license_plates:
                        owner = notify.license_plates[guess].get('owner')
                        make = notify.license_plates[guess].get('make')
                        if notify.license_plates[guess].get("announce", True) == False:
                            print('Ignoring {}\'s vehicle with plate {}'.format(owner, plate))
                            return
                        break
                if owner is None and plate['confidence'] > 0.05:
                    plate_name = plate['name']
                    if 'state' in plate:
                        plate_name = "{} {}".format(plate['state'], plate_name)
            if len(enrichments['message']) > 0:
                message = enrichments['message']
                if owner is not None:
                    message = owner + "'s " + message
                elif make is not None:
                    message = make
                if notify_vehicle:
                    if vehicles[0]['camName'] == 'shed':
                        echo_speaks(config, 'Vehicle in front of garage: ' + message)
                    else:
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
