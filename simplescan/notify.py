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
import re
#import logging

#logging.basicConfig(level=logging.DEBUG)

with open('license-plates.json') as f:
    license_plates = json.load(f)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    #transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def notify(cam, message, image, predictions, config, st):
    mode = st.get_st_mode()
    mode_key = 'priority-%s' % mode
    if mode_key in config:
        mode_priorities = config[mode_key]
    else:
        mode_priorities = {}
    priorities = config['priority']
    priority = None
    priority_type = None
    has_dog = False
    vehicles = list(filter(lambda p: p['tagName'] == 'vehicle' and not 'ignore' in p, predictions))
    has_vehicles = len(vehicles) > 0
    has_person  = len(list(filter(lambda p: p['tagName'] == 'person' and not 'ignore' in p, predictions))) > 0
    has_dog  = len(list(filter(lambda p: p['tagName'] == 'dog', predictions))) > 0
    packages = list(filter(lambda p: p['tagName'] == 'package', predictions))
    has_package = len(packages) > 0
    if has_vehicles:
        notify_vehicle = st.should_notify_vehicle()
    else:
        notify_vehicle = False
    if has_person:
        notify_person = st.should_notify_person()
    else:
        notify_person = False
    sound = 'pushover'
    for p in list(filter(lambda p: not 'ignore' in p, predictions)):
        tagName = p['tagName']
        probability = p['probability']
        i_type = None
        if tagName == 'vehicle' and cam.name == 'front entry':
            i = -3
            i_type = 'vehicle rule'
            # cars should not be possible here, unless in road
        elif tagName == 'person' and cam.name == 'garage':
            i = -3
            i_type = 'person in garage rule'
        elif tagName == 'person' and not notify_person:
            i = -2
            # we are still outside, keep detection off
            st.suppress_notify_person()
            i_type = 'person detection off'
        elif tagName == 'vehicle' and not notify_vehicle:
            i = -2
            i_type = 'vehicle detection off'
        elif tagName in mode_priorities:
            i = mode_priorities.getint(p['tagName'])
            i_type = mode_key
        elif tagName in priorities:
            i = priorities.getint(p['tagName'])
            i_type = 'class {}'.format(tagName)
        else:
            i_type = 'default'
            i = 0
        if tagName in config['sounds']:
            sound = config['sounds'][tagName]
        if tagName == 'dog' and p['camName'] == 'garage' and probability > 0.9 and not has_person:
            i = 1
            i_type = 'dog in garage rule'
        if i is not None:
            if priority is None:
                priority = i
                priority_type = i_type
            else:
                priority = max(i, priority)
                if i >= priority:
                    priority_type = i_type
    # raise priority if dog is near package
    if has_package and has_dog:
        priority = 1
        priority_rule = 'dog near package'
    if priority is None:
        priority = 0
        priority_rule = 'default'

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
    # print("Cropping to %d,%d,%d,%d" % crop_rectangle)
    cropped_image = image.crop(crop_rectangle)

    if priority <= -3:
        print('Ignoring "%s" with priority %s=%d' % (message, priority_type, priority) )
        return priority
    else:
        print('Notifying "%s" with priority %s=%d' % (message, priority_type, priority) )
    if has_vehicles:
        pprint(vehicles)
        st.turn_on_outside_lights()
        left = max(0, min(p['boundingBox']['left'] - 0.05 for p in vehicles) * width)
        right = min(width, max(p['boundingBox']['left'] + p['boundingBox']['width'] + 0.05 for p in vehicles) * width)
        top = max(0, min(p['boundingBox']['top'] - 0.05 for p in vehicles) * height)
        bottom = min(height, max(p['boundingBox']['top'] + p['boundingBox']['height'] + 0.05 for p in vehicles) * height)
        crop_rectangle = (left, top, right, bottom)
        print("Cropping vehicle to {}".format(crop_rectangle))
        vehicle_image = cam.image.crop(crop_rectangle)
        save_dir = os.path.join(config['detector']['save-path'],date.today().strftime("%Y%m%d"))
        save_vehicle = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + vehicles[0]['camName'] + "-" + "sighthound.jpg")
        vehicle_image.save(save_vehicle)
        vehicle_bytes = BytesIO()
        vehicle_image.save(vehicle_bytes, 'jpeg')
        vehicle_bytes.seek(0)
        try:
            save_json = os.path.join(save_dir,datetime.now().strftime("%H%M%S") + "-" + vehicles[0]['camName'] + "-" + "sighthound.txt")
            enrichments = sighthound.enrich(vehicle_bytes.read(), save_json)
            owner = None
            make = None
            plate_name = None
            if enrichments['count'] == 0:
                # Don't announce if sighthound can't find a vehicle
                notify_vehicle = False
            for plate in enrichments['plates']:
                guesses = [plate['name']] + list(edits1(plate['name'])) + list(edits2(plate['name']))
                for guess in guesses:
                    if guess in license_plates:
                        owner = license_plates[guess].get('owner')
                        make = license_plates[guess].get('make')
                        if license_plates[guess].get("announce", True) == False:
                            print('Ignoring {}\'s vehicle with plate {}'.format(owner, plate))
                            return
                        if owner.lower() == "house cleaner":
                            st.set_st_scene("House Cleaning")
                            st.open_garage_door()
                        break
                if owner is None and plate['confidence'] > 0.05:
                    plate_name = plate['name']
                    if 'state' in plate:
                        plate_name = "{} {}".format(plate['state'], plate_name)
            if len(enrichments['message']) > 0:
                vehicle_message = enrichments['message']
                if owner is not None:
                    vehicle_message = owner + "'s " + vehicle_message
                elif make is not None:
                    vehicle_message = make
                if notify_vehicle:
                    if vehicles[0]['camName'] == 'shed':
                        st.echo_speaks('Vehicle in front of garage: ' + vehicle_message)
                    else:
                        st.echo_speaks('Vehicle in driveway: ' + vehicle_message)
                # don't announce plate
                if plate_name is not None:
                    vehicle_message += ' ' + plate_name
                if 'vehicle' in message:
                    message = re.sub('vehicle', vehicle_message, message)
        except:
            traceback.print_exc(file=sys.stdout)
            print('Failed to enrich via sighthound')

    if has_package and priority >= 0:
        prob = max(map(lambda x: x['probability'], packages))
        if prob > 0.9:
            if len(packages) == 1:
                st.echo_speaks('Package delivered near {}'.format(packages[0]['camName']))
            else:
                st.echo_speaks('{} packages delivered near {}'.format(len(packages), packages[0]['camName']))
        else:
            print("Not speaking package delivery because probability {} < 0.9".format(prob))
            if not has_dog:
                priority = -3

    # prepare post
    output_bytes = BytesIO()
    cropped_image.save(output_bytes, 'jpeg')
    output_bytes.seek(0)
    print("Sending Pushover message '{}'".format(message), flush=True)
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
        pprint(r.headers)
    return priority
