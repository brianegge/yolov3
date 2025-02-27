import json
import logging
import os
import re
import sys
import traceback
from datetime import date, datetime
from io import BytesIO
from pprint import pformat

import aiohttp
import requests

import codeproject

logger = logging.getLogger(__name__)

with open("license-plates.json") as f:
    license_plates = json.load(f)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def notify(cam, message, image, predictions, config, ha):
    mode = ha.mode()
    mode_key = "priority-%s" % mode
    if mode_key in config:
        mode_priorities = config[mode_key]
    else:
        mode_priorities = {}
    priorities = config["priority"]
    priority = None
    priority_type = None
    has_dog = False
    vehicles = list(
        filter(lambda p: p["tagName"] == "vehicle" and not "ignore" in p, predictions)
    )
    has_vehicles = len(vehicles) > 0
    has_visible_vehicles = len(
        list(
            filter(
                lambda p: p["tagName"] == "vehicle"
                and not "ignore" in p
                and not "departed" in p
                and p["age"] == 0,
                predictions,
            )
        )
    )
    people = list(
        filter(
            lambda p: p["tagName"] == "person" and not "ignore" in p,
            predictions,
        )
    )
    has_person = len(people) > 0
    has_dog = len(list(filter(lambda p: p["tagName"] == "dog", predictions))) > 0
    has_person_road = (
        len(list(filter(lambda p: p["tagName"] == "person_road", predictions))) > 0
    )
    has_dog_road = (
        len(list(filter(lambda p: p["tagName"] == "dog_road", predictions))) > 0
    )
    packages = list(
        filter(lambda p: p["tagName"] == "package" and not "departed" in p, predictions)
    )
    has_package = len(packages) > 0
    if has_vehicles and cam.name != "mailbox":
        notify_vehicle = ha.should_notify_vehicle()
        logging.info(f"ha.should_notify_vehicle={notify_vehicle}")
    else:
        notify_vehicle = False
    if has_person:
        notify_person = ha.should_notify_person()
        logging.info(f"see {len(people)} people, notify_person={notify_person}")
    else:
        notify_person = False
    # if notify_person:
    #    door_left = ha.get_door_left()
    #    door_right = ha.get_door_right()
    #    do_ignore = (
    #        cam.name in ["deck", "play"] and (door_left or door_left) and mode == "home"
    #    )
    #    if do_ignore:
    #        logging.info(
    #            f"Ignoring person ignore={do_ignore} because door left={door_left}, door_right={door_right}, cam={cam.name} and mode={mode}"
    #        )
    #        notify_person = False
    #        ha.suppress_notify_person()
    #    else:
    #        logging.info(
    #            f"Notifying person because door left={door_left}, door_right={door_right}, cam={cam.name} and mode={mode}"
    #        )
    # else:
    #    # If person detection is off, override night or away mode
    #    mode = "home"
    sound = "pushover"
    for p in list(filter(lambda p: "ignore" in p or "iou" in p, predictions)):
        p["priority"] = -4
    for p in list(filter(lambda p: not "priority" in p, predictions)):
        tagName = p["tagName"]
        probability = p["probability"]
        i_type = None
        if tagName == "person_road":
            if mode == "night" and ha.is_time_after_midnight_and_before_six():
                notify_person = True
                i = 0
            else:
                notify_person = False
                i = -4
        elif tagName == "dog_road":
            if has_person_road:
                i = -3
                i_type = "person walking dog"
            else:
                i = 1
                i_type = "dog without person"
        elif tagName == "vehicle" and cam.name == "front entry":
            i = -3
            i_type = "vehicle rule"
            # cars should not be possible here, unless in road
        elif tagName == "cat" and cam.name == "garage":
            i = -2
            i_type = "cat in garage rule"
        elif tagName == "person" and cam.name == "garage":
            i = -3
            i_type = "person in garage rule"
        elif tagName == "person" and has_person and not notify_person:
            i = -4
            # we are still outside, keep detection off
            # ha.suppress_notify_person()
            i_type = "person detection off"
        elif tagName == "deer" and has_person:
            i = -1
            # this should never occur
            i_type = "deer and person not possible"
        elif tagName == "vehicle" and not notify_vehicle:
            i = -4
            i_type = "vehicle detection off"
        elif tagName in mode_priorities:
            i = mode_priorities.getint(p["tagName"])
            i_type = mode_key
        elif tagName in priorities:
            i = priorities.getint(p["tagName"])
            i_type = "class {}".format(tagName)
        else:
            i_type = "default"
            i = 0
        if cam.name == "peach tree" and mode == "night" and i < 1:
            i = 1
            i_type = "fruit robber"
        if "departed" in p:
            sound = config["sounds"]["departed"]
        elif tagName in config["sounds"]:
            sound = config["sounds"][tagName]
        # if tagName == "dog" and (p["camName"] == "deck") and probability < 0.9:
        #    i = 0
        #    i_type = "maybe dog rule"
        if (
            tagName == "dog"
            and (p["camName"] == "garage")
            and probability > 0.9
            and not has_person
        ):
            i = 1
            i_type = "dog in garage rule"
        # elif tagName == "dog" and p["camName"] == "deck" and i > -3:
        #    i = -3
        #    i_type = f"{tagName} on {p['camName']}"
        if tagName in ["fox", "coyote"] and p["camName"] == "deck" and i < 1:
            i = 1
            i_type = f"{tagName} on {p['camName']}"
        if i is not None:
            p["priority"] = i
            p["priority_type"] = i_type
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
        priority_rule = "dog near package"
    if priority is None:
        for p in predictions:
            if "priority" in p:
                priority = p["priority"]
                prior_rule = "prior_priority"
                logging.info(f"Using prior priority={priority}")
    if priority is None:
        priority = 0
        priority_rule = "default"
        logging.info("Using default priority")

    # crop to area of interest
    width, height = image.size
    left = min(p["boundingBox"]["left"] - 0.05 for p in predictions) * width
    right = (
        max(
            p["boundingBox"]["left"] + p["boundingBox"]["width"] + 0.05
            for p in predictions
        )
        * width
    )
    top = min(p["boundingBox"]["top"] - 0.05 for p in predictions) * height
    bottom = (
        max(
            p["boundingBox"]["top"] + p["boundingBox"]["height"] + 0.05
            for p in predictions
        )
        * height
    )
    center_x = left + (right - left) / 2
    center_y = top + (bottom - top) / 2
    show_width = max(width / 4, right - left)
    show_height = max(height / 4, bottom - top)
    left = min(left, center_x - show_width / 2)
    left = max(0, left)
    top = max(0, top)
    if left + show_width > width:
        show_width = width - left
    if top + show_height > height:
        show_height = height - top
    top = min(top, center_y - show_height / 2)
    top = max(0, top)
    crop_rectangle = (left, top, left + show_width, top + show_height)
    # logging.info("Cropping to %d,%d,%d,%d" % crop_rectangle)
    cropped_image = image.crop(crop_rectangle)

    static_dir = os.path.join(config["detector"]["save-path"], "static")
    for p in predictions:
        cropped_image.save(os.path.join(static_dir, f"{p['tagName']}.jpg"))

    if priority <= -3:
        # logging.info('Ignoring "%s" with priority %s=%d' % (message, priority_type, priority) )
        return priority
    # else:
    #    logging.info('Notifying "%s" with priority %s=%d' % (message, priority_type, priority) )
    if has_visible_vehicles:
        logging.info(pformat(vehicles))
        left = max(0, min(p["boundingBox"]["left"] - 0.05 for p in vehicles) * width)
        right = min(
            width,
            max(
                p["boundingBox"]["left"] + p["boundingBox"]["width"] + 0.05
                for p in vehicles
            )
            * width,
        )
        top = max(0, min(p["boundingBox"]["top"] - 0.05 for p in vehicles) * height)
        bottom = min(
            height,
            max(
                p["boundingBox"]["top"] + p["boundingBox"]["height"] + 0.05
                for p in vehicles
            )
            * height,
        )
        crop_rectangle = (left, top, right, bottom)
        vehicle_image = image.crop(crop_rectangle)
        save_dir = os.path.join(
            config["detector"]["save-path"], date.today().strftime("%Y%m%d")
        )
        save_vehicle = os.path.join(
            save_dir,
            datetime.now().strftime("%H%M%S")
            + "-"
            + vehicles[0]["camName"].replace(" ", "_")
            + "-"
            + "codeproject.jpg",
        )
        vehicle_image.save(save_vehicle)
        vehicle_bytes = BytesIO()
        vehicle_image.save(vehicle_bytes, "jpeg")
        vehicle_bytes.seek(0)
        try:
            save_json = os.path.join(
                save_dir,
                datetime.now().strftime("%H%M%S")
                + "-"
                + vehicles[0]["camName"].replace(" ", "_")
                + "-"
                + "codeproject.txt",
            )
            enrichments = codeproject.enrich(vehicle_bytes.read(), save_json)
            make = None
            plate_name = None
            vehicle_message = ""
            if enrichments["count"] == 0:
                # Don't announce if sighthound can't find a vehicle
                notify_vehicle = False
            for plate in enrichments["plates"]:
                guesses = (
                    [plate]
                    + [plate.replace(" ", "")]
                    + list(edits1(plate))
                    + list(edits2(plate))
                )
                for guess in guesses:
                    if guess in license_plates:
                        if len(vehicle_message) > 0:
                            vehicle_message += " and "
                        r = license_plates[guess]
                        if "owner" in r:
                            vehicle_message += r["owner"] + "'s "
                            if r["owner"].lower() == "house cleaner":
                                ha.house_cleaners_arrived()
                        if "color" in r:
                            vehicle_message += r["color"] + " "
                        if "make" in r:
                            vehicle_message += r["make"] + " "
                            if "model" in r:
                                vehicle_message += r["model"]
                        else:
                            vehicle_message += "vehicle"
                        if r.get("announce", True) == False:
                            logging.info(
                                "Ignoring {}'s vehicle with plate {}".format(
                                    owner, plate
                                )
                            )
                            vehicle_message = None
                        break
                if vehicle_message is not None:
                    if vehicle_message == "":
                        vehicle_message = "Vehicle"
                    if notify_vehicle:
                        if cam.name == "shed":
                            ha.echo_speaks(f"{vehicle_message} in front of garage")
                        else:
                            ha.echo_speaks(f"{vehicle_message} in driveway")
                    # don't announce plate
                    message += "\n" + vehicle_message + " " + plate

        except:
            logging.exception("Failed to enrich via codeproject")

    #    if has_package and (priority >= 0 or has_dog):
    #        prob = max(map(lambda x: x["probability"], packages))
    #        logging.info(
    #            f"has_package={has_package}, has_dog={has_dog}, prob={prob}, mode={mode}"
    #        )
    #        # if has_package and has_dog:
    #        #    ha.echo_speaks(
    #        #        "Rufus is opening package near {}".format(packages[0]["camName"])
    #        #    )
    #        if prob > 0.9:
    #            if len(packages) == 1:
    #                ha.echo_speaks(
    #                    "Package delivered near {}".format(packages[0]["camName"])
    #                )
    #            else:
    #                ha.echo_speaks(
    #                    "{} packages delivered near {}".format(
    #                        len(packages), packages[0]["camName"]
    #                    )
    #                )
    #        else:
    #            logging.info(
    #                "Not speaking package delivery because probability {} < 0.9".format(
    #                    prob
    #                )
    #            )
    #            if not has_dog:
    #                priority = -1

    if priority >= -3 and ha.vacation_mode() == False:
        # prepare post
        output_bytes = BytesIO()
        cropped_image.save(output_bytes, "jpeg")
        output_bytes.seek(0)
        # send as -2 to generate no notification/alert, -1 to always send as a quiet notification, 1 to display as high-priority and bypass the user's quiet hours, or 2 to also require confirmation from the user
        try:
            r = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": config["pushover"]["token"],
                    "user": "uzziquh6d7a4vyouise2ti482gc1pq",
                    "message": message,
                    "priority": priority,
                    "sound": sound,
                },
                files={"attachment": ("image.jpg", output_bytes, "image/jpeg")},
            )
            if r.status_code != 200:
                logging.warning(pformat(r))
                logging.warning(pformat(r.headers))
        except Exception:
            logger.exception("Failed to call Pushover")

    return priority
