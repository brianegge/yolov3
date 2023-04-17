import datetime
import logging
from pprint import pprint
from urllib.parse import quote, urljoin

import requests

log = logging.getLogger(__name__)


class HomeAssistant(object):
    """interface with HomeAssistant"""

    def __init__(self, config):
        self.config = config
        self.headers = {"Authorization": "Bearer {}".format(self.config["token"])}
        self.api = "http://homeassistant.home:8123/api/"
        response = requests.get(self.api, headers=self.headers)
        message = response.json()["message"]
        log.debug(f"Home Assistant {message}")
        assert message == "API running."
        self.names = {}
        response = requests.get(urljoin(self.api, "states"), headers=self.headers)
        for entity in response.json():
            if "friendly_name" in entity["attributes"]:
                self.names[entity["attributes"]["friendly_name"]] = entity
        self.last_house_cleaners_arrived = None

    def set_scene(self, scene):
        r = requests.post(
            f"{self.api}services/scene/turn_on",
            json={"entity_id": f"scene.{scene}"},
            headers=self.headers,
        )
        log.info(f"Turned on scene {scene}={r}")

    def open_garage_door(self):
        raise NotImplementedError

    def crack_garage_door(self):
        if not self.get_state("input_boolean.moon_present"):
            log.debug("Cracking garage door for Moon")
            r = requests.post(
                f"{self.api}services/script/turn_on",
                json={"entity_id": "script.crack_garage_door"},
                headers=self.headers,
            )
            return r.content.decode("utf-8")
        else:
            log.debug("Moon has already returned")

    def let_cat_out(self):
        log.debug("Letting Moon of out the garage")
        r = requests.post(
            f"{self.api}services/script/turn_on",
            json={"entity_id": "script.let_cat_out"},
            headers=self.headers,
        )
        return r.content.decode("utf-8")

    def deer_alert(self, location):
        # invoke webcore piston
        json = {
            "entity_id": "script.deer",
            "variables": {"location": location},
        }
        r = requests.post(
            f"{self.api}services/script/turn_on", json=json, headers=self.headers
        )
        log.info(f"Deer alert={r}")

    def get_device(self, name):
        try:
            return self.devices[name.lower()]
        except KeyError:
            raise KeyError('No such device "{}"'.format(name))

    def get_state(self, entity):
        response = requests.get(
            f"{self.api}states/{entity}", headers=self.headers
        ).json()
        if "state" in response:
            return response["state"] == "on"
        log.debug(response)
        raise RuntimeError(f"Invalid response from entity {entity}={response}")

    def set_input_boolean(self, switch, state):
        json = {
            "entity_id": switch,
        }
        if state:
            response = requests.post(
                f"{self.api}services/input_boolean/turn_on",
                json=json,
                headers=self.headers,
            )
        else:
            response = requests.post(
                f"{self.api}services/input_boolean/turn_off",
                json=json,
                headers=self.headers,
            )
        if response.status_code != 200:
            log.warning(f"Set input_boolean {switch} to {state}={response.content}")
        return response

    def set_notify_vehicle(self, state):
        return self.set_input_boolean("input_boolean.vehicle_detector", state)

    def get_door_left(self):
        return self.get_state("binary_sensor.konnected_198e05_zone_4")

    def get_door_right(self):
        return self.get_state("binary_sensor.konnected_198e05_zone_5")

    def get_presence(self, person):
        response = requests.get(
            f"{self.api}states/{person}", headers=self.headers
        ).json()
        return response["state"] == "home"

    def should_notify_vehicle(self):
        return self.get_state("input_boolean.vehicle_detector")

    def should_notify_person(self):
        return self.get_state("input_boolean.person_detector")

    def echo_speaks(self, message):
        if self.get_presence("group.egge"):
            log.info("Speaking {}".format(message))
            json = {"message": message, "data": {"type": "tts"}}
            r = requests.post(
                f"{self.api}services/notify/alexa_media_kitchen_ecobee4",
                json=json,
                headers=self.headers,
            )
            return r.content.decode("utf-8")
        else:
            log.info("Not speaking {}".format(message))
            return True

    def mode(self):
        if self.get_state("input_boolean.night_mode"):
            return "night"
        elif self.get_presence("group.everyone"):
            return "home"
        else:
            return "away"

    def suppress_notify_person(self):
        log.debug("Keep person notify suppressed")
        r = requests.post(
            f"{self.api}services/script/turn_on",
            json={"entity_id": "script.pause_person_detector"},
            headers=self.headers,
        )
        return r.content.decode("utf-8")

    def house_cleaners_arrived(self):
        if (
            self.last_house_cleaners_arrived is None
            or datetime.datetime.now() - self.last_house_cleaners_arrived
            > datetime.timedelta(days=1)
        ):
            r = requests.post(
                f"{self.api}services/script/turn_on",
                json={"entity_id": "script.house_cleaners_arrive"},
                headers=self.headers,
            )
            log.info(f"Run script house cleaners arrive={r}")
            self.last_house_cleaners_arrived = datetime.datetime.now()
        else:
            log.info(f"House cleaners last arrrived {self.last_house_cleaners_arrived}")
