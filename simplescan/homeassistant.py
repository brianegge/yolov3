import requests
import datetime
from pprint import pprint
from urllib.parse import quote
from urllib.parse import urljoin

class HomeAssistant(object):
    """ interface with HomeAssistant
    """

    def __init__(self, config):
        self.config = config
        self.headers = {"Authorization": "Bearer {}".format(self.config['token'])}
        self.api = 'http://homeassistant.home:8123/api/'
        response = requests.get(self.api, headers=self.headers)
        message = response.json()['message']
        print(f"Home Assistant {message}")
        assert message == "API running."
        self.names = {}
        response = requests.get(urljoin(self.api, "states"), headers=self.headers)
        for entity in response.json():
            if "friendly_name" in entity["attributes"]:
                self.names[entity["attributes"]["friendly_name"]] = entity


    def set_scene(self, scene):
        r = requests.post(f"{self.api}services/scene/turn_on", json={"entity_id":f"scene.{scene}"}, headers=self.headers)
        print(f"Turned on scene {scene}={r}")

    def open_garage_door(self):
        raise NotImplementedError

    def crack_garage_door(self):
        raise NotImplementedError

    def deer_alert(self, location):
        # invoke webcore piston
        json={"entity_id":"script.deer",
              "variables":{
                  "location":location
              },
              }
        r = requests.post(f"{self.api}services/script/turn_on", json=json, headers=self.headers)
        print(f"Deer alert={r}")

    def get_device(self, name):
        try:
            return self.devices[name.lower()]
        except KeyError:
            raise KeyError("No such device \"{}\"".format(name))

    def get_switch(self, switch):
        response = requests.get(f"{self.api}states/{switch}", headers=self.headers).json()
        return response['state'] == "on"

    def get_presence(self, person):
        response = requests.get(f"{self.api}states/{person}", headers=self.headers).json()
        return response['state'] == 'home'

    def should_notify_vehicle(self):
        return self.get_switch('switch.vehicle_detector')

    def should_notify_person(self):
        return self.get_switch('switch.person_detector')

    def echo_speaks(self, message):
        print("Speaking {}".format(message))
        json = {
            "message": message,
            "data":{
                "type":"tts"
                }
            }
        r = requests.post(f"{self.api}services/notify/alexa_media_kitchen_ecobee4", json=json, headers=self.headers)
        return r.content.decode("utf-8")

    def mode(self):
        if self.get_switch('input_boolean.night_mode'):
            return 'night'
        elif self.get_presence('group.everyone'):
            return 'home'
        else:
            return 'away'
    
    def suppress_notify_person(self):
        r = requests.get(self.config['suppress_notify_person'])
        c = r.content.decode("utf-8")
        print("Keep person notify suppressed={}".format(c))
        return r.content.decode("utf-8")

    def turn_on_outside_lights(self):
        r = requests.post(f"{self.api}services/script/turn_on", json={"entity_id":"script.all_exterior_lights_on_bright"}, headers=self.headers)
        print(f"Turned on outside lights {r}")

    def house_cleaners_arrived(self):
        r = requests.post(f"{self.api}services/script/turn_on", json={"entity_id":"script.house_cleaners_arrive"}, headers=self.headers)
        print(f"Run script house cleaners arrive={r}")

