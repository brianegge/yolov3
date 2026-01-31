import datetime
from urllib.parse import quote

import requests


class SmartThings(object):
    """interface with SmartThings"""

    def __init__(self, config):
        self.config = config["smartthings"]
        self.headers = {"Authorization": "Bearer {}".format(self.config["token"])}
        response = requests.get(
            "https://api.smartthings.com/v1/scenes", headers=self.headers
        )
        self.scenes = response.json()["items"]
        self.scene_last_set = {}
        response = requests.get(
            "https://api.smartthings.com/v1/devices", headers=self.headers
        )
        self.devices = {
            device["label"].lower(): device for device in response.json()["items"]
        }
        print(
            "SmartThings started with {} scenes and {} devices".format(
                len(self.scenes), len(self.devices)
            )
        )

    def set_st_scene(self, scene):
        # curl -H "Authorization: Bearer <>" https://api.smartthings.com/v1/scenes | python -m json.tool
        for s in self.scenes:
            if s["sceneName"] == scene:
                if self.scene_last_set.get(scene) == datetime.date.today():
                    print("Already set scene {sceneName} today")
                    return
                else:
                    print("Setting scene={sceneName}".format(**s))
                    r = requests.post(
                        "https://api.smartthings.com/v1/scenes/{sceneId}/execute".format(
                            **s
                        ),
                        headers=self.headers,
                    )
                    self.scene_last_set[scene] = datetime.date.today()
                    return r.content.decode("utf-8")
        print("Failed to find scene={}".format(scene))

    # def open_garage_door(self):
    #    if self.get_contactSensor_value('Garage Door Right') == "closed":
    #        device = self.get_device('Garage Opener Right')
    #        json=[{"component":"main","capability":"momentary","command":"push", "arguments":[]}]
    #        r = requests.post('https://api.smartthings.com/v1/devices/{deviceId}/commands'.format(**device), headers=self.headers, json=json)
    #        return r.content.decode("utf-8")

    def deer_alert(self):
        print("Deer alert!")
        # invoke webcore piston
        r = requests.get(self.config["deer_alert"])
        if r.json()["result"] != "OK":
            print("Failed to alert for deer")
            print(r.text)

    def get_device(self, name):
        try:
            return self.devices[name.lower()]
        except KeyError:
            raise KeyError('No such device "{}"'.format(name))

    # def get_contactSensor_value(self, sensor):
    #    device = self.get_device(sensor)
    #    response = requests.get('https://api.smartthings.com/v1/devices/{deviceId}/components/main/capabilities/contactSensor/status'.format(**device), headers=self.headers).json()
    #    if 'error' in response:
    #        print("response={}".format(response))
    #    return response['contact']['value']

    # def get_switch_value(self, switch):
    #    device = self.get_device(switch)
    #    response = requests.get('https://api.smartthings.com/v1/devices/{deviceId}/components/main/capabilities/switch/status'.format(**device), headers=self.headers).json()
    #    if 'error' in response:
    #        print("response={}".format(response))
    #    return response['switch']['value'] == "on"

    # def should_notify_vehicle(self):
    #    return self.get_switch_value('Vehicle Detector')

    # def should_notify_person(self):
    #    return self.get_switch_value('Person Detector')

    def echo_speaks(self, message):
        url = self.config["echo_speaks"]
        print("Speaking {}".format(message))
        r = requests.get(url + quote(message))
        return r.content.decode("utf-8")

    # def get_st_mode(self):
    #    r = requests.get('http://raspberrypi-zerow.home:8282/mode')
    #    c = r.content.decode("utf-8")
    #    return c.lower()
