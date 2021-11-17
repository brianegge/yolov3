#!/usr/bin/env python3
import configparser
from homeassistant import HomeAssistant

config = configparser.ConfigParser()
config.read("config.txt")

ha = HomeAssistant(config["homeassistant"])
# ha.set_scene("reading")
def test_should_notify_vehicle():
    assert ha.should_notify_vehicle() == True


def test_should_notify_person():
    assert type(ha.should_notify_person()) == bool


def test_door_contacts():
    door_left = ha.get_switch("binary_sensor.door_left_contact")


def test_mode():
    assert ha.mode() in ["home","night"]


# ha.turn_on_outside_lights()
# print("open_garage_door={}".format(st.open_garage_door()))
# ha.deer_alert('tree line')
#print(ha.echo_speaks("hello world"))
