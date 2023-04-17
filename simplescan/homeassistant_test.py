#!/usr/bin/env python3
import configparser

from homeassistant import HomeAssistant

config = configparser.ConfigParser()
config.read("config.txt")

ha = HomeAssistant(config["homeassistant"])
# ha.set_scene("reading")
def test_should_notify_vehicle():
    assert type(ha.should_notify_vehicle()) == bool
    ha.set_notify_vehicle(False)
    assert ha.should_notify_vehicle() == False
    ha.set_notify_vehicle(True)
    assert ha.should_notify_vehicle() == True


def test_should_notify_person():
    assert type(ha.should_notify_person()) == bool
    ha.suppress_notify_person()
    assert ha.should_notify_person() == False


def test_door_contacts():
    door_left = ha.get_door_left()
    door_right = ha.get_door_right()


def test_mode():
    assert ha.mode() in ["home", "night"]


# print("open_garage_door={}".format(st.open_garage_door()))
# ha.deer_alert('tree line')
# print(ha.echo_speaks("hello world"))
