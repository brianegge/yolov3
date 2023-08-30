#!/usr/bin/env python3
import configparser

from homeassistant import HomeAssistant

config = configparser.ConfigParser()
config.read("config.txt")

ha = HomeAssistant(config["homeassistant"])
# ha.set_scene("reading")
def test_should_notify_vehicle():
    orig = ha.should_notify_vehicle()
    assert type(orig) == bool
    ha.set_notify_vehicle(False)
    assert ha.should_notify_vehicle() == False
    ha.set_notify_vehicle(True)
    assert ha.should_notify_vehicle() == True
    ha.set_notify_vehicle(orig)


def test_should_notify_person():
    assert type(ha.should_notify_person()) == bool
    ha.suppress_notify_person()
    assert ha.should_notify_person() == False


def test_door_contacts():
    door_left = ha.get_door_left()
    door_right = ha.get_door_right()


def test_mode():
    assert ha.mode() in ["home", "night"]


def test_dark():
    assert ha.is_dark() in [True, False]


def test_is_before_six():
    # Call the function and print the result
    if ha.is_time_after_midnight_and_before_six():
        print("The current time is after midnight and before 6:00 AM.")
    else:
        print("The current time is not within the specified range.")


# print("open_garage_door={}".format(st.open_garage_door()))
# ha.deer_alert('tree line')
# print(ha.echo_speaks("hello world"))
