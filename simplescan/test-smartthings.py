import configparser
from smartthings import SmartThings

config = configparser.ConfigParser()
config.read("config.txt")

st = SmartThings(config)
# st.set_st_scene("House Cleaning", config)
print("should_notify_vehicle={}".format(st.should_notify_vehicle()))
print("should_notify_person={}".format(st.should_notify_person()))
# print("open_garage_door={}".format(st.open_garage_door()))
