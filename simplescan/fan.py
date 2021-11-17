import aiohttp
import pysmartthings
import asyncio

token = "2fc029f1-a444-4464-99cc-0b6613934922"


async def main():
    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(session, token)
        locations = await api.locations()
        print(len(locations))

        location = locations[0]
        print(location.name)
        print(location.location_id)

        devices = await api.devices()
        print(len(devices))
        for device in devices:
            if device.device_id == "a7460f7e-52d0-45c6-bd9b-b0b54cba2983":
                print(device.name)
                print(device.label)
                print(device.capabilities)
                await device.status.refresh()
                print(device.status.values)
                print(device.status.switch)
                print(device.status.level)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
