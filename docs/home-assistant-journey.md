# My Home Assistant Journey: Stray Dog Detector

One of my favorite automations combines AI-powered object detection with Home Assistant to alert me when a stray dog appears near my house. Here's how I built it.

## The Problem

We live on a street where loose dogs occasionally wander through our yard. With packages on the porch and our own dog Rufus sometimes in the garage, I wanted real-time alerts when an unaccompanied dog showed up -- especially near deliveries.

## Hardware

The detection system runs on an **NVIDIA Jetson Nano** processing feeds from 15 IP cameras. The cameras send motion-triggered snapshots via FTP, which the Jetson picks up and runs through a YOLOv4 model trained to detect people, vehicles, dogs, cats, deer, and packages.

```
IP Cameras (FTP/HTTP) -> Jetson Nano (YOLOv4 inference) -> MQTT / Pushover / Home Assistant
```

## How the Dog Detection Works

The system doesn't just detect dogs -- it classifies _where_ the dog is and _what else_ is in the frame to decide whether to alert.

### Road Classification

Each camera can define a `road_line` -- a piecewise linear boundary separating my property from the public road. When a dog is detected above this line, it's reclassified from `dog` to `dog_road`, indicating it's a stray or roaming animal rather than a dog in my yard.

### Context-Aware Notifications

Not every dog sighting deserves an alert. The system applies several rules:

| Scenario | Priority | Action |
|----------|----------|--------|
| Dog on road **without** a person | **1 (high)** | Push notification + sound |
| Dog on road **with** a person nearby | **-3 (suppressed)** | Logged as "person walking dog" |
| Dog in garage with >90% confidence, no person, and Rufus is inside | **1 (high)** | Push notification |
| Dog near a package (and Rufus is inside) | **1 (high)** | Push notification |

The key insight is the "person walking dog" suppression. A dog on the road with a person is almost certainly being walked on a leash -- no need to alert. But a dog alone on the road is likely a stray.

### Knowing Where Rufus Is

To avoid false alarms from our own dog, the system queries Home Assistant's `sensor.rufus_status` entity. If Rufus is marked as "inside," then any dog detected outside is likely a stray. This simple check dramatically reduces false positives.

```python
def is_dog_inside(self) -> bool:
    response = requests.get(
        f"{self.api}states/sensor.rufus_status", headers=self.headers
    ).json()
    return response.get("state") == "inside"
```

## Home Assistant Integration

The detector integrates with Home Assistant in several ways:

- **Mode awareness** -- The system checks whether we're in `home`, `away`, or `night` mode to adjust notification priorities. Late-night detections between midnight and 6 AM are escalated.
- **Presence detection** -- Alexa announcements ("Dog in driveway") only fire when someone is home, using `group.egge` presence state.
- **MQTT sensors** -- Per-camera object counts are published over MQTT, so I can build Home Assistant dashboards and automations on top of the raw detection data.

## Notifications

Alerts are delivered via [Pushover](https://pushover.net) with a cropped image of the detection. Each notification includes a **"Flag for Review"** link that sends the image to Roboflow for model retraining -- creating a feedback loop that improves detection accuracy over time.

The notification image is cropped and centered on the area of interest, so even on a phone you can immediately see what triggered the alert.

## What I Learned

1. **Suppress the obvious.** The "person walking dog" rule eliminated the majority of false alerts. Think about what _doesn't_ need a notification.
2. **Use context from other systems.** Querying Rufus's indoor/outdoor status from Home Assistant turned a simple dog detector into a _stray_ dog detector.
3. **Edge inference is practical.** The Jetson Nano handles 15 cameras with YOLOv4 ONNX models comfortably. You don't need a cloud GPU for home automation AI.
4. **Build a feedback loop.** The Roboflow review integration means every false positive is an opportunity to improve the model.
