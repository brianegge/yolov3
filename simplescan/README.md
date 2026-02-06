# Dog Notification Triggers

The system detects dogs on camera and classifies them as either `dog` (on property) or `dog_road` (near/on the road) to determine whether to send a notification.

## Road Classification

When a dog is detected, `detect.py` reclassifies it as `dog_road` based on the camera:

| Camera | Rule |
|--------|------|
| **driveway** | Dog above a diagonal road line (`road_y = 0.5 * (1 - x) + 0.2 * x`) is `dog_road` |
| **peach tree** | Dog above an angled road line (piecewise formula based on x position) is `dog_road` |
| **mailbox** | All dogs are `dog_road` (camera views public area) |
| **all others** | Dogs remain classified as `dog` |

## Mode

The system operates in one of three modes, determined by `homeassistant.py:145-151`:

1. **night** — `input_boolean.night_mode` is on
2. **home** — anyone in `group.egge` has presence state `"home"`
3. **away** — otherwise

Mode selects which priority config section is used (`[priority-night]`, `[priority-home]`, or `[priority-away]`), allowing notification thresholds to vary based on whether someone is home.

## Notification Rules

A notification is sent (priority = 1) when any of the following conditions are met:

### Dog on the road without a person (`notify.py:131-137`)
- Detection is `dog_road`
- No `person_road` detected nearby
- If a person *is* on the road, priority drops to -3 (suppressed as "person walking dog")

### Dog in the garage (`notify.py:179-186`)
- Detection is `dog` on the **garage** camera
- Confidence > 90%
- No person detected

### Dog near a package (`notify.py:200-202`)
- Both a `dog` and a `package` are detected in the same frame
- Overrides any lower priority to 1
