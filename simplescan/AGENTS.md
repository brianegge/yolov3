# Agent Instructions

## Project Overview

This is an AI camera detection system that monitors IP cameras for object detection (people, vehicles, animals, packages) using ONNX/TensorRT models on an NVIDIA Jetson Nano. It integrates with Home Assistant, MQTT, and Pushover for notifications.

## Deployment

### Deploy to egge-nano

```bash
# 1. Push changes to GitHub
git push

# 2. Pull on egge-nano (SSH key issues require HTTPS fetch)
ssh egge@egge-nano "cd ~/detector/simplescan && git fetch https://github.com/brianegge/yolov3.git master && git reset --hard FETCH_HEAD"

# 3. Restart the service
ssh egge@egge-nano "sudo systemctl restart aicam"

# 4. Verify service is running
ssh egge@egge-nano "sudo systemctl status aicam"
```

### Configuration

Config file location: `egge@egge-nano:~/detector/simplescan/config.txt`

Key sections:
- `[mqtt]` - MQTT broker credentials
- `[homeassistant]` - Home Assistant API URL and token
- `[pushover]` - Pushover notification credentials
- `[codeproject]` - CodeProject AI ALPR URL
- `[cam0]` through `[camN]` - Camera configurations

## Testing

### Run tests locally with Python 3.6.9 container

```bash
# Build test container
podman build -t simplescan-test -f Containerfile .

# Run tests
podman run --rm -v "$(pwd)":/app -w /app simplescan-test pytest -v

# Compile check
podman run --rm -v "$(pwd)":/app -w /app simplescan-test python -m compileall -q .
```

### Start podman machine (if needed)

```bash
podman machine start
```

## Key Files

- `main.py` - Entry point, camera polling loop
- `detect.py` - Object detection logic
- `notify.py` - Notification handling (Pushover, Home Assistant)
- `camera.py` - Camera capture and MQTT publishing
- `homeassistant.py` - Home Assistant API integration
- `codeproject.py` - CodeProject AI ALPR integration
- `Containerfile` - Python 3.6.9 test container
