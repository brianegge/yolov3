#!/usr/bin/env python3
"""CodeProject AI Server ALPR integration."""
import json
import logging
import sys

import requests

logger = logging.getLogger(__name__)

# CodeProject AI Server endpoint
CODEPROJECT_URL = "http://localhost:32168/v1/image/alpr"


def enrich(image_bytes, save_json=None):
    """
    Send image to CodeProject AI ALPR and extract license plate info.

    Args:
        image_bytes: Raw image bytes
        save_json: Optional path to save raw API response

    Returns:
        dict with keys: message, plates, count
    """
    try:
        response = requests.post(
            CODEPROJECT_URL,
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"CodeProject ALPR request failed: {e}")
        return {"message": "", "plates": [], "count": 0}

    if save_json:
        with open(save_json, "w") as f:
            json.dump(result, f, indent=4)

    plates = []
    if result.get("success") and "predictions" in result:
        for prediction in result["predictions"]:
            plate = prediction.get("plate", "")
            if plate:
                plates.append(plate)

    message = ""
    if plates:
        message = f"Vehicle with plate {', '.join(plates)}"

    return {
        "message": message,
        "plates": plates,
        "count": len(result.get("predictions", [])),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = sys.argv[1]
    else:
        f = "tests/175627-garage-r-person_dog_vehicle.jpg"
    print(enrich(open(f, "rb").read()))
