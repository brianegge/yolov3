#!/usr/bin/env python3
"""Standalone HTTP server that uploads flagged detection images to Roboflow.

Listens for POST /upload requests from Home Assistant and uploads the
referenced image to the Roboflow REST API for model retraining.

Python 3.6 compatible (runs on Jetson Nano).
"""
import argparse
import base64
import json
import logging
import os
import sys
from configparser import ConfigParser
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
except ImportError:
    pass

logger = logging.getLogger("aicam-review")


class Config(object):
    def __init__(self, config):
        section = config["roboflow"]
        self.api_key = section["api-key"]
        self.project_id = section["project-id"]
        self.delete_after_upload = section.getboolean("delete-after-upload", True)
        self.save_path = config["detector"]["save-path"]
        self.review_dir = os.path.join(self.save_path, "review")


_config = None


class UploadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/upload":
            self._respond(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._respond(400, {"error": "empty body"})
            return

        try:
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (json.JSONDecodeError, ValueError) as e:
            self._respond(400, {"error": "invalid json: %s" % e})
            return

        filename = body.get("file")
        model = body.get("model", "unknown")
        cam = body.get("cam", "unknown")

        if not filename:
            self._respond(400, {"error": "missing 'file' field"})
            return

        # Prevent path traversal
        filename = os.path.basename(filename)
        filepath = os.path.join(_config.review_dir, filename)

        if not os.path.isfile(filepath):
            self._respond(404, {"error": "file not found: %s" % filename})
            return

        try:
            with open(filepath, "rb") as f:
                image_data = f.read()
        except IOError as e:
            self._respond(500, {"error": "failed to read file: %s" % e})
            return

        encoded = base64.b64encode(image_data).decode("utf-8")
        name = os.path.splitext(filename)[0]
        tags = "%s,%s" % (cam.replace(" ", "_"), model)
        url = "https://api.roboflow.com/dataset/%s/upload?api_key=%s&name=%s&split=train&tag=%s" % (
            _config.project_id, _config.api_key, name, tags
        )

        try:
            req = Request(url, data=encoded.encode("utf-8"), method="POST")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            resp = urlopen(req, timeout=30)
            resp_body = resp.read().decode("utf-8")
            logger.info("Uploaded %s to Roboflow: %s", filename, resp_body)
        except (HTTPError, URLError) as e:
            error_body = ""
            if hasattr(e, "read"):
                error_body = e.read().decode("utf-8", errors="replace")
            logger.error("Roboflow upload failed for %s: %s %s", filename, e, error_body)
            self._respond(502, {"error": "roboflow upload failed: %s" % e})
            return

        if _config.delete_after_upload:
            try:
                os.remove(filepath)
                logger.info("Deleted review file %s", filepath)
            except OSError as e:
                logger.warning("Failed to delete %s: %s", filepath, e)

        self._respond(200, {"status": "uploaded", "file": filename, "tags": tags})

    def _respond(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def log_message(self, format, *args):
        logger.info(format, *args)


def main():
    global _config

    parser = argparse.ArgumentParser(description="Roboflow review image upload server")
    parser.add_argument("--port", type=int, default=5050, help="HTTP listen port")
    parser.add_argument("--config", default="config.txt", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = ConfigParser()
    config.read(args.config)

    if "roboflow" not in config:
        logger.error("Missing [roboflow] section in %s", args.config)
        sys.exit(1)
    if "detector" not in config:
        logger.error("Missing [detector] section in %s", args.config)
        sys.exit(1)

    _config = Config(config)
    os.makedirs(_config.review_dir, exist_ok=True)

    server = HTTPServer(("", args.port), UploadHandler)
    logger.info("Listening on port %d, review dir: %s", args.port, _config.review_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    logger.info("Server stopped")


if __name__ == "__main__":
    main()
