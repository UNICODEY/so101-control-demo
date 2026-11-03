"""Small local API client, using only the Python standard library."""
import argparse
import json
import sys
import urllib.error
import urllib.request
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status",):
        sub.add_parser(name)
    p = sub.add_parser("pick")
    p.add_argument("object_id", choices=["red_cube", "blue_cube", "yellow_cube"])
    p.add_argument("target_id", choices=["left_box", "right_box"])
    p.add_argument("--id", default=None)
    p = sub.add_parser("json")
    p.add_argument("file", help="UTF-8 JSON command file")
    args = parser.parse_args()
    path, payload = "/status", None
    if args.command in ("pick", "json"):
        path = "/commands"
        if args.command == "json":
            with open(args.file, encoding="utf-8-sig") as f:
                payload = json.load(f)
        else:
            payload = dict(id=args.id or "cmd-" + uuid4().hex[:8], action="pick_place", object_id=args.object_id, target_id=args.target_id)
    request = urllib.request.Request(f"http://127.0.0.1:{args.port}" + path,
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.load(response)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except urllib.error.HTTPError as exc:
        print(exc.read().decode())
        return 1
    except urllib.error.URLError as exc:
        print(f"Cannot reach simulation. Start run_demo.py first. {exc.reason}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
