"""
BCI IoT State Testing Script
==============================
Simulates BCI motor imagery classification outputs (left hand, right hand,
neutral/steady-state) at random and sends the corresponding device control
commands to the IoT server.

NO model or NPG Lite hardware is used — states are generated purely randomly.
This is for validating the IoT device response pipeline before connecting
real EEG hardware.

State → IoT Device Mapping
---------------------------
  LEFT  hand  →  device1 ON  +  device2 OFF
  RIGHT hand  →  device2 ON  +  device1 OFF
  NEUTRAL     →  all     OFF

Usage
-----
  # Using a full URL (ngrok, https, etc.):
  python test_iot_bci_states.py --url https://0700-103-160-195-164.ngrok-free.app

  # Using local IP/port:
  python test_iot_bci_states.py --host 192.168.1.10 --port 5000

  # Run a fixed number of trials then stop:
  python test_iot_bci_states.py --url https://xxxx.ngrok-free.app --trials 20

  # Faster trials (1 second each):
  python test_iot_bci_states.py --url https://xxxx.ngrok-free.app --interval 1.0
"""

import argparse
import random
import time
import sys
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library not found.")
    print("Install it with:  pip install requests")
    sys.exit(1)

# ─────────────────────────────────────────────
#  CONFIGURATION  ← edit these defaults if needed
# ─────────────────────────────────────────────
DEFAULT_URL      = "https://0bac-103-160-195-164.ngrok-free.app"  # ngrok tunnel
DEFAULT_HOST     = None       # used only when --url is not set
DEFAULT_PORT     = 5000       # used only when --url is not set
DEFAULT_INTERVAL = 3.0      # seconds between each generated state
DEFAULT_TRIALS   = None      # None = run forever until Ctrl+C

# Class probabilities (must sum to 1.0)
# Adjust to reflect realistic BCI distribution
CLASS_WEIGHTS = {
    "LEFT":    0.35,    # 35% left hand
    "RIGHT":   0.35,    # 35% right hand
    "NEUTRAL": 0.30,    # 30% neutral / steady state
}

# ─────────────────────────────────────────────
#  DEVICE COMMAND ROUTES
#  These must match exactly what your IoT server expects.
# ─────────────────────────────────────────────
DEVICE_ROUTES = {
    "device1_on":  "/device1/on",
    "device1_off": "/device1/off",
    "device2_on":  "/device2/on",
    "device2_off": "/device2/off",
    "all_on":      "/all/on",
    "all_off":     "/all/off",
}

# Console colour codes (work on Windows 10+ and Unix)
COLOUR = {
    "LEFT":    "\033[94m",   # blue
    "RIGHT":   "\033[92m",   # green
    "NEUTRAL": "\033[93m",   # yellow
    "ERROR":   "\033[91m",   # red
    "RESET":   "\033[0m",
    "BOLD":    "\033[1m",
    "DIM":     "\033[2m",
}


class IoTClient:
    """Thin HTTP client for the IoT device control server."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        # Strip trailing slash so route concatenation is always clean
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.stats    = {"sent": 0, "ok": 0, "failed": 0}

    def send(self, route: str) -> bool:
        """
        Send a GET request to the given route.
        Returns True if the server responded with 2xx.
        """
        url = self.base_url + route
        try:
            resp = requests.get(url, timeout=self.timeout)
            self.stats["sent"] += 1
            if resp.ok:
                self.stats["ok"] += 1
                return True
            else:
                self.stats["failed"] += 1
                print(f"  {COLOUR['ERROR']}Server returned {resp.status_code} for {url}{COLOUR['RESET']}")
                return False
        except requests.exceptions.ConnectionError:
            self.stats["failed"] += 1
            print(f"  {COLOUR['ERROR']}Cannot connect to {url}  — is the IoT server running?{COLOUR['RESET']}")
            return False
        except requests.exceptions.Timeout:
            self.stats["failed"] += 1
            print(f"  {COLOUR['ERROR']}Request to {url} timed out{COLOUR['RESET']}")
            return False

    def ping(self) -> bool:
        """Check if the server is reachable at all."""
        for probe in ["/", "/status", "/api/health"]:
            try:
                requests.get(self.base_url + probe, timeout=2.0)
                return True
            except Exception:
                continue
        return False


def pick_state() -> str:
    """Randomly pick a BCI state according to CLASS_WEIGHTS."""
    states  = list(CLASS_WEIGHTS.keys())
    weights = list(CLASS_WEIGHTS.values())
    return random.choices(states, weights=weights, k=1)[0]


def execute_state(client: IoTClient, state: str) -> None:
    """
    Send the device commands that correspond to a given BCI state.

    LEFT    → device1/on  + device2/off
    RIGHT   → device2/on  + device1/off
    NEUTRAL → all/off
    """
    colour = COLOUR[state]
    ts     = datetime.now().strftime("%H:%M:%S")

    if state == "LEFT":
        label = "LEFT  HAND"
        cmds  = [DEVICE_ROUTES["device1_on"], DEVICE_ROUTES["device2_off"]]
        info  = "→ device1 ON  | device2 OFF"
    elif state == "RIGHT":
        label = "RIGHT HAND"
        cmds  = [DEVICE_ROUTES["device2_on"], DEVICE_ROUTES["device1_off"]]
        info  = "→ device1 OFF | device2 ON"
    else:  # NEUTRAL
        label = "NEUTRAL    "
        cmds  = [DEVICE_ROUTES["all_off"]]
        info  = "→ all OFF"

    print(f"\n[{ts}] {colour}{COLOUR['BOLD']}{label}{COLOUR['RESET']}  {COLOUR['DIM']}{info}{COLOUR['RESET']}")

    for route in cmds:
        ok = client.send(route)
        status = f"{COLOUR['RIGHT']}OK{COLOUR['RESET']}" if ok else f"{COLOUR['ERROR']}FAIL{COLOUR['RESET']}"
        print(f"         {status}  {client.base_url}{route}")


def print_summary(client: IoTClient, trial_log: list) -> None:
    """Print a final summary of the test session."""
    counts = {"LEFT": 0, "RIGHT": 0, "NEUTRAL": 0}
    for s in trial_log:
        counts[s] += 1
    total = len(trial_log)

    print(f"\n{'─'*50}")
    print(f"  {COLOUR['BOLD']}Session Summary{COLOUR['RESET']}")
    print(f"{'─'*50}")
    print(f"  Total trials  : {total}")
    print(f"  LEFT  hand    : {counts['LEFT']:3d}  ({100*counts['LEFT']/max(1,total):.0f}%)")
    print(f"  RIGHT hand    : {counts['RIGHT']:3d}  ({100*counts['RIGHT']/max(1,total):.0f}%)")
    print(f"  NEUTRAL       : {counts['NEUTRAL']:3d}  ({100*counts['NEUTRAL']/max(1,total):.0f}%)")
    print()
    print(f"  HTTP requests : {client.stats['sent']}")
    print(f"  Succeeded     : {COLOUR['RIGHT']}{client.stats['ok']}{COLOUR['RESET']}")
    print(f"  Failed        : {COLOUR['ERROR'] if client.stats['failed'] else ''}{client.stats['failed']}{COLOUR['RESET']}")
    print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="BCI IoT state tester — sends random LEFT/RIGHT/NEUTRAL commands to IoT server"
    )
    parser.add_argument("--url",      default=DEFAULT_URL,      help=f"Full base URL of IoT server, e.g. https://xxxx.ngrok-free.app (overrides --host/--port)")
    parser.add_argument("--host",     default=DEFAULT_HOST,     help="IoT server hostname/IP (ignored when --url is set)")
    parser.add_argument("--port",     default=DEFAULT_PORT,     type=int, help=f"IoT server port (ignored when --url is set, default: {DEFAULT_PORT})")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, type=float, help=f"Seconds between trials (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--trials",   default=DEFAULT_TRIALS,   type=int,   help="Number of trials then stop (default: run forever)")
    args = parser.parse_args()

    # Resolve base URL: --url takes precedence over --host/--port
    if args.url:
        base_url = args.url.rstrip("/")
    elif args.host:
        base_url = f"http://{args.host}:{args.port}"
    else:
        print("ERROR: provide either --url or --host")
        sys.exit(1)

    client = IoTClient(base_url=base_url)

    print(f"\n{COLOUR['BOLD']}BCI IoT State Tester{COLOUR['RESET']}")
    print(f"{'─'*50}")
    print(f"  Target server  : {base_url}")
    print(f"  Trial interval : {args.interval}s")
    print(f"  Max trials     : {args.trials if args.trials else 'unlimited (Ctrl+C to stop)'}")
    print(f"  State weights  : LEFT={CLASS_WEIGHTS['LEFT']:.0%}  RIGHT={CLASS_WEIGHTS['RIGHT']:.0%}  NEUTRAL={CLASS_WEIGHTS['NEUTRAL']:.0%}")
    print(f"{'─'*50}")

    # Connectivity check
    print(f"\nChecking server connectivity...")
    if client.ping():
        print(f"  {COLOUR['RIGHT']}[OK]  Server reachable{COLOUR['RESET']}")
    else:
        print(f"  {COLOUR['ERROR']}[!!] Server not reachable at {base_url}{COLOUR['RESET']}")
        print(f"     Start your IoT server first, then re-run this script.")
        print(f"     (Continuing anyway — requests will show individual errors)\n")

    print(f"\nGenerating states  (Ctrl+C to stop)...\n")

    trial_log = []
    trial_num = 0

    try:
        while True:
            trial_num += 1

            # Stop after fixed number of trials if specified
            if args.trials is not None and trial_num > args.trials:
                break

            state = pick_state()
            trial_log.append(state)

            header = f"{COLOUR['DIM']}Trial {trial_num}" + (f"/{args.trials}" if args.trials else "") + f"{COLOUR['RESET']}"
            print(header, end="")

            execute_state(client, state)

            # Wait before next trial
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n\n  Interrupted by user.")

    finally:
        print_summary(client, trial_log)


if __name__ == "__main__":
    main()
