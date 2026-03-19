# Toggle API Call Scheme Implementation

## Overview
The API integration has been refactored to use a simple **toggle scheme with GET requests** to separate endpoints for each device's on/off states.

## Architecture

### Endpoint Format
```
Base URL: http://your-host:port
Endpoints:
  - {base}/device1/on    → Turn device1 ON
  - {base}/device1/off   → Turn device1 OFF
  - {base}/device2/on    → Turn device2 ON
  - {base}/device2/off   → Turn device2 OFF
```

### Toggle Logic
```
Command Mapping:
  LEFT_HAND  → device1 toggle (on/off/on/off...)
  RIGHT_HAND → device2 toggle (on/off/on/off...)
```

**First encounter of command**: Device turns ON
**Second encounter of command**: Device turns OFF
**Third encounter of command**: Device turns ON again
...and so on

## Implementation Details

### State Tracking
The system maintains `device_toggle_state` for each command:
- `device_toggle_state['LEFT_HAND']`: Tracks device1 state (False=OFF, True=ON)
- `device_toggle_state['RIGHT_HAND']`: Tracks device2 state (False=OFF, True=ON)

### Key Methods

#### `_build_api_url(device: str, state: str) -> Optional[str]`
Constructs the full GET request URL:
```python
# Input: device='device1', state='on'
# Output: 'http://127.0.0.1:8000/device1/on'
```

#### `_get_request(url: str, command: str, confidence: float)`
Sends simple GET requests (no POST body required):
```python
# Makes HTTP GET call to constructed URL
# Logs success/failure with device state and confidence
```

#### `_toggle_device_for_command(command: str, confidence: float)`
Handles toggle logic:
```python
# LEFT_HAND:  toggle device1 state, call GET /device1/on or /device1/off
# RIGHT_HAND: toggle device2 state, call GET /device2/on or /device2/off
```

## Usage

### Installation
No additional dependencies required - uses standard `urllib.request` for GET requests.

### Command Line Arguments

```bash
# Start with API enabled
python npg_realtime_bci.py --api-base-url http://127.0.0.1:8000

# With simulator mode
python npg_realtime_bci.py --simulate --api-base-url http://192.168.1.100:5000

# With custom timeout and thresholds
python npg_realtime_bci.py \
    --api-base-url http://127.0.0.1:8000 \
    --api-timeout 5.0 \
    --api-conf-threshold 0.7 \
    --api-neutral-hold 1.0 \
    --api-cooldown 3.0
```

### Example Flow

```
User Input Sequence:
1. LEFT_HAND detected → GET http://base/device1/on   (device1 turns ON)
2. RIGHT_HAND detected → GET http://base/device2/on  (device2 turns ON)
3. LEFT_HAND detected again → GET http://base/device1/off (device1 turns OFF)
4. RIGHT_HAND detected again → GET http://base/device2/off (device2 turns OFF)
5. LEFT_HAND detected again → GET http://base/device1/on   (device1 turns ON again)
```

## Logging Output

When commands are processed, you'll see logs like:

```
🌐 GET REQUEST OK | device1/ON | status=200 | conf=82.5% | http://127.0.0.1:8000/device1/on
💡 TOGGLE | LEFT_HAND -> device1/on

🌐 GET REQUEST OK | device2/OFF | status=200 | conf=78.1% | http://127.0.0.1:8000/device2/off
🌀 TOGGLE | RIGHT_HAND -> device2/off
```

## Safety Features

### Anti-Bounce Protection
- **Edge Trigger**: Commands ignored if repeated without state change
- **Cooldown**: Per-command throttling (default: 2.5s)
- **Neutral Hold**: Commands need 0.8s of NEUTRAL state before re-arming
- **Confidence Threshold**: Minimum certainty required (default: 0.75)

### Example of Anti-Bounce
```
Time 0s:    LEFT_HAND (high conf) → device1/on
Time 0.5s:  LEFT_HAND (high conf) → IGNORED (no edge, within cooldown)
Time 1s:    LEFT_HAND (high conf) → IGNORED (no edge, within cooldown)
Time 2.6s:  LEFT_HAND (high conf) → device1/off (edge detected, cooldown expired)
```

## Benefits of GET Requests

✅ **Simpler**: No JSON body serialization needed
✅ **Stateless**: Easier for IoT devices with limited resources
✅ **Cacheable**: Can be cached by intermediaries (if needed)
✅ **Standard**: Works with any simple REST API
✅ **Lightweight**: Minimal network overhead

## Migration from POST to GET

### What Changed
| Aspect | Old (POST) | New (GET) |
|--------|-----------|----------|
| HTTP Method | POST | GET |
| Endpoints | `/light`, `/fan` | `/device1/on`, `/device1/off`, `/device2/on`, `/device2/off` |
| Request Body | JSON payload | None |
| Mapping | Light=LEFT, Fan=RIGHT | device1=LEFT, device2=RIGHT |
| Device Names | `api_light_path`, `api_fan_path` | Hardcoded as `device1`, `device2` |

### Removed CLI Arguments
- `--api-light-path` (was `/light`)
- `--api-fan-path` (was `/fan`)

These are no longer needed since device names are fixed as `device1` and `device2`.

## Testing

### Minimal Server
To test locally, run a simple HTTP server in Python:

```bash
# terminal 1: Start mock server
python -m http.server 8000

# terminal 2: Run BCI with API
python npg_realtime_bci.py --simulate --api-base-url http://127.0.0.1:8000
```

### Server Logs
```
# You'll see GET requests like:
GET /device1/on HTTP/1.1
GET /device2/off HTTP/1.1
```

## Future Enhancements

Possible improvements:
- Add support for device naming via CLI (--device1-name, --device2-name)
- Support for more than 2 devices
- Device-specific state caching
- Batch commands in a single request
- WebSocket support for real-time feedback
