# Quick Start: Toggle API Integration

## Setup

1. **Ensure your API server is running** with these endpoints:
   ```
   GET /device1/on
   GET /device1/off
   GET /device2/on
   GET /device2/off
   ```

2. **Start the BCI system with API enabled:**
   ```bash
   cd CODE
   python npg_realtime_bci.py --simulate --api-base-url http://127.0.0.1:8000
   ```

## How It Works

| Command | Action | Endpoint Called |
|---------|--------|-----------------|
| LEFT_HAND (1st) | Turn ON | `GET /device1/on` |
| LEFT_HAND (2nd) | Turn OFF | `GET /device1/off` |
| LEFT_HAND (3rd) | Turn ON again | `GET /device1/on` |
| RIGHT_HAND (1st) | Turn ON | `GET /device2/on` |
| RIGHT_HAND (2nd) | Turn OFF | `GET /device2/off` |
| RIGHT_HAND (3rd) | Turn ON again | `GET /device2/on` |

## Expected Log Output

```
🌐 GET REQUEST OK | device1/ON | status=200 | conf=82.5% | http://127.0.0.1:8000/device1/on
💡 TOGGLE | LEFT_HAND -> device1/on

🌐 GET REQUEST OK | device2/OFF | status=200 | conf=78.1% | http://127.0.0.1:8000/device2/off
🌀 TOGGLE | RIGHT_HAND -> device2/off
```

## Testing with Mock Server

```bash
# Terminal 1: Start Python HTTP server (responds to any GET)
python -m http.server 8000

# Terminal 2: Run BCI with simulator
python npg_realtime_bci.py --simulate --api-base-url http://127.0.0.1:8000
```

## Configuration Parameters

```bash
python npg_realtime_bci.py \
    --api-base-url http://192.168.1.100:8000 \       # Base URL
    --api-timeout 3.0 \                                # Connection timeout
    --api-conf-threshold 0.75 \                        # Min confidence to send API calls
    --api-neutral-hold 0.8 \                           # Neutral hold to re-arm
    --api-cooldown 2.5                                 # Cooldown between same command
```

## Troubleshooting

**Issue**: API calls not being sent
- Check `--api-conf-threshold` - might be too high (default: 0.75)
- Check `--api-cooldown` - you need to wait between repeated commands
- Check `--api-neutral-hold` - send NEUTRAL command after LEFT/RIGHT to re-arm

**Issue**: Connection errors
- Verify API server is running on the specified URL
- Check firewall/network connectivity
- Try with `--api-timeout 5.0` for slower networks

**Issue**: Unexpected device state
- System state restarts when you run again (not persistent)
- To test persistence, keep the same instance running
