"""
Test raw serial connection to NPG Lite
to see what data format it's actually sending
"""

import serial
import time

port = 'COM9'
baudrate = 230400

print(f"Opening {port} at {baudrate} baud...")
print("Make sure NPG Lite is powered on and connected.\n")

try:
    ser = serial.Serial(port, baudrate, timeout=1)
    time.sleep(2)  # Wait for device
    
    # Flush buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    print(f"✅ Connected to {port}")
    print("Reading data for 10 seconds...\n")
    print("="*70)
    
    start = time.time()
    line_count = 0
    
    while time.time() - start < 10:
        if ser.in_waiting > 0:
            # Try reading as line
            line = ser.readline()
            line_count += 1
            
            print(f"Line {line_count}: {line}")
            
            # Try to decode
            try:
                decoded = line.decode('utf-8').strip()
                print(f"  Decoded: '{decoded}'")
            except:
                print(f"  (Could not decode as UTF-8)")
            
            print()
            
            if line_count >= 20:  # First 20 lines
                break
        else:
            time.sleep(0.01)
    
    print("="*70)
    
    if line_count == 0:
        print("\n❌ No data received!")
        print("   Check:")
        print("   - Is NPG Lite powered on? (check LED)")
        print("   - Is it connected to COM9?")
        print("   - Does it need initialization command?")
        print("   - Try a different baud rate: 115200, 230400, 460800, 921600")
    else:
        print(f"\n✅ Received {line_count} lines of data")
    
    ser.close()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n   Make sure:")
    print("   - NPG Lite is connected to COM9")
    print("   - No other program is using COM9")
    print("   - CH340 drivers are installed")
