"""
NPG Lite Connection Diagnostic Tool
Tests serial ports to find NPG Lite and verify data format
"""

import serial
import serial.tools.list_ports
import time
import sys

def list_ports():
    """List all available COM ports."""
    print("\n📋 Available COM Ports:")
    print("=" * 60)
    
    ports = list(serial.tools.list_ports.comports())
    
    if not ports:
        print("❌ No COM ports found!")
        return []
    
    for p in ports:
        status = "🔌 USB" if "USB" in p.description else "📡 BT"
        print(f"  {status} {p.device}: {p.description}")
        if p.vid and p.pid:
            print(f"       VID:PID = {hex(p.vid)}:{hex(p.pid)}")
    
    return [p.device for p in ports]

def test_port(port, baudrate=230400, duration=3):
    """
    Test if NPG Lite is on this port.
    
    Args:
        port: COM port to test
        baudrate: Baud rate to try
        duration: Seconds to listen for data
        
    Returns:
        True if data received, False otherwise
    """
    print(f"\n🔍 Testing {port} at {baudrate} baud...")
    print("-" * 60)
    
    try:
        # Try to open port
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1
        )
        
        print(f"✅ Port opened successfully")
        
        # Clear buffers
        ser.reset_input_buffer()
        time.sleep(0.5)
        
        # Listen for data
        print(f"📡 Listening for {duration} seconds...")
        
        lines_received = 0
        start_time = time.time()
        sample_lines = []
        
        while (time.time() - start_time) < duration:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        lines_received += 1
                        
                        # Store first 5 lines
                        if lines_received <= 5:
                            sample_lines.append(line)
                            print(f"  📥 Line {lines_received}: {line[:80]}...")  # Truncate long lines
                
                except Exception as e:
                    if lines_received < 3:  # Only show first few errors
                        print(f"  ⚠️  Parse error: {e}")
        
        ser.close()
        
        if lines_received > 0:
            print(f"\n✅ SUCCESS: Received {lines_received} lines in {duration}s")
            print(f"   Rate: ~{lines_received/duration:.1f} lines/sec")
            
            # Analyze format
            if sample_lines:
                print(f"\n📊 Data Format Analysis:")
                first_line = sample_lines[0]
                
                # Try CSV
                try:
                    values = [float(v) for v in first_line.split(',')]
                    print(f"  ✅ CSV format detected: {len(values)} columns")
                    print(f"     Example values: {values[:5]}")
                except:
                    print(f"  ⚠️  Not standard CSV format")
                    
                    # Try space-separated
                    try:
                        values = [float(v) for v in first_line.split()]
                        print(f"  ✅ Space-separated format: {len(values)} columns")
                    except:
                        print(f"  ⚠️  Unknown format")
                        print(f"     Raw: {first_line[:100]}")
            
            print(f"\n   This is likely your NPG Lite!")
            return True
        else:
            print(f"\n❌ No data received (timeout after {duration}s)")
            print(f"   Device may be on this port but not streaming")
            return False
    
    except serial.SerialException as e:
        print(f"❌ Cannot open port: {e}")
        if "PermissionError" in str(e) or "Access is denied" in str(e):
            print(f"   ⚠️  Port is in use by another program!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run diagnostic."""
    print("=" * 60)
    print("🔧 NPG Lite Connection Diagnostic Tool")
    print("=" * 60)
    
    # List available ports
    ports = list_ports()
    
    if not ports:
        print("\n❌ No COM ports found. Check USB connection.")
        return
    
    # Focus on USB Serial Device (COM9)
    usb_ports = [p for p in ports if p == 'COM9']
    
    if not usb_ports:
        print("\n⚠️  COM9 (USB Serial Device) not in port list")
        print("Testing all ports...")
        test_ports = ports
    else:
        print(f"\n🎯 Found USB Serial Device: COM9")
        print("Testing COM9 first, then others if needed...")
        test_ports = usb_ports + [p for p in ports if p != 'COM9']
    
    print(f"\n📡 Testing baud rates: 230400, 115200, 9600")
    
    # Test each port with common baud rates
    baud_rates = [230400, 115200, 9600]
    
    for port in test_ports:
        for baud in baud_rates:
            if test_port(port, baud, duration=3):
                print("\n" + "=" * 60)
                print(f"✅ NPG Lite found on {port} at {baud} baud")
                print("=" * 60)
                print(f"\nUse this command:")
                print(f"  python npg_realtime_bci.py --direct --port {port} --baudrate {baud} --confidence 0.6")
                return
            
            time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print("❌ NPG Lite not detected - No data received")
    print("=" * 60)
    print("\nTroubleshooting:")
    print("  1. ⚡ Check NPG Lite is powered on (LED should be lit)")
    print("  2. 🔄 Unplug and replug USB cable")
    print("  3. 🔧 Check Windows Device Manager for CH340/CH341 driver")
    print("  4. ❌ Close any other programs using the serial port")
    print("  5. 👤 Try running PowerShell as Administrator")
    print("  6. 🎮 Press any buttons on NPG Lite to wake it up")
    print("  7. 📱 Check NPG Lite user manual for startup procedure")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
