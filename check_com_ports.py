import serial.tools.list_ports

print("Available COM ports:")
ports = serial.tools.list_ports.comports()
for i, port in enumerate(ports):
    print(f"{i+1}. {port.device} - {port.description}")
