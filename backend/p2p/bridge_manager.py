"""
P2P Serial Bridge Manager -- Hardware Integration

Sends approved allocation results to the ESP32 hub via Serial.
ESP32 then broadcasts via ESP-NOW to ESP8266 village nodes.

Usage:
    bridge = SerialBridge(port='COM3', baudrate=115200)
    bridge.connect()
    bridge.send_route_update(village='Panauti', vehicle='Heli-01', eta_min=5)
    bridge.disconnect()
"""
from __future__ import annotations

import json
import time
from typing import Dict, Optional

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("WARNING: pyserial not installed. Install: pip install pyserial")


# ================================================================== #
#  SerialBridge                                                        #
# ================================================================== #

class SerialBridge:
    """
    Serial communication bridge to ESP32 hardware hub.
    
    Sends compressed JSON payloads containing route assignments
    from an approved allocation solution to ESP32 for ESP-NOW broadcast.
    
    Args:
        port:     COM port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
        baudrate: Serial baud rate (default 115200)
        timeout:  Read timeout in seconds (default 2.0)
    """
    
    def __init__(
        self,
        port: str = 'COM3',
        baudrate: int = 115200,
        timeout: float = 2.0,
    ) -> None:
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed. Cannot create SerialBridge.")
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection: Optional[serial.Serial] = None
        self.connected = False
    
    # ---------------------------------------------------------------- #
    #  Connection management                                            #
    # ---------------------------------------------------------------- #
    
    def connect(self) -> bool:
        """
        Open serial connection to ESP32.
        
        Returns True on success, False on failure.
        """
        if self.connected:
            print(f"Already connected to {self.port}")
            return True
        
        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            # Wait for ESP32 to initialize
            time.sleep(2.0)
            self.connected = True
            print(f"✓ Connected to ESP32 on {self.port} @ {self.baudrate} baud")
            return True
        
        except serial.SerialException as exc:
            print(f"✗ Failed to connect to {self.port}: {exc}")
            return False
    
    def disconnect(self) -> None:
        """Close serial connection."""
        if self.connection and self.connected:
            self.connection.close()
            self.connected = False
            print(f"✓ Disconnected from {self.port}")
    
    def is_connected(self) -> bool:
        """Check if serial connection is active."""
        return self.connected and self.connection is not None
    
    # ---------------------------------------------------------------- #
    #  Data transmission                                                #
    # ---------------------------------------------------------------- #
    
    def send_payload(self, payload: Dict) -> bool:
        """
        Send JSON payload to ESP32.
        
        Payload is serialized to JSON string and sent with newline terminator.
        
        Args:
            payload: Dictionary to send (will be JSON-encoded)
        
        Returns:
            True on successful send, False on failure
        """
        if not self.is_connected():
            print("✗ Not connected. Call connect() first.")
            return False
        
        try:
            # Serialize to JSON
            json_str = json.dumps(payload, separators=(',', ':'))  # Compact format
            
            # Send with newline terminator
            message = (json_str + '\n').encode('utf-8')
            self.connection.write(message)
            self.connection.flush()
            
            print(f"→ Sent: {json_str}")
            
            # Optional: wait for ESP32 ACK (if implemented)
            # ack = self.connection.readline().decode('utf-8').strip()
            # if ack == "ACK":
            #     print("✓ ESP32 acknowledged")
            
            return True
        
        except Exception as exc:
            print(f"✗ Failed to send payload: {exc}")
            return False
    
    def send_route_update(
        self,
        village: str,
        vehicle: str,
        eta_min: int,
    ) -> bool:
        """
        Send route assignment update to ESP32 for ESP-NOW broadcast.
        
        Creates compressed payload: {"v":"Panauti","h":"Heli-01","t":5}
        
        Args:
            village:  Village name (e.g., "Panauti")
            vehicle:  Assigned vehicle ID (e.g., "Heli-01")
            eta_min:  Estimated time of arrival in minutes
        
        Returns:
            True on successful send, False on failure
        """
        payload = {
            "v": village,
            "h": vehicle,
            "t": eta_min,
        }
        return self.send_payload(payload)
    
    def send_heartbeat(self) -> bool:
        """
        Send heartbeat ping to ESP32.
        
        Payload: {"cmd":"ping"}
        """
        return self.send_payload({"cmd": "ping"})
    
    def send_reset(self) -> bool:
        """
        Send reset command to ESP32.
        
        Payload: {"cmd":"reset"}
        """
        return self.send_payload({"cmd": "reset"})


# ================================================================== #
#  Demo / Testing                                                      #
# ================================================================== #

def demo_bridge():
    """
    Demo: Connect to ESP32 and send sample route updates.
    """
    print("\n" + "=" * 65)
    print("  RAKSHYANET P2P BRIDGE MANAGER - DEMO")
    print("=" * 65)
    
    # Change port to match your system
    # Windows: 'COM3', 'COM4', etc.
    # Linux: '/dev/ttyUSB0', '/dev/ttyACM0', etc.
    # macOS: '/dev/cu.usbserial-XXXX'
    bridge = SerialBridge(port='COM3', baudrate=115200)
    
    if not bridge.connect():
        print("\n✗ Connection failed. Check:")
        print("  1. ESP32 is plugged in via USB")
        print("  2. Correct COM port (use Device Manager on Windows)")
        print("  3. CH340/CP2102 USB-to-Serial drivers installed")
        print("  4. No other program using the port (close Arduino IDE Serial Monitor)")
        return
    
    try:
        # Send heartbeat
        print("\n--- Sending Heartbeat ---")
        bridge.send_heartbeat()
        time.sleep(1)
        
        # Send sample route updates
        print("\n--- Sending Route Updates ---")
        routes = [
            ("Panauti", "Heli-01", 5),
            ("Dhulikhel", "Truck-03", 12),
            ("Banepa", "Heli-02", 8),
        ]
        
        for village, vehicle, eta in routes:
            bridge.send_route_update(village, vehicle, eta)
            time.sleep(0.5)  # Small delay between messages
        
        print("\n✓ All messages sent successfully")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    
    finally:
        bridge.disconnect()
        print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    if not SERIAL_AVAILABLE:
        print("\n✗ pyserial not installed.")
        print("Install: pip install pyserial")
        print("Or add to requirements.txt\n")
    else:
        demo_bridge()
