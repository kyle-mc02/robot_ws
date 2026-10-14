#include "DualVNH5019MotorShield.h"

// Initialize the shield
DualVNH5019MotorShield md;

void setup() {
  // 115200 baud to match the Python ROS 2 hardware bridge
  Serial.begin(115200);
  md.init();
}

void loop() {
  // Wait until a full 5-byte packet (Header + Left + Right) is in the buffer
  if (Serial.available() >= 5) {
    
    // Check for our synchronization byte (0xAA)
    if (Serial.read() == 0xAA) {
      uint8_t buffer[4];
      
      // Read the 4 bytes containing the two 16-bit integers
      Serial.readBytes(buffer, 4);
      
      // Reconstruct the integers using bitwise logic (Little-endian)
      int16_t left_speed = buffer[0] | (buffer[1] << 8);
      int16_t right_speed = buffer[2] | (buffer[3] << 8);
      
      // M1 = Left Side (M1A / M1B)
      // M2 = Right Side (M2A / M2B)
      // If a side spins backward when pushing the GUI slider up, swap the 
      // physical wires in that M's terminal block or add a negative sign here.
      md.setM1Speed(left_speed);
      md.setM2Speed(right_speed);
    }
  }
}
