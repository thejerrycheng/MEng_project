#include <unistd.h>
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"


int main() {

  SerialPort  serial("/dev/ttyUSB0");
  MotorCmd    cmd;
  MotorData   data;

  while(true) 
  {
    cmd.motorType = MotorType::GO_M8010_6;
    data.motorType = MotorType::GO_M8010_6;
    cmd.mode = queryMotorMode(MotorType::GO_M8010_6,MotorMode::FOC);
    cmd.id   = 0;
    cmd.kp   = 0.0;
    cmd.kd   = 0.05;
    cmd.q    = 0.0; //0.5*queryGearRatio(MotorType::GO_M8010_6);
    cmd.dq   = -6.28*queryGearRatio(MotorType::GO_M8010_6);
    cmd.tau  = 0.0;
    serial.sendRecv(&cmd,&data);

    std::cout <<  std::endl;
    std::cout <<  "motor.q: "    << data.q    <<  std::endl;
    std::cout <<  "motor.temp: "   << data.temp   <<  std::endl;
    std::cout <<  "motor.W: "      << data.dq      <<  std::endl;
    std::cout <<  "motor.merror: " << data.merror <<  std::endl;
    std::cout <<  std::endl;

    usleep(200);
  }

}


// #include <unistd.h>
// #include <iostream>
// #include <cmath> // For sine function
// #include "serialPort/SerialPort.h"
// #include "unitreeMotor/unitreeMotor.h"

// int main() {

//   SerialPort  serial("/dev/ttyUSB0");
//   MotorCmd    cmd;
//   MotorData   data;

//   const double amplitude = M_PI / 2.0; // 90 degrees in radians
//   const double frequency = 0.1; // Frequency of the sine wave (Hz)
//   const double kp = 5; // Position control gain
//   const double kd = 0.02; // Velocity control gain

//   cmd.motorType = MotorType::GO_M8010_6;
//   data.motorType = MotorType::GO_M8010_6;
//   cmd.mode = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
//   cmd.id = 0;

//   double t = 0.0; // Time variable
//   const double dt = 0.005; // Time step in seconds (10ms)

//   while (true) 
//   {
//     // Compute the desired position based on sine function
//     double desired_position = amplitude * std::sin(2 * M_PI * frequency * t);

//     // Set motor command values
//     cmd.kp = kp;
//     cmd.kd = kd;
//     cmd.q = desired_position; // Desired position
//     cmd.dq = 0.0;             // Desired velocity
//     cmd.tau = 0.0;            // Torque feedforward (not used here)

//     // Communicate with the motor
//     serial.sendRecv(&cmd, &data);

//     // Print motor data for debugging
//     std::cout << std::endl;
//     std::cout << "motor.q: "    << data.q    << std::endl;
//     std::cout << "motor.temp: " << data.temp << std::endl;
//     std::cout << "motor.W: "    << data.dq   << std::endl;
//     std::cout << "motor.merror: " << data.merror << std::endl;
//     std::cout << std::endl;

//     // Increment time
//     t += dt;

//     // Sleep for the time step duration
//     usleep(static_cast<useconds_t>(dt * 1e6));
//   }

//   return 0;
// }
