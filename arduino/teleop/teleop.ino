/*
 * teleop.ino — GELLO-Lite leader-follower teleoperation.
 *
 * Copied from github.com/imjbassi/gello-lite into this project so the
 * recording sketch and policy_follower.ino stay in sync: JOINT_MIN/JOINT_MAX,
 * POT_MIN/POT_MAX, INVERT and SMOOTHING must be IDENTICAL in both files, or a
 * policy trained on recordings will command different angles than the operator
 * demonstrated.
 *
 * Reads four potentiometers (the leader) and drives the follower's hobby
 * servos (MG90S shoulder/elbow; SG90 base/gripper) with a direct 1:1
 * joint-space mapping. No inverse kinematics.
 *
 * Also streams joint angles over serial as CSV, so record_episode.py can
 * record demonstration episodes.
 *
 * Wiring (EEZYbotARM build):
 *   Potentiometers  outer pins -> 5V and GND, wiper -> A0..A3
 *   Servos          signal -> D3/D5/D6/D9, red -> servo supply +, brown -> common GND
 *
 * Serial output format (115200 baud):
 *   <millis>,<base>,<shoulder>,<elbow>,<gripper>
 *
 * Library: Servo (bundled with the Arduino IDE)
 */

#include <Servo.h>

// ---------------------------------------------------------------- config

const uint8_t  NUM_JOINTS   = 4;
const uint8_t  POT_PINS[]   = {A0, A1, A2, A3};
const uint8_t  SERVO_PINS[] = {3, 5, 6, 9};
const char*    JOINT_NAMES[] = {"base", "shoulder", "elbow", "gripper"};

// Measured with limit_finder.ino on the EEZYbotARM build (2026-09-19).
// The linkage restricts the shoulder and elbow far below a servo's travel.
// Keep identical to policy_follower.ino and config/robot.json.
const int JOINT_MIN[] = {  0,  60,  75,  40};
const int JOINT_MAX[] = {180, 130, 120, 180};

// Set to true for any joint whose leader and follower move opposite ways.
// Cheaper than rewiring or remounting a horn.
const bool INVERT[] = {false, false, false, false};

// Raw ADC range actually reachable on each potentiometer.
// Calibrated 2026-09-19: these WH148s reach both rails on all four channels,
// so the full range is correct here rather than a placeholder.
const int POT_MIN[] = {0, 0, 0, 0};
const int POT_MAX[] = {1023, 1023, 1023, 1023};

// Exponential smoothing factor, 0.0-1.0.
//   1.0 = raw, no smoothing (jittery — ADC noise goes straight to the servo)
//   0.2 = heavily smoothed (calm, but laggy)
const float SMOOTHING = 0.35;

const unsigned long LOOP_INTERVAL_MS = 20;   // 50 Hz control loop
const bool STREAM_TELEMETRY = true;

// ---------------------------------------------------------------- state

Servo servos[NUM_JOINTS];
float smoothed[NUM_JOINTS];
unsigned long nextLoop = 0;

// ------------------------------------------------------------------ util

/*
 * One potentiometer reading -> one servo angle.
 *
 * This function IS the teleoperation mechanism. Everything else in this file
 * is plumbing around it.
 */
int potToAngle(uint8_t joint, int raw) {
  raw = constrain(raw, POT_MIN[joint], POT_MAX[joint]);

  int angle = map(raw,
                  POT_MIN[joint], POT_MAX[joint],
                  JOINT_MIN[joint], JOINT_MAX[joint]);

  if (INVERT[joint]) {
    angle = JOINT_MAX[joint] - (angle - JOINT_MIN[joint]);
  }

  return constrain(angle, JOINT_MIN[joint], JOINT_MAX[joint]);
}

// ------------------------------------------------------------------ setup

void setup() {
  Serial.begin(115200);

  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    // Seed the smoothing filter with the leader's actual current position.
    // Without this, every joint snaps from 0 to wherever the leader is on the
    // first loop — a violent jolt that can knock the arm over or strip a gear.
    smoothed[i] = potToAngle(i, analogRead(POT_PINS[i]));
    servos[i].write((int)smoothed[i]);   // set pulse before attach: no snap to 90
    servos[i].attach(SERVO_PINS[i]);
  }

  delay(500);   // let the servos physically reach the start pose

  if (STREAM_TELEMETRY) {
    Serial.println(F("millis,base,shoulder,elbow,gripper"));
  }
}

// ------------------------------------------------------------------- loop

void loop() {
  unsigned long now = millis();
  if (now < nextLoop) return;
  nextLoop = now + LOOP_INTERVAL_MS;

  int angles[NUM_JOINTS];

  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    int target = potToAngle(i, analogRead(POT_PINS[i]));

    // Exponential moving average. The ADC's last couple of bits are noisy
    // even when nothing is moving, and writing that noise straight to a servo
    // makes it buzz audibly while holding still.
    smoothed[i] = SMOOTHING * target + (1.0 - SMOOTHING) * smoothed[i];

    angles[i] = (int)(smoothed[i] + 0.5);
    servos[i].write(angles[i]);
  }

  if (STREAM_TELEMETRY) {
    Serial.print(now);
    for (uint8_t i = 0; i < NUM_JOINTS; i++) {
      Serial.print(',');
      Serial.print(angles[i]);
    }
    Serial.println();
  }
}
