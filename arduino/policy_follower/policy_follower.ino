/*
 * policy_follower.ino — gello-lite follower driven by a policy over serial.
 *
 * Two modes:
 *   TELEOP  (at boot)  identical to teleop.ino: pots drive servos. Use it to
 *                      put the arm in its start pose between trials.
 *   POLICY             servos follow "C,<base>,<shoulder>,<elbow>,<gripper>"
 *                      commands from the PC, in degrees.
 *
 * PC -> Arduino (newline-terminated):
 *   C,b,s,e,g   set targets and enter POLICY mode
 *   T           return to TELEOP mode
 *   H           hold: stay in POLICY mode, keep current targets
 *
 * Arduino -> PC:
 *   <millis>,<base>,<shoulder>,<elbow>,<gripper>   every loop, the angles actually
 *                                                   written — same format as teleop.ino,
 *                                                   so record_episode.py can record it
 *   #MODE,TELEOP / #MODE,POLICY                     on every mode change
 *   #WATCHDOG                                       no command for WATCHDOG_MS: holding
 *
 * Safety, enforced here regardless of what the PC sends:
 *   - targets clamped to JOINT_MIN / JOINT_MAX
 *   - each joint moves at most MAX_STEP_DEG per 20 ms loop
 *   - if commands stop for WATCHDOG_MS, servos hold their last written angle
 *     (they are not released; the hobby servos would drop under gravity)
 *
 * Copy JOINT_MIN / JOINT_MAX / POT_MIN / POT_MAX / INVERT / SMOOTHING from your
 * calibrated teleop.ino so both sketches agree.
 */

#include <Servo.h>

const uint8_t NUM_JOINTS    = 4;
const uint8_t POT_PINS[]    = {A0, A1, A2, A3};
const uint8_t SERVO_PINS[]  = {3, 5, 6, 9};

// Measured on the EEZYbotARM build with limit_finder.ino (2026-09-19).
// The shoulder and elbow are heavily restricted by the linkage; base and
// gripper reach nearly full travel.
const int  JOINT_MIN[] = {  0,  60,  75,  40};
const int  JOINT_MAX[] = {180, 130, 120, 180};
const bool INVERT[]    = {false, false, false, false};
// Calibrated 2026-09-19: these WH148s reach both rails on all four channels,
// so full range here is the measured result, not a placeholder.
const int  POT_MIN[]   = {0, 0, 0, 0};
const int  POT_MAX[]   = {1023, 1023, 1023, 1023};
const float SMOOTHING  = 0.35;

const unsigned long LOOP_INTERVAL_MS = 20;    // 50 Hz
const float         MAX_STEP_DEG     = 6.0;   // per loop -> 300 deg/s cap
const unsigned long WATCHDOG_MS      = 500;

enum Mode { TELEOP, POLICY };

Servo servos[NUM_JOINTS];
float current[NUM_JOINTS];      // angle last written
float target[NUM_JOINTS];       // policy target
Mode mode = TELEOP;
unsigned long nextLoop = 0;
unsigned long lastCommandMs = 0;
bool watchdogReported = false;

char lineBuf[40];
uint8_t lineLen = 0;

int potToAngle(uint8_t joint, int raw) {
  raw = constrain(raw, POT_MIN[joint], POT_MAX[joint]);
  int angle = map(raw, POT_MIN[joint], POT_MAX[joint], JOINT_MIN[joint], JOINT_MAX[joint]);
  if (INVERT[joint]) angle = JOINT_MAX[joint] - (angle - JOINT_MIN[joint]);
  return constrain(angle, JOINT_MIN[joint], JOINT_MAX[joint]);
}

void setMode(Mode m) {
  if (m == mode) return;
  mode = m;
  if (mode == POLICY) {
    for (uint8_t i = 0; i < NUM_JOINTS; i++) target[i] = current[i];   // no jump on entry
  }
  Serial.println(mode == POLICY ? F("#MODE,POLICY") : F("#MODE,TELEOP"));
}

// Parse "C,b,s,e,g". Returns true on success; ignores malformed lines entirely.
bool parseCommand(char* line) {
  int values[NUM_JOINTS];
  char* token = strtok(line, ",");          // "C"
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    token = strtok(NULL, ",");
    if (token == NULL) return false;
    char* end;
    long v = strtol(token, &end, 10);
    if (end == token) return false;
    values[i] = (int)v;
  }
  if (strtok(NULL, ",") != NULL) return false;
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    target[i] = constrain(values[i], JOINT_MIN[i], JOINT_MAX[i]);
  }
  return true;
}

void handleLine(char* line) {
  if (line[0] == 'C' && line[1] == ',') {
    // setMode first (it resets targets to the current pose on entry); parseCommand
    // only overwrites targets once the whole line is valid.
    setMode(POLICY);
    if (parseCommand(line)) {
      lastCommandMs = millis();
      watchdogReported = false;
    }
  } else if (line[0] == 'T' && line[1] == '\0') {
    setMode(TELEOP);
  } else if (line[0] == 'H' && line[1] == '\0') {
    setMode(POLICY);
    lastCommandMs = millis();
  }
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      if (lineLen > 0) handleLine(lineBuf);
      lineLen = 0;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;                           // overlong line: drop it
    }
  }
}

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    current[i] = potToAngle(i, analogRead(POT_PINS[i]));
    target[i] = current[i];
    servos[i].write((int)current[i]);        // set pulse before attach: no snap to 90
    servos[i].attach(SERVO_PINS[i]);
  }
  delay(500);
  Serial.println(F("millis,base,shoulder,elbow,gripper"));
  Serial.println(F("#MODE,TELEOP"));
}

void loop() {
  readSerial();

  unsigned long now = millis();
  if (now < nextLoop) return;
  nextLoop = now + LOOP_INTERVAL_MS;

  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    float goal;
    if (mode == TELEOP) {
      goal = SMOOTHING * potToAngle(i, analogRead(POT_PINS[i])) + (1.0 - SMOOTHING) * current[i];
    } else {
      goal = target[i];
    }
    float step = constrain(goal - current[i], -MAX_STEP_DEG, MAX_STEP_DEG);
    current[i] = constrain(current[i] + step, JOINT_MIN[i], JOINT_MAX[i]);
    servos[i].write((int)(current[i] + 0.5));
  }

  if (mode == POLICY && now - lastCommandMs > WATCHDOG_MS) {
    for (uint8_t i = 0; i < NUM_JOINTS; i++) target[i] = current[i];   // hold where we are
    if (!watchdogReported) {
      Serial.println(F("#WATCHDOG"));
      watchdogReported = true;
    }
  }

  Serial.print(now);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    Serial.print(',');
    Serial.print((int)(current[i] + 0.5));
  }
  Serial.println();
}
