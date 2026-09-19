/*
 * limit_finder.ino — find safe JOINT_MIN / JOINT_MAX for a built arm.
 *
 * Jog one joint at a time in small steps, mark the angles where the joint
 * reaches its safe travel, and get paste-ready arrays for teleop.ino and
 * policy_follower.ino.
 *
 * This exists because the EEZYbotARM's linkage couples the joints: angles
 * that are fine on their own can drive the arm into itself. Servo gears
 * strip when that happens, so find the limits by hand, slowly, once.
 *
 * Wiring is the same as teleop.ino (servo signals on D3/D5/D6/D9).
 * The pots are ignored here — this sketch never moves a joint on its own.
 *
 * Serial monitor at 115200 baud, "Newline" line ending.
 *
 *   1 2 3 4   select joint (base / shoulder / elbow / gripper)
 *   a / d     jog -1 / +1 degree
 *   z / c     jog -5 / +5 degrees
 *   n         mark this angle as the joint's MIN
 *   x         mark this angle as the joint's MAX
 *   h         go to 90 degrees (midpoint)
 *   p         print the JOINT_MIN[] / JOINT_MAX[] arrays so far
 *   ?         print this help
 *
 * Marks start at the current angle, so a joint you haven't marked reports
 * its own position rather than a guess. Mark MIN and MAX for all four.
 *
 * SAFETY
 *   - Move in single degrees near a limit. If a servo buzzes, strains, or
 *     the arm stops moving while the number keeps changing, you are past
 *     the limit: back off and mark there, not where it stalled.
 *   - Leave a few degrees of margin on each end.
 *   - Keep a hand near the power switch.
 */

#include <Servo.h>

const uint8_t NUM_JOINTS = 4;
const uint8_t SERVO_PINS[] = {3, 5, 6, 9};
const char* JOINT_NAMES[] = {"base", "shoulder", "elbow", "gripper"};

// Deliberately wide while searching — the whole point is to discover the
// real limits. Do NOT copy these into the other sketches.
const int SEARCH_MIN = 0;
const int SEARCH_MAX = 180;

const unsigned long STEP_DELAY_MS = 15;   // per degree, so jogs are visibly slow

Servo servos[NUM_JOINTS];
int angle[NUM_JOINTS];
int markedMin[NUM_JOINTS];
int markedMax[NUM_JOINTS];
uint8_t joint = 0;

void printHelp() {
  Serial.println(F("keys: 1-4 select joint | a/d jog -1/+1 | z/c jog -5/+5"));
  Serial.println(F("      n mark MIN | x mark MAX | h go to 90 | p print arrays"));
}

void printState() {
  Serial.print(F("joint "));
  Serial.print(joint);
  Serial.print(F(" ("));
  Serial.print(JOINT_NAMES[joint]);
  Serial.print(F(")  angle "));
  Serial.print(angle[joint]);
  Serial.print(F("  marked min "));
  Serial.print(markedMin[joint]);
  Serial.print(F(" max "));
  Serial.println(markedMax[joint]);
}

void printArrays() {
  Serial.println(F("\n// paste into teleop.ino AND policy_follower.ino"));
  Serial.print(F("const int JOINT_MIN[] = {"));
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    Serial.print(markedMin[i]);
    if (i < NUM_JOINTS - 1) Serial.print(F(", "));
  }
  Serial.println(F("};"));
  Serial.print(F("const int JOINT_MAX[] = {"));
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    Serial.print(markedMax[i]);
    if (i < NUM_JOINTS - 1) Serial.print(F(", "));
  }
  Serial.println(F("};"));

  Serial.println(F("\n// gripper angles for config/robot.json:"));
  Serial.print(F("//   note which of "));
  Serial.print(markedMin[NUM_JOINTS - 1]);
  Serial.print(F(" / "));
  Serial.print(markedMax[NUM_JOINTS - 1]);
  Serial.println(F(" is OPEN and which is CLOSED\n"));
}

void jog(int delta) {
  int target = constrain(angle[joint] + delta, SEARCH_MIN, SEARCH_MAX);
  int step = (target > angle[joint]) ? 1 : -1;
  while (angle[joint] != target) {
    angle[joint] += step;
    servos[joint].write(angle[joint]);
    delay(STEP_DELAY_MS);
  }
  printState();
}

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    angle[i] = 90;
    markedMin[i] = 90;
    markedMax[i] = 90;
    servos[i].write(90);            // set pulse before attach: no snap
    servos[i].attach(SERVO_PINS[i]);
  }
  delay(500);
  Serial.println(F("limit_finder — all joints at 90 degrees"));
  Serial.println(F("If a joint is badly off its midpoint, power down and"));
  Serial.println(F("re-seat that horn before continuing."));
  printHelp();
  printState();
}

void loop() {
  if (Serial.available() <= 0) return;
  char c = (char)Serial.read();

  switch (c) {
    case '1': case '2': case '3': case '4':
      joint = c - '1';
      printState();
      break;
    case 'a': jog(-1); break;
    case 'd': jog(+1); break;
    case 'z': jog(-5); break;
    case 'c': jog(+5); break;
    case 'h': jog(90 - angle[joint]); break;
    case 'n':
      markedMin[joint] = angle[joint];
      Serial.print(F("marked MIN "));
      printState();
      break;
    case 'x':
      markedMax[joint] = angle[joint];
      Serial.print(F("marked MAX "));
      printState();
      break;
    case 'p': printArrays(); break;
    case '?': printHelp(); break;
    default: break;                 // ignore newlines and stray characters
  }
}
