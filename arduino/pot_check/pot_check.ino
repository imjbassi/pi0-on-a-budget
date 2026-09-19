/*
 * pot_check.ino — confirm the four pots are actually connected.
 *
 * pot_calibrate.ino reports the range each pot reached, but a pot whose lug
 * has lost contact reads the same full 0..1023 range: a floating analog pin
 * drifts across the whole scale on its own. This sketch tells the two apart.
 *
 * Prints all four raw ADC values ~5x a second, plus how much each one moved
 * since the last print.
 *
 * A CONNECTED pot:
 *   - holds steady when you aren't touching it (drift of a couple of counts)
 *   - sits near 512 with the knob centred
 *   - changes smoothly and monotonically as you turn the knob
 *
 * A DISCONNECTED (floating) pin:
 *   - wanders by tens or hundreds of counts while you touch nothing
 *   - jumps when you move your hand near the breadboard
 *   - ignores the knob
 *
 * Serial monitor at 115200 baud.
 */

const uint8_t POT_PINS[] = {A0, A1, A2, A3};
const char* JOINT_NAMES[] = {"base", "shoulder", "elbow", "gripper"};

int previous[4];

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < 4; i++) {
    previous[i] = analogRead(POT_PINS[i]);
  }
  Serial.println(F("pot_check — leave the knobs alone first: values should barely move."));
  Serial.println(F("Then turn each knob and watch only that joint's value change.\n"));
  Serial.println(F("     base        shoulder      elbow        gripper"));
}

void loop() {
  for (uint8_t i = 0; i < 4; i++) {
    int value = analogRead(POT_PINS[i]);
    int delta = value - previous[i];
    previous[i] = value;

    Serial.print(JOINT_NAMES[i][0]);
    Serial.print(':');
    Serial.print(value);
    Serial.print(F(" ("));
    if (delta >= 0) Serial.print('+');
    Serial.print(delta);
    Serial.print(F(")\t"));
  }
  Serial.println();
  delay(200);
}
