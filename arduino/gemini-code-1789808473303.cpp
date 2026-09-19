const uint8_t POT_PINS[] = {A0, A1, A2, A3};
int potMin[] = {1023, 1023, 1023, 1023};
int potMax[] = {0, 0, 0, 0};

void setup() {
  Serial.begin(115200);
  Serial.println("Twist all knobs fully left and right...");
}

void loop() {
  bool changed = false;
  for (int i = 0; i < 4; i++) {
    int val = analogRead(POT_PINS[i]);
    if (val < potMin[i]) { potMin[i] = val; changed = true; }
    if (val > potMax[i]) { potMax[i] = val; changed = true; }
  }

  if (changed) {
    Serial.print("POT_MIN[] = {");
    for (int i = 0; i < 4; i++) { Serial.print(potMin[i]); if (i < 3) Serial.print(", "); }
    Serial.println("};");
    
    Serial.print("POT_MAX[] = {");
    for (int i = 0; i < 4; i++) { Serial.print(potMax[i]); if (i < 3) Serial.print(", "); }
    Serial.println("};");
    Serial.println("--------------------");
  }
  delay(50);
}