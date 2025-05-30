// Arduino/motor_control.ino
/*─────────────────────────────────────────────────────────────
   HPE Candy Demo –  Bidirectional UART motor controller
   -----------------------------------------------------------
   • Computer sends ASCII commands terminated by '\n'
       SPIN:<0‑255>          → set PWM duty  (configuration only)
       TIME:<milliseconds>   → set run time  (configuration only)
       SMILE                 → run once with current settings
   • Arduino acknowledges every command and reports completion.
─────────────────────────────────────────────────────────────*/

const byte MOTOR_PWM_PIN = 5;          // D5 on Nano (must be PWM‑capable)
const byte LED_PIN = LED_BUILTIN; // Or 13 if LED_BUILTIN isn't defined for your board setup

// ---------- user‑configurable defaults ----------
uint8_t  spinDuty     = 100;           // 0‑255 ( ≈ 39 % duty )
uint32_t runDuration  = 1000;          // ms    (1 s)
const unsigned int BAUD = 115200;      // fast, but match your PC
// ------------------------------------------------

// ── helper prototypes ───────────────────────────
void processCommand(const String& line);
void runMotorOnce();

void setup() {
  // Debugging LED
  pinMode(LED_PIN, OUTPUT);       // Setup the LED pin
  digitalWrite(LED_PIN, LOW);   // Turn LED off initially

  Serial.begin(BAUD);
  while (!Serial) { /* wait for serial on some boards */ }

  pinMode(MOTOR_PWM_PIN, OUTPUT);
  analogWrite(MOTOR_PWM_PIN, 0);       // ensure motor off

  Serial.println(F("ARDUINO READY"));
  Serial.print  (F("Defaults  SPIN="));  Serial.print(spinDuty);
  Serial.print  (F("  TIME="));          Serial.print(runDuration);
  Serial.println(F(" ms"));
  Serial.println(F("Send commands:  SPIN:x  TIME:y  SMILE"));
}

void loop() {
  // Read serial lines non‑blocking
  if (Serial.available()) {
    digitalWrite(LED_PIN, HIGH); // Turn LED ON when data is available

    String line = Serial.readStringUntil('\n');
    line.trim();                        // remove CR/LF & whitespace
    if (line.length() > 0) {
      // You could add another quick blink here to show a non-empty line was processed
      processCommand(line);
    }
    // if (line.length()) processCommand(line);

    digitalWrite(LED_PIN, LOW);  // Turn LED OFF after processing
  }

  // ...add any background tasks here...
}

/*─────────────────────────────────────────────────────────────
                       Command parser
─────────────────────────────────────────────────────────────*/
void processCommand(const String& line) {
  if (line.startsWith("SPIN:")) {
    int val = line.substring(5).toInt();
    val = constrain(val, 0, 255);
    spinDuty = static_cast<uint8_t>(val);
    Serial.print(F("ACK SPIN set to "));
    Serial.println(spinDuty);
  }
  else if (line.startsWith("TIME:")) {
    uint32_t val = line.substring(5).toInt();
    // minimal sanity: 10 ms ≤ TIME ≤ 30 000 ms
    runDuration = constrain(val, 10UL, 30000UL);
    Serial.print(F("ACK TIME set to "));
    Serial.print(runDuration);
    Serial.println(F(" ms"));
  }
  else if (line.equalsIgnoreCase("SMILE")) {
    Serial.println(F("CMD SMILE received – activating motor"));
    runMotorOnce();
    Serial.println(F("DONE motor cycle complete"));
  }
  else {
    Serial.print(F("ERR unknown command: "));
    Serial.println(line);
  }
}

/*─────────────────────────────────────────────────────────────
                     Motor control helper
─────────────────────────────────────────────────────────────*/
void runMotorOnce() {
  analogWrite(MOTOR_PWM_PIN, spinDuty);
  delay(runDuration);
  analogWrite(MOTOR_PWM_PIN, 0);
}