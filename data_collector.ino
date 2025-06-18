#include <DHT.h>

#define DHTPIN 4         // GPIO pin where the DHT22 is connected
#define DHTTYPE DHT22    // DHT 22 (AM2302)
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();
  Serial.println("timestamp,temperature_C");
  // Error feedback: blink LED if sensor not found (optional)
  pinMode(2, OUTPUT);
  float t = dht.readTemperature();
  if (isnan(t)) {
    for (int i = 0; i < 5; i++) {
      digitalWrite(2, HIGH); delay(200);
      digitalWrite(2, LOW); delay(200);
    }
  }
}

void loop() {
  float temp = dht.readTemperature();
  if (isnan(temp)) {
    Serial.println("Sensor read error");
    delay(1000);
    return;
  }
  unsigned long now = millis();
  if (!isnan(temp)) {
    Serial.print(now / 1000.0, 2); // seconds since boot
    Serial.print(",");
    Serial.println(temp, 2);
  }
  delay(1000);
}
