// ESP32 LSTM Autoencoder Inference (C++ pseudocode, to be adapted for Arduino)
// Requires: lstm_autoencoder_weights.h, anomaly_threshold.npy (hardcoded)
#include <DHT.h>
#include "lstm_autoencoder_weights.h"
#include "model_params.h"
#include <math.h>

#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

float temp_buffer[SEQ_LEN];
int buf_idx = 0;

void setup() {
  Serial.begin(115200);
  dht.begin();
  pinMode(2, OUTPUT); // LED or buzzer
  for (int i = 0; i < SEQ_LEN; i++) temp_buffer[i] = 0;
  // Error feedback: blink LED if model weights missing (pseudo-check)
  #ifndef LSTM_AUTOENCODER_WEIGHTS_H
    for (int i = 0; i < 10; i++) {
      digitalWrite(2, HIGH); delay(100);
      digitalWrite(2, LOW); delay(100);
    }
    while (1); // Halt
  #endif
}

float normalize_temp(float temp) {
  // Min-max normalization using training scaler values
  return (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);
}

// Helper functions for activation
float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}
float tanh_act(float x) {
  return tanhf(x);
}

// LSTM cell for 1D input, 16 hidden units (assuming 16 from your weights)
// #define HIDDEN_SIZE 16  // Removed, now from model_params.h

// Use the ONNX weights (update names if needed)
extern const float onnx__LSTM_204[64]; // W_ih (input weights)
extern const float onnx__LSTM_205[1024]; // W_hh (recurrent weights)
extern const float onnx__LSTM_206[128]; // b_ih (input bias)
extern const float onnx__LSTM_230[64]; // b_hh (recurrent bias)

// LSTM cell step
void lstm_cell(float x, float* h, float* c) {
  // Gates: i, f, g, o (input, forget, cell, output)
  float gates[4 * HIDDEN_SIZE];
  for (int j = 0; j < 4 * HIDDEN_SIZE; j++) {
    // W_ih: [4*hidden, input] (input is 1D)
    gates[j] = onnx__LSTM_204[j] * x + onnx__LSTM_206[j];
    // W_hh: [4*hidden, hidden]
    for (int k = 0; k < HIDDEN_SIZE; k++) {
      gates[j] += onnx__LSTM_205[j * HIDDEN_SIZE + k] * h[k];
    }
    gates[j] += onnx__LSTM_230[j];
  }
  // Split gates
  float i[HIDDEN_SIZE], f[HIDDEN_SIZE], g[HIDDEN_SIZE], o[HIDDEN_SIZE];
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    i[j] = sigmoid(gates[j]);
    f[j] = sigmoid(gates[HIDDEN_SIZE + j]);
    g[j] = tanh_act(gates[2 * HIDDEN_SIZE + j]);
    o[j] = sigmoid(gates[3 * HIDDEN_SIZE + j]);
  }
  // Update cell and hidden state
  for (int j = 0; j < HIDDEN_SIZE; j++) {
    c[j] = f[j] * c[j] + i[j] * g[j];
    h[j] = o[j] * tanh_act(c[j]);
  }
}

float run_lstm_autoencoder(float* seq_in) {
  // Normalize input sequence
  float norm_seq[SEQ_LEN];
  for (int i = 0; i < SEQ_LEN; i++) {
    norm_seq[i] = normalize_temp(seq_in[i]);
  }
  // Encoder: process sequence
  float h[HIDDEN_SIZE] = {0};
  float c[HIDDEN_SIZE] = {0};
  for (int t = 0; t < SEQ_LEN; t++) {
    lstm_cell(norm_seq[t], h, c);
  }
  // Decoder: repeat last hidden state as input
  float dec_h[HIDDEN_SIZE];
  float dec_c[HIDDEN_SIZE] = {0};
  for (int j = 0; j < HIDDEN_SIZE; j++) dec_h[j] = h[j];
  float output_seq[SEQ_LEN];
  for (int t = 0; t < SEQ_LEN; t++) {
    lstm_cell(0.0f, dec_h, dec_c); // Use 0 as input for decoder (simplified)
    // For demo, use first hidden unit as output (since output size is 1)
    output_seq[t] = dec_h[0];
  }
  // Compute reconstruction error (MSE)
  float mse = 0.0f;
  for (int i = 0; i < SEQ_LEN; i++) {
    float diff = output_seq[i] - norm_seq[i];
    mse += diff * diff;
  }
  mse /= SEQ_LEN;
  return mse;
}

void loop() {
  float temp = dht.readTemperature();
  if (isnan(temp)) {
    Serial.println("Sensor read error");
    delay(1000);
    return;
  }
  temp_buffer[buf_idx] = temp;
  buf_idx = (buf_idx + 1) % SEQ_LEN;
  if (buf_idx == 0) {
    float error = run_lstm_autoencoder(temp_buffer);
    Serial.print("Reconstruction error: ");
    Serial.println(error, 6);
    if (error > ANOMALY_THRESHOLD) {
      digitalWrite(2, HIGH); // Alert
    } else {
      digitalWrite(2, LOW);
    }
  }
  delay(1000);
}
