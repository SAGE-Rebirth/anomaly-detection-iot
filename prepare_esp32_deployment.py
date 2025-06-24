import os
import subprocess
import sys

# Unified build script for ESP32 deployment
# Runs all steps: data generation, training, ONNX export, header generation, param extraction, and C++ LSTM code generation

def run(cmd, desc):
    print(f"\n[Step] {desc}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}")
        sys.exit(1)

def main():
    # 1. Generate synthetic data
    run("python generate_synthetic_data.py", "Generate synthetic temperature data")
    # 2. Train model
    run("python train_lstm_autoencoder.py", "Train LSTM autoencoder and export ONNX")
    # 3. Convert ONNX weights to C header
    run("python convert_to_header.py", "Convert ONNX weights to C header")
    # 4. Extract scaler min/max and threshold to header
    run("python extract_params_to_header.py", "Extract scaler min/max and threshold to C header")
    # 5. Generate C++ LSTM forward pass code
    run("python generate_lstm_cpp.py", "Generate C++ LSTM forward pass code for ESP32")
    print("\nAll steps completed. Copy generated headers and C++ code to your ESP32 project.")

if __name__ == "__main__":
    main()
