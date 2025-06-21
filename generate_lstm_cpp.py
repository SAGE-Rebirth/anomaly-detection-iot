import onnx
import numpy as np

# This script generates C++ code for a single-layer LSTM forward pass using ONNX weights
# It outputs a file: lstm_forward_pass.cpp with the LSTM cell and inference logic

ONNX_PATH = 'lstm_autoencoder.onnx'
CPP_OUT = 'lstm_forward_pass.cpp'

# Helper: Write C++ array from numpy array
def cpp_array(name, arr):
    arr_flat = arr.flatten()
    return f"float {name}[{arr_flat.size}] = {{{','.join(map(str, arr_flat))}}};\n"

def main():
    model = onnx.load(ONNX_PATH)
    weights = {t.name: np.frombuffer(t.raw_data, dtype=np.float32).reshape(tuple(t.dims)) for t in model.graph.initializer}
    print("Available weights and shapes:")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")
    # Stop here so user can see the mapping and we can update the script accordingly
    print("\nPlease review the above weight names and shapes. Update the script to map them to encoder/decoder weights as needed.")
    return

if __name__ == '__main__':
    main()
