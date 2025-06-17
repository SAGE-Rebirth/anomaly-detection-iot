import onnx
import numpy as np
import argparse
import sys

# Load ONNX model
def extract_weights(onnx_path):
    model = onnx.load(onnx_path)
    weights = {}
    for tensor in model.graph.initializer:
        arr = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(tuple(tensor.dims))
        weights[tensor.name] = arr
    return weights

def write_header(weights, out_path):
    with open(out_path, 'w') as f:
        f.write('#ifndef LSTM_AUTOENCODER_WEIGHTS_H\n#define LSTM_AUTOENCODER_WEIGHTS_H\n\n')
        for name, arr in weights.items():
            arr_flat = arr.flatten()
            f.write(f'// {name}\n')
            f.write(f'const float {name}[{arr_flat.size}] = {{')
            f.write(','.join(map(str, arr_flat)))
            f.write('};\n\n')
        f.write('#endif\n')

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX weights to C header.')
    parser.add_argument('--onnx', type=str, default='lstm_autoencoder.onnx', help='Input ONNX file')
    parser.add_argument('--header', type=str, default='lstm_autoencoder_weights.h', help='Output header file')
    args = parser.parse_args()
    try:
        weights = extract_weights(args.onnx)
        if not weights:
            print('Error: No weights found in ONNX file.', file=sys.stderr)
            sys.exit(1)
        write_header(weights, args.header)
        print(f'Header file generated: {args.header}')
    except FileNotFoundError:
        print(f'Error: ONNX file {args.onnx} not found.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
