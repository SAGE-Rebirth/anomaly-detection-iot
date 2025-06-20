import joblib
import numpy as np
import torch
import inspect
import ast

# Load scaler min/max
scaler = joblib.load('scaler.save')
min_val = scaler.data_min_[0]
max_val = scaler.data_max_[0]

# Load anomaly threshold
threshold = float(np.load('anomaly_threshold.npy'))

# Load SEQ_LEN and HIDDEN_SIZE from training config
from train_lstm_autoencoder import SEQ_LEN
# HIDDEN_SIZE is the embedding_dim used in LSTMAutoencoder

# Parse HIDDEN_SIZE from train_lstm_autoencoder.py
with open('train_lstm_autoencoder.py', 'r') as f:
    source = f.read()

class LSTMConfigVisitor(ast.NodeVisitor):
    def __init__(self):
        self.embedding_dim = None
    def visit_ClassDef(self, node):
        if node.name == 'LSTMAutoencoder':
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == '__init__':
                    for arg in stmt.args.args:
                        if arg.arg == 'embedding_dim':
                            # Try to find default value
                            defaults = stmt.args.defaults
                            if defaults:
                                self.embedding_dim = defaults[-1].value

visitor = LSTMConfigVisitor()
visitor.visit(ast.parse(source))
HIDDEN_SIZE = visitor.embedding_dim if visitor.embedding_dim is not None else 16

with open('model_params.h', 'w') as f:
    f.write('#ifndef MODEL_PARAMS_H\n#define MODEL_PARAMS_H\n\n')
    f.write(f'// Min and max for normalization (from training scaler)\n')
    f.write(f'const float TEMP_MIN = {min_val}f;\n')
    f.write(f'const float TEMP_MAX = {max_val}f;\n\n')
    f.write(f'// Anomaly threshold (from training)\n')
    f.write(f'const float ANOMALY_THRESHOLD = {threshold}f;\n\n')
    f.write(f'#define SEQ_LEN {SEQ_LEN}\n')
    f.write(f'#define HIDDEN_SIZE {HIDDEN_SIZE}\n\n')
    f.write('#endif\n')
print('model_params.h generated with scaler min/max, anomaly threshold, SEQ_LEN, and HIDDEN_SIZE.')
