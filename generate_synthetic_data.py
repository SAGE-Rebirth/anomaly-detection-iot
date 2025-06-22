import numpy as np
import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic temperature data with anomalies.')
    parser.add_argument('--minutes', type=int, default=24*60, help='Total minutes of data (default: 1 day)')
    parser.add_argument('--anomalies', type=int, default=10, help='Number of anomalies to inject')
    parser.add_argument('--outfile', type=str, default='temperature_log.csv', help='Output CSV file')
    args = parser.parse_args()

    total_minutes = args.minutes
    anomaly_count = args.anomalies
    outfile = args.outfile
    base_temp = 25.0
    noise_std = 0.3
    anomaly_magnitude = 5.0

    np.random.seed(42)

    # Generate normal temperature data (sinusoidal daily pattern + noise)
    time = np.arange(total_minutes)
    daily_pattern = 2 * np.sin(2 * np.pi * time / (24 * 60))  # 24h cycle
    normal_temps = base_temp + daily_pattern + np.random.normal(0, noise_std, total_minutes)

    data = pd.DataFrame({
        'timestamp': time,
        'temperature_C': normal_temps
    })

    # Inject anomalies (spikes)
    anomaly_indices = np.random.choice(total_minutes, anomaly_count, replace=False)
    for idx in anomaly_indices:
        data.loc[idx, 'temperature_C'] += anomaly_magnitude * (1 if np.random.rand() > 0.5 else -1)
        data.loc[idx, 'is_anomaly'] = 1

    data['is_anomaly'] = data['is_anomaly'].fillna(0).astype(int)

    # Save to CSV
    try:
        data.to_csv(outfile, index=False)
        print(f'Synthetic {outfile} generated with anomalies.')
    except Exception as e:
        print(f'Error writing file: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print(f'Missing package: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
