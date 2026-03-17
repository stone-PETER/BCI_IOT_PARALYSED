"""Quick script to inspect BCI IV 2b MAT file structure."""
import scipy.io
import numpy as np


def balance_summary(labels, tolerance=0.10):
    """Return class balance details for binary labels.

    tolerance=0.10 means a max allowed class-ratio deviation of 10% from 1.0.
    """
    flattened = labels.reshape(-1)
    unique, counts = np.unique(flattened, return_counts=True)
    distribution = {int(k): int(v) for k, v in zip(unique, counts)}

    if len(counts) < 2:
        return distribution, "UNKNOWN", float("inf")

    ratio = float(np.max(counts) / np.min(counts))
    is_balanced = ratio <= (1.0 + tolerance)
    status = "BALANCED" if is_balanced else "IMBALANCED"
    return distribution, status, ratio

mat = scipy.io.loadmat('BCI/bci4_2b/B01T.mat')
print("="*70)
print("BCI IV 2b MAT File Structure")
print("="*70)
print("\nTop-level keys:", list(mat.keys()))

data = mat['data']
print(f"\ndata shape: {data.shape}")
print(f"data dtype: {data.dtype}")
print(f"\nNote: data has {data.shape[1]} sessions/runs")

# Access first session
data_struct = data[0, 0]
print(f"\ndata[0,0] (Session 1) dtype: {data_struct.dtype}")

if hasattr(data_struct.dtype, 'names'):
    print(f"\nField names: {data_struct.dtype.names}")
    
    for field in data_struct.dtype.names:
        field_data = data_struct[field][0, 0]
        print(f"\n{field}:")
        print(f"  Type: {type(field_data)}")
        print(f"  Shape: {field_data.shape if hasattr(field_data, 'shape') else 'N/A'}")
        print(f"  Dtype: {field_data.dtype if hasattr(field_data, 'dtype') else 'N/A'}")
        
        if field == 'X':
            print(f"  EEG Signal shape (samples, channels): {field_data.shape}")
        elif field == 'y':
            print(f"  Labels (unique classes): {np.unique(field_data)}")
            print(f"  Labels shape: {field_data.shape}")
        elif field == 'fs':
            print(f"  Sampling rate: {field_data} Hz")
        elif field == 'trial':
            print(f"  Trial indices shape: {field_data.shape}")
            if field_data.shape[0] < 20:
                print(f"  Trial indices (first few): {field_data[:5] if len(field_data) > 5 else field_data}")

print("\n" + "="*70)
print("Checking all 3 sessions:")
all_labels = []
for i in range(data.shape[1]):
    sess_data = data[0, i]
    X = sess_data['X'][0, 0]
    y = sess_data['y'][0, 0]
    trial = sess_data['trial'][0, 0]
    all_labels.append(y.reshape(-1))
    dist, status, ratio = balance_summary(y)

    print(f"\nSession {i+1}:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  trial shape: {trial.shape}")
    print(f"  Unique labels: {np.unique(y)}")
    print(f"  Class distribution: {dist}")
    print(f"  Balance status: {status} (max/min ratio: {ratio:.3f})")

print("\n" + "="*70)
all_labels = np.concatenate(all_labels)
overall_dist, overall_status, overall_ratio = balance_summary(all_labels)
print("Overall dataset balance across all sessions:")
print(f"  Class distribution: {overall_dist}")
print(f"  Balance status: {overall_status} (max/min ratio: {overall_ratio:.3f})")

