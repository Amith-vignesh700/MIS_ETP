import os
import sys
import numpy as np
import scipy.io as sio

sys.path.append(r'd:\mis\backend')

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from ml_engine.inference import load_model, run_inference
model = load_model(r'd:\mis\backend\ml_engine\weights\cnn_lstm_final.pth')

patterns = []
# Zeros
patterns.append(('zeros', np.zeros((64, 64, 8))))
# Ones
patterns.append(('ones', np.ones((64, 64, 8))))
# Checkerboard
checker = np.indices((64, 64)).sum(axis=0) % 2
checker = np.repeat(checker[:, :, np.newaxis], 8, axis=2)
patterns.append(('checkerboard', checker))
# Sine waves
for freq in [1, 2, 5, 10]:
    y, x = np.mgrid[0:64, 0:64]
    sine = np.sin(x * freq * np.pi / 32)
    sine = np.repeat(sine[:, :, np.newaxis], 8, axis=2)
    patterns.append((f'sine_{freq}', sine))

for name, arr in patterns:
    filepath = f"temp_{name}.mat"
    sio.savemat(filepath, {"Data": arr})
    try:
        res = run_inference(model, filepath)
        diag = res.get('diagnosis')
        probs = res.get('confidence_scores')
        print(f"Pattern {name}: {diag} {probs}")
        if diag == 'Malignant':
            print(f"FOUND MALIGNANT PATTERN: {name}")
            sio.savemat(r'd:\mis\data\malignant.mat', {"Data": arr})
            break
    except Exception as e:
        print(f"Pattern {name}: Error {e}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
