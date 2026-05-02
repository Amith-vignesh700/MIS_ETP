import os
import sys
import numpy as np

sys.path.append(r'd:\mis\backend')

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from ml_engine.inference import load_model, run_inference
model = load_model(r'd:\mis\backend\ml_engine\weights\cnn_lstm_final.pth')

directory = r'd:\mis\data\phantom'
for filename in os.listdir(directory):
    if filename.endswith('.mat'):
        filepath = os.path.join(directory, filename)
        try:
            res = run_inference(model, filepath)
            diag = res.get('diagnosis')
            if diag == 'Malignant':
                print(f'FOUND MALIGNANT: {filename}')
                break
        except Exception as e:
            pass
