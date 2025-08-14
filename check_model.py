import pickle
import os

print("Checking existing model...")
try:
    with open('pickle files/randomf.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model)}")
    if hasattr(model, 'n_features_in_'):
        print(f"Model features: {model.n_features_in_}")
    if hasattr(model, 'classes_'):
        print(f"Model classes: {model.classes_}")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

print("\nChecking if framingham.csv exists...")
if os.path.exists('framingham.csv'):
    print("framingham.csv found")
    # Check columns
    import pandas as pd
    df = pd.read_csv('framingham.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if 'TenYearCHD' in df.columns:
        print(f"Target distribution: {df['TenYearCHD'].value_counts()}")
else:
    print("framingham.csv not found!")
