import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/heart_disease_uci.csv')

# Drop 'id' and 'dataset' (non-predictive columns) 
columns_to_drop = ['id', 'dataset']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Rename 'num' to 'target' 
if 'num' in df.columns:
    df = df.rename(columns={'num': 'target'})

# Select columns 13 features + target)
expected_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
if set(expected_columns).issubset(df.columns):
    df = df[expected_columns]
else:
    raise ValueError(f"Expected columns {expected_columns} not found in dataset. Found: {df.columns}")

# Check the shape (920, 14) before split
print(f"Dataset shape before split: {df.shape}")

# Split into 80% train 20% test 
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

# Save
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)

print("Data split successfully into data/train.csv and data/test.csv")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
