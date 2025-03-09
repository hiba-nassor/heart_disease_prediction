from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from lightgbm import LGBMClassifier
import pandas as pd

# Column renaming mapping
column_mapping = {
    'id': 'patient_id',
    'age': 'age',
    'sex': 'sex',
    'dataset': 'place_of_study',
    'cp': 'chest_pain_type',
    'trestbps': 'resting_blood_pressure',
    'chol': 'serum_cholesterol',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'resting_ecg_results',
    'thalch': 'max_heart_rate_achieved',
    'exang': 'exercise_induced_angina',
    'oldpeak': 'st_depression',
    'slope': 'st_slope',
    'ca': 'num_major_vessels',
    'thal': 'thalassemia',
    'num': 'heart_disease_presence'
}

# Columns to drop
drop_cols = ['st_slope', 'num_major_vessels', 'thalassemia', 'patient_id', 'place_of_study']

# Function to rename columns
def rename_columns(X):
    return X.rename(columns=column_mapping)

# Function to drop unnecessary columns
def drop_unnecessary_columns(X):
    return X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')

# Define feature columns
numeric_cols = ['age', 'resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate_achieved', 'st_depression']
categorical_cols = ['sex', 'chest_pain_type', 'resting_ecg_results', 'exercise_induced_angina']

# Define transformers
rename_transformer = FunctionTransformer(rename_columns, validate=False)
drop_columns_transformer = FunctionTransformer(drop_unnecessary_columns, validate=False)

# Define column transformer for preprocessing
column_transformer = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_cols),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(drop='first', sparse_output=False)), categorical_cols),
    remainder='passthrough'
)

# Define pipeline
def get_estimator():
    return make_pipeline(
        rename_transformer,  # Rename columns
        drop_columns_transformer,  # Drop unnecessary columns
        column_transformer,  # Transform numerical & categorical columns
        LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=60)  # LightGBM Classifier
    )
