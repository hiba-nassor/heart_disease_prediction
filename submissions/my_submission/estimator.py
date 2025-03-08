from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
numeric_cols = ['age', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak'] 
categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'ca', 'thal']  

transformer = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_cols),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(drop='first', sparse_output=False)), categorical_cols),
    remainder='passthrough'
)

def get_estimator():
    return make_pipeline(
        transformer,  
        RandomForestClassifier(n_estimators=100, random_state=60)  
    )
