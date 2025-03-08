from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

def get_estimator():
    return make_pipeline(
        SimpleImputer(strategy='most_frequent'),  
        DummyClassifier(strategy='most_frequent', random_state=60)  
    )
