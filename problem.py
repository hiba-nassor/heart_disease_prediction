import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Heart Disease Prediction'
_target_column_name = 'target'
_ignore_column_names = []  
_prediction_label_names = [0, 1]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.Accuracy(name='acc', precision=4),
]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=60)
    return cv.split(X, y)

# READ DATA
def _read_data(path, f_name):
    data_path = os.path.join(path, 'data', f_name)
    data = pd.read_csv(data_path)
    y_array = data[_target_column_name].values
    X_df = data.drop(columns=[_target_column_name] + _ignore_column_names)
    return X_df, y_array

def get_train_data(path='.'):
    return _read_data(path, 'train.csv')

def get_test_data(path='.'):
    return _read_data(path, 'test.csv') 