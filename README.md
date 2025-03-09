Heart Disease Prediction

Project Overview

This repository contains a machine learning pipeline for predicting heart disease based on the UCI Heart Disease dataset. The project includes data preprocessing, model training, and evaluation using Random Forest Classifier.

Dataset

The dataset used in this project is the Heart Disease UCI dataset. It consists of 14 features and a target variable indicating the presence of heart disease.

Features

age: Age of the patient

sex: Gender (0 = female, 1 = male)

cp: Chest pain type (categorical: 1-4)

trestbps: Resting blood pressure (mm Hg)

chol: Serum cholesterol (mg/dl)

fbs: Fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false)

restecg: Resting electrocardiographic results (categorical: 0-2)

thalch: Maximum heart rate achieved

exang: Exercise-induced angina (1 = yes, 0 = no)

oldpeak: ST depression induced by exercise relative to rest

slope: Slope of the peak exercise ST segment

ca: Number of major vessels (0-3) colored by fluoroscopy

thal: Thalassemia type (categorical: 0-3)

target: Presence of heart disease (1 = yes, 0 = no)


Installation

To set up the project, follow these steps:

Clone the repository:

git clone https://github.com/hiba-nassor/heart_disease_prediction.git
cd heart-disease-prediction

Create a virtual environment and install dependencies:

conda env create -f environment.yml
conda activate heart_disease_prediction

OR (if using pip)

pip install -r requirements.txt

Usage

Data Preprocessing

Run the following script to preprocess the data and split it into training and testing sets:

python src/split_data.py

Fix Target Encoding

Ensure the target variable is correctly encoded:

python src/fix_target.py

Model Training

Train the Random Forest Classifier by running:

ramp-test --submission my_submission

Model

The Random Forest Classifier is used as the primary model with the following pipeline:

Imputation: Missing values handled using median and mode strategies.

Scaling: Standardization applied to numerical features.

Encoding: One-hot encoding for categorical variables.

Classification: RandomForestClassifier with 100 trees and random_state=60.

Cross-Validation

The model is evaluated using StratifiedShuffleSplit with 8 splits and an 80-20 train-test ratio.

Results

After training, the model achieves competitive performance, ensuring a balance between precision and recall.

Contributing

Feel free to submit issues or pull requests to improve the project!

License

This project is open-source and available under the MIT License.

