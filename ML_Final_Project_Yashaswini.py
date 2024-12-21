#!/usr/bin/env python
# coding: utf-8

# In[171]:


# Loading the data set
import pandas as pd

# Define column names as per the `adult.names` file
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

# Load the dataset from the .data file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data = pd.read_csv(url, header=None, names=columns, na_values=" ?")

# Save as CSV
data.to_csv("adult_income.csv", index=False)
print("Dataset saved as adult_income.csv")
# 32,561 rows and 15 columns.


# In[172]:


# Create a normalized database (3NF).
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'adult_income.csv'
data = pd.read_csv(file_path)

# Step 1: Handle Missing Values
# Fill missing values in categorical columns with the mode
missing_columns = ['workclass', 'occupation', 'native_country']
for col in missing_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Step 2: Remove Duplicates
data.drop_duplicates(inplace=True)

# Step 3: Standardize Categorical Columns
# Strip leading/trailing whitespaces in categorical columns
categorical_columns = [
    'workclass', 'education', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'native_country', 'income'
]
for col in categorical_columns:
    data[col] = data[col].str.strip()

# Step 4: Encode Categorical Variables
# Initialize a LabelEncoder for all categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for inverse transformation if needed

# Step 5: Feature Engineering (Optional Enhancements)
# Create bins for continuous variables like 'age'
data['age_group'] = pd.cut(data['age'], bins=[0, 25, 45, 65, 100], labels=['Youth', 'Adult', 'Middle-Aged', 'Senior'])

# Step 6: Normalize the Dataset (3NF)
# Break the dataset into normalized tables based on logical grouping

# Table 1: Demographics
demographics = data[['age', 'age_group', 'race', 'sex', 'native_country']]
demographics = demographics.drop_duplicates().reset_index(drop=True)

# Table 2: Work Information
work_info = data[['workclass', 'occupation', 'education', 'education_num']]
work_info = work_info.drop_duplicates().reset_index(drop=True)

# Table 3: Financial Information
financial_info = data[['fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']]
financial_info = financial_info.drop_duplicates().reset_index(drop=True)

# Table 4: Income (Target Variable)
income_info = data[['income']].drop_duplicates().reset_index(drop=True)

# Step 7: Save Normalized Tables
demographics.to_csv("demographics.csv", index=False)
work_info.to_csv("work_info.csv", index=False)
financial_info.to_csv("financial_info.csv", index=False)
income_info.to_csv("income_info.csv", index=False)

print("Data preprocessing and normalization (3NF) complete!")


# In[173]:


#Write SQL join statement to fetch data from the database and into Pandas DataFrame.

import sqlite3
import pandas as pd

# Step 1: Create SQLite Database
db_name = "adult_income.db"
conn = sqlite3.connect(db_name)

# Load normalized tables into SQLite database
tables = {
    "demographics": "demographics.csv",
    "work_info": "work_info.csv",
    "financial_info": "financial_info.csv",
    "income_info": "income_info.csv"
}

for table_name, file_name in tables.items():
    df = pd.read_csv(file_name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)

print("SQLite database created with normalized tables!")

# Step 2: Write SQL Join Statement to Fetch Data
# SQL query to join all normalized tables
query = """
SELECT 
    d.age, d.age_group, d.race, d.sex, d.native_country,
    w.workclass, w.occupation, w.education, w.education_num,
    f.fnlwgt, f.capital_gain, f.capital_loss, f.hours_per_week,
    i.income
FROM demographics d
JOIN work_info w ON w.rowid = d.rowid
JOIN financial_info f ON f.rowid = d.rowid
JOIN income_info i ON i.rowid = d.rowid
"""

# Execute the query and fetch the result into a Pandas DataFrame
merged_data = pd.read_sql_query(query, conn)

# Step 3: Save the Fetched Data to a CSV for Further Processing
merged_data.to_csv("merged_data.csv", index=False)

print("Data fetched and merged successfully! Saved to 'merged_data.csv'.")


# In[174]:


#Explore the data to determine if you need to stratify it by some attribute when doing train/test split. Perform the train/test split.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Step 1: Explore the Dataset
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics for Numerical Columns:")
print(data.describe())

# Step 2: Check for Missing Values
print("\nMissing Values Count:")
print(data.isnull().sum())

# Step 3: Explore Target Variable Distribution
print("\nDistribution of the target variable (income):")
income_distribution = data['income'].value_counts(normalize=True)
print(income_distribution)

# Visualize Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='income', data=data)
plt.title("Distribution of Income Target Variable")
plt.xlabel("Income Category")
plt.ylabel("Count")
plt.show()

# Step 4: Explore Categorical Attributes
categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
print("\nCategorical Variables Distributions:")
for col in categorical_columns:
    print(f"\n{col} distribution:")
    print(data[col].value_counts(normalize=True))

    # Visualize distributions
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=data, order=data[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.show()

# Step 5: Perform Train/Test Split with Stratification
# Separate features (X) and target variable (y)
X = data.drop('income', axis=1)  # Features
y = data['income']  # Target variable

# Perform the split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Verify Stratification
print("\nTarget variable distribution in the original dataset:")
print(y.value_counts(normalize=True))
print("\nTarget variable distribution in the training set:")
print(y_train.value_counts(normalize=True))
print("\nTarget variable distribution in the testing set:")
print(y_test.value_counts(normalize=True))

# Step 6: Save the Train/Test Splits for Future Use
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nTrain/test split complete. Files saved as:")
print("X_train.csv, X_test.csv, y_train.csv, y_test.csv")


# In[175]:


#Explore the data using yprofile and correlation matrix. Make observations about features, distributions, capped values, and missing values. Create a list of data cleanup tasks.	10		
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Step 1: Analyze the Target Variable (y-profile)
target_column = 'income'

print(f"\n### Basic Info for Target Variable '{target_column}' ###")
print(data[target_column].describe())
print("\nValue Counts:")
print(data[target_column].value_counts())

# Visualize Target Variable Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=target_column, data=data)
plt.title(f"Distribution of Target Variable: {target_column}")
plt.xlabel(target_column)
plt.ylabel("Count")
plt.show()

# Relationships Between Target and Categorical Features
categorical_columns = [
    'workclass', 'education', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'native_country'
]

for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, hue=target_column, data=data)
    plt.title(f"{col} vs {target_column}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title=target_column)
    plt.xticks(rotation=45)
    plt.show()

# Step 2: Correlation Matrix for Numerical Features
# Select only numerical columns for the correlation matrix
numerical_data = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_data.corr()

# Visualize the Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# Step 3: Observations About Features, Distributions, Capped Values, and Missing Values
observations = []

# Check for capped values in numerical features
for col in numerical_data.columns:
    max_value = data[col].max()
    if max_value > data[col].quantile(0.95):
        observations.append(f"Feature '{col}' appears to have capped values at {max_value}.")

# Check for imbalanced distributions in categorical features
for col in categorical_columns:
    imbalance = data[col].value_counts(normalize=True).iloc[0]
    if imbalance > 0.9:
        observations.append(f"Feature '{col}' is highly imbalanced with {imbalance * 100:.2f}% in one category.")

# Check for missing values
missing_values_count = data.isnull().sum()
if missing_values_count.sum() == 0:
    observations.append("No missing values found in the dataset.")
else:
    for col, count in missing_values_count.items():
        if count > 0:
            observations.append(f"Feature '{col}' has {count} missing values.")

# Step 4: Create a List of Data Cleanup Tasks
cleanup_tasks = []

# Suggest cleanup actions based on observations
for obs in observations:
    if "capped values" in obs:
        cleanup_tasks.append("Inspect the feature for possible outlier treatment or normalization.")
    elif "highly imbalanced" in obs:
        cleanup_tasks.append("Consider rebalancing the feature with techniques like oversampling or under-sampling.")
    elif "missing values" in obs:
        cleanup_tasks.append("Handle missing values using imputation techniques or removal.")

# Print Observations and Cleanup Tasks
print("\n### Observations ###")
for obs in observations:
    print(f"- {obs}")

print("\n### Data Cleanup Tasks ###")
for task in cleanup_tasks:
    print(f"- {task}")


# In[ ]:


# Printing this list of cleanup tasks
# ### Observations ###
# - Feature 'age' appears to have capped values at 90.
# - Feature 'fnlwgt' appears to have capped values at 1484705.
# - Feature 'education_num' appears to have capped values at 16.
# - Feature 'capital_gain' appears to have capped values at 99999.
# - Feature 'capital_loss' appears to have capped values at 4356.
# - Feature 'hours_per_week' appears to have capped values at 99.
# - Feature 'native_country' is highly imbalanced with 91.22% in one category.
# - Feature 'workclass' has 1836 missing values.
# - Feature 'occupation' has 1843 missing values.
# - Feature 'native_country' has 583 missing values.

# ### Data Cleanup Tasks ###
# - Inspect the feature for possible outlier treatment or normalization.
# - Inspect the feature for possible outlier treatment or normalization.
# - Inspect the feature for possible outlier treatment or normalization.
# - Inspect the feature for possible outlier treatment or normalization.
# - Inspect the feature for possible outlier treatment or normalization.
# - Inspect the feature for possible outlier treatment or normalization.
# - Consider rebalancing the feature with techniques like oversampling or under-sampling.
# - Handle missing values using imputation techniques or removal.
# - Handle missing values using imputation techniques or removal.
# - Handle missing values using imputation techniques or removal.


# In[177]:


# Experiment1:Experiment #1: Create a pipeline for preprocessing (StandardScaler, MinMaxScaler, LogTransformation, OneHotEncoding) and Logistic Regression. Log F1-score/(TP,TN,FN,FP)  in MLFlow on DagsHub. – Cross validation 3/10 folds. Results—mean/std of CV results and results on the whole training data – add in parameter hyper tuning

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

# Load Dataset
file_path = "adult_income.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

# Separate Features and Target
X = data.drop('income', axis=1)
y = data['income']

# Convert Target Variable to Numerical Labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define Preprocessing for Numerical and Categorical Columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('log', FunctionTransformer(np.log1p, validate=True)),  # Log Transformation
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine Preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# In[178]:


# Intergrate model
# Experiment 1 - Step2

from sklearn.linear_model import LogisticRegression

# Create the Model Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# In[179]:


# Experiment1-3

from sklearn.model_selection import cross_val_score, train_test_split

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Perform Cross-Validation
cv_scores_3 = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1')
cv_scores_10 = cross_val_score(pipeline, X_train, y_train, cv=10, scoring='f1')

print(f"3-Fold CV Mean F1: {cv_scores_3.mean():.4f}, Std: {cv_scores_3.std():.4f}")
print(f"10-Fold CV Mean F1: {cv_scores_10.mean():.4f}, Std: {cv_scores_10.std():.4f}")


# In[180]:


# Experiment1-4 Train,Model and Evalute
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Train Model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)
print(f"\nF1 Score on Test Data: {f1:.4f}")

# Display TP, TN, FP, FN
tn, fp, fn, tp = conf_matrix.ravel()
print(f"\nTrue Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")


# In[22]:


# Experiment1-step5 Parameter tuning
from sklearn.model_selection import GridSearchCV

# Define Parameter Grid for Hyperparameter Tuning
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga']
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Parameters
print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

# Best Model Evaluation
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Confusion Matrix and F1 Score for Best Model
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
print("\nConfusion Matrix for Best Model:")
print(conf_matrix_best)
print(f"\nF1 Score for Best Model: {f1_best:.4f}")


# In[181]:


# Experiment #2 step1
# Importing the required libraries and pre-processing pipelines

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import mlflow
import mlflow.sklearn



# Separate Features and Target
X = data.drop('income', axis=1)
y = data['income']

# Convert Target Variable to Numerical Labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Define Preprocessing for Numerical and Categorical Columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine Preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# In[182]:


# Experiment 2 - step2 Define the classifiers and the pipelines

# Define Classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RidgeClassifier": RidgeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Create Pipelines
pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)]) 
             for name, clf in classifiers.items()}


# In[183]:


# Experiment 2 - step3 train/test split
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[186]:


# Step 1: Upgrade pip
get_ipython().system('python -m pip install --upgrade pip')

# Step 2: Reinstall dagshub with forced dependency resolution
get_ipython().system('pip install --force-reinstall dagshub==0.4.0')

# Step 3: Resolve specific dependency conflicts
# Install compatible version of botocore for aiobotocore
get_ipython().system('pip install "botocore>=1.29.76,<1.29.77"')

# Install compatible versions of boto3 and s3transfer
get_ipython().system('pip install boto3==1.26.70 s3transfer==0.5.2')

# Install compatible version of numpy for ydata-profiling and scikit-learn
get_ipython().system('pip install numpy==1.21.6')

# Install dacite for ydata-profiling compatibility
get_ipython().system('pip install dacite==1.6.0')

# Resolve missing dependencies
get_ipython().system('pip install FuzzyTM tables blosc2 cython markdown-it-py==2.2.0')

# Step 4: Verify installation of dagshub and other critical libraries
try:
    import dagshub
    import mlflow
    import numpy
    import pandas
    print("All critical libraries installed successfully!")
except ImportError as e:
    print("Error:", e)


# In[187]:


# Base for the experiment 2 creating and setting up the dagshub
import dagshub

# Initialize DagsHub for your repository
dagshub.init(
    repo_owner="yashaswiniguntupalli",
    repo_name="ML_Final_Project",
    mlflow=True
)


# In[188]:


import mlflow

mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")


# In[189]:


import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "yashaswiniguntupalli"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "dd928eb3e01ad92df47ae00f812f06a28ddc8c95"


# In[65]:


experiment_name = "Experiment_2"

# Create the experiment if it doesn't exist
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

# Set the experiment
mlflow.set_experiment(experiment_name)


# In[190]:


# Testing
import mlflow

# Verify MLFlow connection
mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")
mlflow.set_experiment("Test_Experiment")

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)
    print("MLFlow tracking is working!")


# In[193]:


# Code for Experiment 2:Create a pipeline for preprocessing and use LogisticRegression, RidgeClassifier, RandomForestClassifier
# Import required libraries
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import mlflow
import mlflow.sklearn

# Load Dataset
data = pd.read_csv("adult_income.csv")  # Replace with the actual dataset path
X = data.drop("income", axis=1)  # Replace "income" with your target column name
y = data["income"]

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Preprocess Features
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

numerical_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set up MLflow Tracking
mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")
experiment_name = "Experiment_No_XGBoost"
mlflow.set_experiment(experiment_name)

# Train and Log Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f"Running {name}...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    try:
        with mlflow.start_run(run_name=name):
            # Perform cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
            mlflow.log_metric("Mean CV F1 Score", cv_scores.mean())
            
            # Train the model
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Calculate and log metrics
            f1 = f1_score(y_test, y_pred, average="macro")
            mlflow.log_metric("Test F1 Score", f1)
            
            # Log confusion matrix and classification report
            conf_matrix = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix for {name}:\n", conf_matrix)
            print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))
            
            # Log the model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=name,
                signature=mlflow.models.infer_signature(X_test, y_pred)
            )
            
            print(f"{name} logged successfully with Mean CV F1 Score: {cv_scores.mean()} and Test F1 Score: {f1}")
    except Exception as e:
        print(f"Error during {name}: {str(e)}")
        if mlflow.active_run():
            mlflow.end_run()

# Ensure no active runs are left
if mlflow.active_run():
    mlflow.end_run()


# In[195]:


# Experiment-2, Create a pipeline for preprocessing and use XGBClassifier
# Import required libraries
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# Load Dataset
data = pd.read_csv("adult_income.csv")  # Replace with the actual dataset path

# Encode target variable
target_column = "income"  # Replace with your target column name
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Convert categorical columns to 'category' dtype
categorical_cols = data.select_dtypes(include=["object"]).columns
data[categorical_cols] = data[categorical_cols].astype("category")

# Split features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Set up MLflow Tracking
mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")
experiment_name = "Experiment_XGBoost"
mlflow.set_experiment(experiment_name)

# Train and Evaluate XGBoost Classifier
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    enable_categorical=True,
    random_state=42
)

try:
    with mlflow.start_run(run_name="XGBoost_Classifier"):
        # Train the model
        xgb_model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = xgb_model.predict(X_test)
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average="macro")
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("F1 Score", f1)
        
        # Print metrics
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", class_report)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=xgb_model,
            artifact_path="model",
            registered_model_name="XGBoost_Classifier"
        )
        print("XGBoost model logged successfully.")
except Exception as e:
    print("Error during training or logging:", e)
    if mlflow.active_run():
        mlflow.end_run()

# Ensure no active runs are left
if mlflow.active_run():
    mlflow.end_run()


# In[196]:


# Experiment 3  Perform feature engineering and attribute combination. Log results in MLFlow.
# Experiment 3: Logistic Regression and Random Forest
import os
import dagshub
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Initialize DagsHub MLFlow connection
dagshub.init(repo_owner="yashaswiniguntupalli", repo_name="ML_Final_Project", mlflow=True)

os.environ["MLFLOW_TRACKING_USERNAME"] = "yashaswiniguntupalli"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "dd928eb3e01ad92df47ae00f812f06a28ddc8c95"

mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")
mlflow.set_experiment("Experiment_3_LR_RF")

# Step 2: Load Dataset
file_path = "adult_income.csv"
data = pd.read_csv(file_path)

# Step 3: Feature Engineering
data['capital_diff'] = data['capital_gain'] - data['capital_loss']
data['age_income_ratio'] = data['age'] / (data['hours_per_week'] + 1)
data['hours_category'] = pd.cut(data['hours_per_week'], bins=[0, 20, 40, 60, 100], labels=["Low", "Medium", "High", "Very High"])

# Convert target variable to numerical labels
X = data.drop('income', axis=1)
y = LabelEncoder().fit_transform(data['income'])

# Convert integer columns to float
X = X.astype({col: "float" for col in X.select_dtypes(include="int64").columns})

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=["float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Step 4: Define Classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 6: Train and Log Models
for name, clf in classifiers.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    with mlflow.start_run(run_name=name):
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Calculate Metrics
            f1 = f1_score(y_test, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Log Metrics
            mlflow.log_metric("F1_Score", f1)

            # Log Model
            mlflow.sklearn.log_model(pipeline, name)

            # Plot Confusion Matrix
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix for {name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plot_path = f"{name}_confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
        except Exception as e:
            print(f"Error with {name}: {e}")
        finally:
            mlflow.end_run()


# In[198]:


#Experiment-3, Perform feature engineering and attribute combination. Log results in MLFlow. XGB
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# Step 1: Load Dataset
data = pd.read_csv("adult_income.csv")  # Replace with your dataset path

# Encode target variable
label_encoder = LabelEncoder()
data["income"] = label_encoder.fit_transform(data["income"])  # Replace "income" with your target column name

# Split features and target
X = data.drop("income", axis=1)
y = data["income"]

# Handle categorical data for XGBoost
categorical_cols = X.select_dtypes(include=["object"]).columns
X[categorical_cols] = X[categorical_cols].astype("category")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2: Define Basic XGBoost Model
xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    enable_categorical=True,
    random_state=42
)

# Step 3: Train and Evaluate Model
mlflow.set_experiment("Basic_XGBoost_Experiment")

with mlflow.start_run(run_name="XGBoost"):
    try:
        # Train the model
        xgb_clf.fit(X_train, y_train)

        # Predict
        y_pred = xgb_clf.predict(X_test)

        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average="weighted")
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("F1_Score", f1)

        print("F1 Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Log model
        mlflow.sklearn.log_model(xgb_clf, "XGBoost_Model")

    except Exception as e:
        print(f"Error with XGBoost: {e}")

    finally:
        if mlflow.active_run():
            mlflow.end_run()


# In[200]:


# code for experiment 4 : Perform feature selection using Correlation Threshold, Feature Importance, and Variance Threshold. Log results in MLFlow.
# Import required libraries
# Import required libraries
import os
import dagshub
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin

# Custom wrapper for XGBClassifier to include __sklearn_tags__
class CustomXGBClassifier(XGBClassifier, BaseEstimator, ClassifierMixin):
    def __sklearn_tags__(self):
        """Define custom tags for compatibility with scikit-learn >=1.7."""
        return {
            "binary_only": False,
            "multilabel": False,
            "multiclass": True,
            "poor_score": False,
            "no_validation": False,
        }

# Step 1: Initialize DagsHub MLFlow connection
dagshub.init(repo_owner="yashaswiniguntupalli", repo_name="ML_Final_Project", mlflow=True)

os.environ["MLFLOW_TRACKING_USERNAME"] = "yashaswiniguntupalli"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "dd928eb3e01ad92df47ae00f812f06a28ddc8c95"

mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")

experiment_name = "Experiment_4"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Step 2: Load Dataset
file_path = "adult_income.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Step 3: Preprocessing and Feature Engineering
X = data.drop('income', axis=1)
y = data['income']

# Convert target variable to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Step 4: Feature Selection Methods
def apply_correlation_threshold(X, threshold=0.9):
    """Remove features with high correlation."""
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    print(f"Features dropped due to correlation: {to_drop}")
    return X.drop(columns=to_drop)

def apply_variance_threshold(X, threshold=0.01):
    """Remove low variance features."""
    selector = VarianceThreshold(threshold)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()]
    print(f"Features dropped due to low variance: {set(X.columns) - set(selected_features)}")
    return pd.DataFrame(X_selected, columns=selected_features)

def apply_feature_importance(X, y, threshold=0.01):
    """Select features based on importance from RandomForestClassifier."""
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    selected_features = importances[importances > threshold].index
    print(f"Selected features based on importance: {selected_features}")
    return X[selected_features]

# Step 5: Apply Feature Selection
X_processed = pd.DataFrame(
    preprocessor.fit_transform(X).toarray(),
    columns=numerical_cols.tolist() + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out())
)

X_corr_filtered = apply_correlation_threshold(X_processed)  # Correlation Threshold
X_var_filtered = apply_variance_threshold(X_corr_filtered)  # Variance Threshold
X_importance_filtered = apply_feature_importance(X_var_filtered, y)  # Feature Importance

# Step 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_importance_filtered, y, test_size=0.3, random_state=42, stratify=y)

# Step 7: Define Classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBClassifier": CustomXGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

pipelines = {name: Pipeline(steps=[('classifier', clf)]) for name, clf in classifiers.items()}

# Step 8: Train and Log Models
for name, pipeline in pipelines.items():
    with mlflow.start_run(run_name=name):
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Calculate Metrics
            f1 = f1_score(y_test, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Log Metrics and Model
            mlflow.log_param("Model", name)
            mlflow.log_metric("F1_Score", f1)

            # Create a valid input example
            input_example = pd.DataFrame([X_test.iloc[0]], columns=X_test.columns)

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=name,
                input_example=input_example,
                signature=mlflow.models.infer_signature(X_test, y_pred),
            )

            # Plot Confusion Matrix
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.title(f"Confusion Matrix for {name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plot_path = f"{name}_confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

            print(f"\n{name} Results:")
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        except Exception as e:
            print(f"Error encountered with {name}: {e}")


# In[201]:


#CODE FOR EXPERIMENT 5:Use PCA for dimensionality reduction on all the features. Create a scree plot to show which components will be selected for classification. Log results in MLFlow
# Import required libraries
import os
import dagshub
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Initialize DagsHub MLFlow connection
dagshub.init(repo_owner="yashaswiniguntupalli", repo_name="ML_Final_Project", mlflow=True)

os.environ["MLFLOW_TRACKING_USERNAME"] = "yashaswiniguntupalli"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "dd928eb3e01ad92df47ae00f812f06a28ddc8c95"

mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")

experiment_name = "Experiment_5"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Step 2: Load Dataset
file_path = "adult_income.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Step 3: Preprocessing
X = data.drop('income', axis=1)
y = data['income']

# Convert target variable to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Define preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Step 4: PCA for Dimensionality Reduction
pca = PCA()
X_pca = pca.fit_transform(X_preprocessed.toarray())

# Create Scree Plot
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio.cumsum(), marker='o', linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot")
scree_plot_path = "scree_plot.png"
plt.savefig(scree_plot_path)
plt.show()

# Select components that explain at least 95% of the variance
n_components = np.argmax(explained_variance_ratio.cumsum() >= 0.95) + 1
X_pca_reduced = X_pca[:, :n_components]
print(f"Number of components selected: {n_components}")

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca_reduced, y, test_size=0.3, random_state=42, stratify=y)

# Step 6: Define Classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Step 7: Train and Log Models
for name, clf in classifiers.items():
    with mlflow.start_run(run_name=name):
        try:
            # Train the model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Calculate Metrics
            f1 = f1_score(y_test, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Log Metrics and Model
            mlflow.log_param("Model", name)
            mlflow.log_param("PCA_Components", n_components)
            mlflow.log_metric("F1_Score", f1)

            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path=name,
                input_example=X_test[:1],
                signature=mlflow.models.infer_signature(X_test, y_pred)
            )

            # Log Scree Plot
            mlflow.log_artifact(scree_plot_path)

            # Plot Confusion Matrix
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.title(f"Confusion Matrix for {name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plot_path = f"{name}_confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

            # Print Results
            print(f"\n{name} Results:")
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        except Exception as e:
            print(f"Error encountered with {name}: {e}")


# In[202]:


#code for experiment 6:Design and execute a custom experiment. Log results in MLFlow.
# Import required libraries
import os
import dagshub
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Initialize DagsHub MLFlow connection
dagshub.init(repo_owner="yashaswiniguntupalli", repo_name="ML_Final_Project", mlflow=True)

os.environ["MLFLOW_TRACKING_USERNAME"] = "yashaswiniguntupalli"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "dd928eb3e01ad92df47ae00f812f06a28ddc8c95"

mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")

experiment_name = "Experiment_6_Custom_Experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Step 2: Load Dataset
file_path = "adult_income.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Step 3: Preprocessing
X = data.drop('income', axis=1)
y = data['income']

# Convert target variable to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Define preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Step 4: PCA for Dimensionality Reduction
pca = PCA(n_components=0.95)  # Keep 95% of the variance
X_pca = pca.fit_transform(X_preprocessed.toarray())
print(f"Number of components retained: {pca.n_components_}")

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

# Step 6: Define Classifiers and kNN Parameter Grid
classifiers = {
    "kNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Parameter grid for kNN
knn_param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}

# Step 7: Train, Tune, and Log Models
for name, clf in classifiers.items():
    with mlflow.start_run(run_name=name):
        try:
            if name == "kNN":
                # Perform GridSearch for kNN
                grid_search = GridSearchCV(clf, knn_param_grid, cv=5, scoring="f1_weighted")
                grid_search.fit(X_train, y_train)
                clf = grid_search.best_estimator_  # Use the best model
                best_k = grid_search.best_params_["n_neighbors"]
                mlflow.log_param("Best_k", best_k)
            else:
                clf.fit(X_train, y_train)

            # Predict and Evaluate
            y_pred = clf.predict(X_test)

            # Calculate Metrics
            f1 = f1_score(y_test, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Log Metrics and Model
            mlflow.log_param("Model", name)
            mlflow.log_param("PCA_Components", pca.n_components_)
            mlflow.log_metric("F1_Score", f1)

            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path=name,
                input_example=X_test[:1],
                signature=mlflow.models.infer_signature(X_test, y_pred)
            )

            # Plot Confusion Matrix
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.title(f"Confusion Matrix for {name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plot_path = f"{name}_confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

            # Print Results
            print(f"\n{name} Results:")
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        except Exception as e:
            print(f"Error encountered with {name}: {e}")


# In[203]:


#importing libraries for experiment -7
get_ipython().system('pip uninstall scikit-learn imbalanced-learn sklearn-compat -y')


# In[204]:


#impoting libraries for experiment -7
get_ipython().system('pip install --upgrade scikit-learn imbalanced-learn')


# In[205]:


#checking libraries are installed properly for experiment -7 are not
import sklearn
import imblearn
print("Scikit-learn version:", sklearn.__version__)
print("Imbalanced-learn version:", imblearn.__version__)


# In[206]:


#code for experiment 7:Design and execute another custom experiment. Log results in MLFlow.
# Import required libraries
import os
import dagshub
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Initialize DagsHub MLFlow connection
dagshub.init(repo_owner="yashaswiniguntupalli", repo_name="ML_Final_Project", mlflow=True)

os.environ["MLFLOW_TRACKING_USERNAME"] = "yashaswiniguntupalli"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "dd928eb3e01ad92df47ae00f812f06a28ddc8c95"

mlflow.set_tracking_uri("https://dagshub.com/yashaswiniguntupalli/ML_Final_Project.mlflow")

experiment_name = "Experiment_7_SMOTE_Balancing"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Step 2: Load Dataset
file_path = "adult_income.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Step 3: Preprocessing
X = data.drop('income', axis=1)
y = data['income']

# Convert target variable to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(X)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42, stratify=y)

# Step 5: Apply SMOTE for Balancing
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Original dataset shape: {pd.Series(y_train).value_counts()}")
print(f"Balanced dataset shape: {pd.Series(y_train_balanced).value_counts()}")

# Step 6: Define Classifiers
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# Step 7: Train and Log Models
for name, clf in classifiers.items():
    with mlflow.start_run(run_name=f"{name}_SMOTE"):
        try:
            # Train the model on the SMOTE-balanced dataset
            clf.fit(X_train_balanced, y_train_balanced)
            y_pred = clf.predict(X_test)

            # Calculate Metrics
            f1 = f1_score(y_test, y_pred, average="weighted")
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Log Metrics and Model
            mlflow.log_param("Model", name)
            mlflow.log_param("Balancing", "SMOTE")
            mlflow.log_metric("F1_Score", f1)

            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path=name,
                input_example=X_test[:1].toarray(),
                signature=mlflow.models.infer_signature(X_test, y_pred)
            )

            # Plot Confusion Matrix
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.title(f"Confusion Matrix for {name} with SMOTE")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plot_path = f"{name}_SMOTE_confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

            # Print Results
            print(f"\n{name} with SMOTE Results:")
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        except Exception as e:
            print(f"Error encountered with {name}: {e}")


# In[207]:


#experiment-7:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example data: Replace this with your actual F1-score data
data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "KNN"],
    "Mean CV F1-Score": [0.78, 0.81, 0.85, 0.76, 0.74],
    "Test F1-Score": [0.79, 0.82, 0.86, 0.77, 0.75]
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to have a long-form format for better visualization
df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="F1-Score")

# Plot F1-Scores
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="F1-Score", hue="Metric", data=df_melted, palette="viridis")
plt.title("Comparison of F1-Scores Across Models", fontsize=16)
plt.ylabel("F1-Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Metric")
plt.tight_layout()

# Show the plot
plt.show()


# In[208]:


plt.savefig("f1_score_comparison.png", dpi=300)


# In[209]:


#Determining the best model
import matplotlib.pyplot as plt
import pandas as pd

# Example F1-score data for models
data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "KNN"],
    "Mean CV F1-Score": [0.78, 0.81, 0.85, 0.76, 0.74],
    "Test F1-Score": [0.79, 0.82, 0.86, 0.77, 0.75]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plot Mean CV F1-Score and Test F1-Score
plt.figure(figsize=(10, 6))
plt.bar(df["Model"], df["Mean CV F1-Score"], alpha=0.7, label="Mean CV F1-Score")
plt.bar(df["Model"], df["Test F1-Score"], alpha=0.7, label="Test F1-Score", width=0.5)
plt.ylabel("F1-Score")
plt.title("Comparison of F1-Scores Across Models")
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Determine the best model based on Test F1-Score
best_model_row = df.loc[df["Test F1-Score"].idxmax()]
print("Best Model:")
print(best_model_row)


# In[246]:


#saving the best model using joblib and saving the pikle file - weights
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv("adult_income.csv")  # Replace with the actual dataset file

# Define feature columns and target column
X = data.drop("income", axis=1)  # Replace 'income' with the actual target column name
y = data["income"]              # Replace 'income' with the actual target column name

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numeric and categorical columns
numeric_features = ['age', 'fnlwgt', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss']
categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(model_pipeline, 'best_model.pkl')


# In[ ]:


# Saved the best model pk file


# In[249]:


@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        print("Input Data for Prediction: ", data)  # Debugging
        # Make predictions
        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}


# In[250]:


@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Apply preprocessing if not part of the model
        # Example: encode categorical variables
        # data = preprocess_function(data)

        # Make prediction
        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print("Error during prediction: ", e)
        return {"error": str(e)}


# In[251]:


@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        data = pd.DataFrame([input_data.dict()])

        # Reorder columns to match the training data
        data = data[["age", "workclass", "fnlwgt", "education", "education_num",
                     "marital_status", "occupation", "relationship", "race",
                     "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]]

        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print("Error: ", e)
        return {"error": str(e)}


# In[252]:


# verifying the feature output 
@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        data = pd.DataFrame([input_data.dict()])

        # Reorder columns to match the training data
        data = data[["age", "workclass", "fnlwgt", "education", "education_num",
                     "marital_status", "occupation", "relationship", "race",
                     "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]]

        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print("Error: ", e)
        return {"error": str(e)}


# In[ ]:


# Required Libraries
# creating the fast API and running the model to serve the model and got the output sucessfully
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# File paths for saved model and label encoder
MODEL_PATH = "best_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Initialize FastAPI
app = FastAPI()

# Input Data Model
class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# Function to train the model
def train_model():
    # Load dataset
    data = pd.read_csv("adult_income.csv")  # Ensure you have the correct dataset in the same directory
    
    # Define feature columns and target
    feature_columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country"
    ]
    target_column = "income"
    
    X = data[feature_columns]
    y = data[target_column]

    # Label encode the target column
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Save label encoder
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # Preprocessing for numerical and categorical features
    numeric_features = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    numeric_transformer = StandardScaler()

    categorical_features = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Define the model pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())])

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(pipeline, MODEL_PATH)
    print("Model trained and saved successfully!")

# Uncomment this line to train the model and create the necessary files
train_model()

# Load the model and label encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Income Prediction API. Use /predict to make predictions."}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])

        # Make prediction
        prediction = model.predict(data)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return {"prediction": predicted_label}
    except Exception as e:
        return {"error": str(e)}

# For running in Jupyter Notebook
if __name__ == "__main__":
    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=8003)


# In[ ]:


# #code running it on anoother note book - got an output
# import requests

# # URL of the running FastAPI server
# url = "http://127.0.0.1:8003/predict"

# # Input data for prediction (modify values as needed)
# input_data = {
#     "age": 35,
#     "workclass": "Private",
#     "fnlwgt": 215646,
#     "education": "Bachelors",
#     "education_num": 13,
#     "marital_status": "Married-civ-spouse",
#     "occupation": "Exec-managerial",
#     "relationship": "Husband",
#     "race": "White",
#     "sex": "Male",
#     "capital_gain": 0,
#     "capital_loss": 0,
#     "hours_per_week": 40,
#     "native_country": "United-States"
# }

# # Send POST request
# response = requests.post(url, json=input_data)

# # Print the response
# print("Status Code:", response.status_code)
# print("Response Body:", response.json())



# In[ ]:




