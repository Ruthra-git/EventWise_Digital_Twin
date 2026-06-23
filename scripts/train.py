#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ============================================================
# EventWise AI / TrafficTwin
# 03_train.ipynb
#
# Cell 1 : Imports & Project Configuration
# ============================================================

import os
import sqlite3
import warnings
import joblib
import json
from datetime import datetime

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Machine Learning
# ------------------------------------------------------------

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# Gradient Boosting
# ------------------------------------------------------------

import lightgbm as lgb

from xgboost import XGBClassifier

# ------------------------------------------------------------
# Explainable AI
# ------------------------------------------------------------

import shap

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print

# ------------------------------------------------------------
# General Settings
# ------------------------------------------------------------

warnings.filterwarnings("ignore")

np.random.seed(42)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "database" / "brain.db"

# ------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------


MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)

os.makedirs(REPORT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Features expected from 02_cluster.py
# ------------------------------------------------------------

FEATURES = [

    "hour",

    "weekday",

    "month",

    "is_peak_hour",

    "is_weekend",

    "event_duration",

    "cause_code",

    "priority_score",

    "road_closure",

    "historical_frequency",

    "location_frequency",

    "cluster_id",

    "hotspot_density",

    "cluster_risk_score",

    "avg_cluster_duration",

    "planned_ratio",

    "recent_cluster_activity",

    "dist_from_center"

]

# ------------------------------------------------------------
# Notebook Header
# ------------------------------------------------------------

print("=" * 60)
print(" TrafficTwin : 03_train.ipynb")
print(" AI Intelligence Layer")
print("=" * 60)

print("\nDatabase :", DATABASE_PATH)

print("Model Dir :", MODEL_DIR)

print("Report Dir:", REPORT_DIR)

print("\nExpected Features :", len(FEATURES))

print(FEATURES)

print("\nInitialization Complete ✓")


# In[6]:


# ============================================================
# Cell 2 : Load brain.db & Dataset Validation
# ============================================================

print("\nLoading Event Database...\n")

# ------------------------------------------------------------
# Connect to SQLite
# ------------------------------------------------------------

conn = sqlite3.connect(DATABASE_PATH)

# ------------------------------------------------------------
# Load events table
# ------------------------------------------------------------

df = pd.read_sql(

    "SELECT * FROM events",

    conn

)

conn.close()

print("✓ events table loaded successfully")

# ------------------------------------------------------------
# Basic Dataset Information
# ------------------------------------------------------------

print("\n" + "="*60)

print("DATASET SUMMARY")

print("="*60)

print("Rows    :", len(df))

print("Columns :", len(df.columns))

print("\n")

# ------------------------------------------------------------
# Check required features
# ------------------------------------------------------------

print("="*60)

print("FEATURE VALIDATION")

print("="*60)

missing_features = []

for feature in FEATURES:

    if feature in df.columns:

        print(f"✓ {feature}")

    else:

        print(f"✗ {feature}")

        missing_features.append(feature)

# ------------------------------------------------------------
# Validation Result
# ------------------------------------------------------------

if len(missing_features)==0:

    print("\n✓ All required engineered features present.")

else:

    print("\nMissing Features:")

    print(missing_features)

# ------------------------------------------------------------
# Data Types
# ------------------------------------------------------------

print("\n")

print("="*60)

print("COLUMN TYPES")

print("="*60)

print(

    df[FEATURES].dtypes

)

# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------

print("\n")

print("="*60)

print("MISSING VALUES")

print("="*60)

missing = (

    df[FEATURES]

    .isnull()

    .sum()

)

print(

    missing

)

# ------------------------------------------------------------
# Duplicate Records
# ------------------------------------------------------------

duplicates = (

    df.duplicated()

    .sum()

)

print("\nDuplicate Rows :", duplicates)

# ------------------------------------------------------------
# Target Preview
# ------------------------------------------------------------

print("\n")

print("="*60)

print("AVAILABLE COLUMNS")

print("="*60)

print(

    sorted(df.columns.tolist())

)

# ------------------------------------------------------------
# Preview Dataset
# ------------------------------------------------------------

print("\n")

print("="*60)

print("FIRST FIVE ROWS")

print("="*60)

display(

    df.head()

)

print("\nDataset Loaded Successfully ✓")


# In[7]:


# ============================================================
# Cell 3 : Feature Quality Audit
# ============================================================

print("="*70)
print("FEATURE QUALITY AUDIT")
print("="*70)

# ------------------------------------------------------------
# Use severity as prediction target
# ------------------------------------------------------------

TARGET = "severity"

print("\nTarget Variable :", TARGET)

# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------

print("\n" + "="*70)
print("MISSING VALUES")
print("="*70)

missing = df[FEATURES].isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:

    print("✓ No missing values in selected features")

else:

    print(missing)

# ------------------------------------------------------------
# Constant Features
# ------------------------------------------------------------

print("\n" + "="*70)
print("CONSTANT FEATURES")
print("="*70)

constant_cols = []

for col in FEATURES:

    if df[col].nunique() <= 1:

        constant_cols.append(col)

if len(constant_cols) == 0:

    print("✓ No constant features detected")

else:

    print(constant_cols)

# ------------------------------------------------------------
# Low Variance Features
# ------------------------------------------------------------

print("\n" + "="*70)
print("LOW VARIANCE FEATURES")
print("="*70)

low_variance = []

for col in FEATURES:

    if pd.api.types.is_numeric_dtype(df[col]):

        if df[col].std() < 0.01:

            low_variance.append(col)

if len(low_variance) == 0:

    print("✓ No suspicious low variance features")

else:

    print(low_variance)

# ------------------------------------------------------------
# Target Distribution
# ------------------------------------------------------------

print("\n" + "="*70)
print("TARGET DISTRIBUTION")
print("="*70)

target_dist = (

    df[TARGET]

    .value_counts()

    .sort_index()

)

print(target_dist)

print("\nTarget Percentage (%)")

print(

    round(

        df[TARGET]

        .value_counts(normalize=True)

        .sort_index()*100,

        2

    )

)

# ------------------------------------------------------------
# Cluster Distribution
# ------------------------------------------------------------

print("\n" + "="*70)
print("CLUSTER DISTRIBUTION")
print("="*70)

cluster_dist = (

    df["cluster_id"]

    .value_counts()

)

largest_cluster = (

    cluster_dist.iloc[0]

    /

    len(df)

    *100

)

print("Clusters :", len(cluster_dist))

print("Largest Cluster :", round(largest_cluster,2), "%")

if largest_cluster > 50:

    print("⚠ WARNING : One cluster dominates dataset")

else:

    print("✓ Cluster distribution acceptable")

# ------------------------------------------------------------
# Highly Correlated Features
# ------------------------------------------------------------

print("\n" + "="*70)
print("HIGH CORRELATION CHECK")
print("="*70)

numeric_df = df[FEATURES].select_dtypes(include=np.number)

corr = numeric_df.corr().abs()

upper = corr.where(

    np.triu(

        np.ones(corr.shape),

        k=1

    ).astype(bool)

)

high_corr = []

for column in upper.columns:

    correlated = upper.index[upper[column] > 0.95].tolist()

    for c in correlated:

        high_corr.append((c,column))

if len(high_corr)==0:

    print("✓ No highly correlated features")

else:

    for pair in high_corr:

        print(pair)

# ------------------------------------------------------------
# Candidate Features to Drop
# ------------------------------------------------------------

print("\n" + "="*70)
print("SUGGESTED FEATURES TO DROP")
print("="*70)

drop_features = list(

    set(

        constant_cols +

        low_variance

    )

)

if len(drop_features)==0:

    print("✓ No features recommended for dropping")

else:

    print(drop_features)

# ------------------------------------------------------------
# Final Summary
# ------------------------------------------------------------

print("\n" + "="*70)
print("AUDIT SUMMARY")
print("="*70)

print("Total Features :", len(FEATURES))

print("Constant :", len(constant_cols))

print("Low Variance :", len(low_variance))

print("High Correlation :", len(high_corr))

print("Recommended Drop :", len(drop_features))

print("\n✓ Feature Quality Audit Complete")


# In[8]:


# ============================================================
# Cell 4 : Build Training Matrix
# ============================================================

print("="*70)
print("BUILDING TRAINING MATRIX")
print("="*70)

# ------------------------------------------------------------
# Final feature list
# ------------------------------------------------------------

FINAL_FEATURES = [

    "hour",

    "weekday",

    "month",

    "minutes_since_midnight",

    "is_peak_hour",

    "is_weekend",

    "is_night",

    "event_duration",

    "cause_code",

    "priority_score",

    "road_closure",

    "historical_frequency",

    "location_frequency",

    "zone_frequency",

    "junction_frequency",

    "police_frequency",

    "hotspot_density",

    "cluster_risk_score",

    "cluster_risk_tier",

    "avg_cluster_duration",

    "planned_ratio",

    "peak_hour_ratio",

    "night_ratio",

    "road_closure_ratio",

    "recent_cluster_activity",

    "dist_from_center"

]

# ------------------------------------------------------------
# Keep only existing columns
# ------------------------------------------------------------

FINAL_FEATURES = [

    col for col in FINAL_FEATURES

    if col in df.columns

]

print("\nFinal Feature Count :", len(FINAL_FEATURES))

# ------------------------------------------------------------
# Build X and y
# ------------------------------------------------------------

X = df[FINAL_FEATURES].copy()

y = df[TARGET].copy()

# ------------------------------------------------------------
# Fill numeric missing values
# ------------------------------------------------------------

for col in X.columns:

    if pd.api.types.is_numeric_dtype(X[col]):

        X[col] = X[col].fillna(

            X[col].median()

        )

# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

print("\nChecking missing values...")

missing_total = X.isnull().sum().sum()

if missing_total == 0:

    print("✓ No missing values remaining")

else:

    print("Remaining Missing :", missing_total)

# ------------------------------------------------------------
# Dataset shape
# ------------------------------------------------------------

print("\nFeature Matrix Shape")

print(X.shape)

print("\nTarget Shape")

print(y.shape)

# ------------------------------------------------------------
# Target balance
# ------------------------------------------------------------

print("\nTarget Distribution (%)")

target_percent = (

    y.value_counts(normalize=True)

    .sort_index()

    *100

).round(2)

print(target_percent)

# ------------------------------------------------------------
# Save feature list
# ------------------------------------------------------------

joblib.dump(

    FINAL_FEATURES,

    os.path.join(

        MODEL_DIR,

        "feature_columns.pkl"

    )

)

print("\n✓ feature_columns.pkl saved")

# ------------------------------------------------------------
# Preview
# ------------------------------------------------------------

print("\n")

print("="*70)

print("FIRST FIVE TRAINING ROWS")

print("="*70)

display(

    X.head()

)

print("\n")

print("Target Preview")

display(

    y.head()

)

print("\n✓ Training Matrix Ready")


# In[10]:


# ============================================================
# Cell 5 : Train/Test Split
# ============================================================

print("="*70)
print("CREATING TRAIN / TEST SPLIT")
print("="*70)

from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Stratified Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y,

    shuffle=True

)

print("\n✓ Stratified split completed")

# ------------------------------------------------------------
# Shapes
# ------------------------------------------------------------

print("\nTraining Shapes")

print("X_train :", X_train.shape)

print("y_train :", y_train.shape)

print("\nTesting Shapes")

print("X_test :", X_test.shape)

print("y_test :", y_test.shape)

# ------------------------------------------------------------
# Verify target balance
# ------------------------------------------------------------

print("\nOriginal Distribution (%)")

print(

    (

        y.value_counts(normalize=True)

        .sort_index()

        *100

    ).round(2)

)

print("\nTrain Distribution (%)")

print(

    (

        y_train.value_counts(normalize=True)

        .sort_index()

        *100

    ).round(2)

)

print("\nTest Distribution (%)")

print(

    (

        y_test.value_counts(normalize=True)

        .sort_index()

        *100

    ).round(2)

)

# ------------------------------------------------------------
# Missing value check
# ------------------------------------------------------------

print("\nChecking missing values...")

assert X_train.isnull().sum().sum() == 0

assert X_test.isnull().sum().sum() == 0

print("✓ No missing values")

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n")

print("="*70)

print("TRAIN / TEST SPLIT READY")

print("="*70)

print("Training Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

print("\n✓ LightGBM, Random Forest and XGBoost")
print("will all use this SAME split for fair comparison.")


# In[11]:


print("=" * 70)
print("REMOVING CONSTANT FEATURES")
print("=" * 70)

from sklearn.feature_selection import VarianceThreshold
import joblib
import os

# ------------------------------------------------------------
# Check feature variances
# ------------------------------------------------------------

feature_variance = X_train.var()

constant_features = feature_variance[
    feature_variance == 0
].index.tolist()

print("\nConstant Features Found :", len(constant_features))

if len(constant_features) > 0:

    print()

    for col in constant_features:
        print(col)

else:

    print("✓ No constant features detected")

# ------------------------------------------------------------
# Remove constant features
# ------------------------------------------------------------

selector = VarianceThreshold(
    threshold=0.0
)

selector.fit(X_train)

selected_columns = X_train.columns[
    selector.get_support()
]

X_train = X_train[selected_columns]

X_test = X_test[selected_columns]

# ------------------------------------------------------------
# Save updated feature list
# ------------------------------------------------------------

joblib.dump(

    list(selected_columns),

    os.path.join(

        MODEL_DIR,

        "feature_columns_after_variance.pkl"

    )

)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nOriginal Feature Count :", len(feature_variance))

print("Remaining Features     :", len(selected_columns))

print("Removed Features       :", len(feature_variance) - len(selected_columns))

print("\nTraining Shape")

print(X_train.shape)

print("\nTesting Shape")

print(X_test.shape)

print("\n✓ feature_columns_after_variance.pkl saved")

print("\n✓ Constant feature removal completed")


# In[12]:


# ============================================================
# Cell 7 : Correlation Analysis
# ============================================================

print("=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ------------------------------------------------------------
# Compute correlation matrix
# ------------------------------------------------------------

corr_matrix = X_train.corr(numeric_only=True)

print("\nCorrelation Matrix Shape")

print(corr_matrix.shape)

# ------------------------------------------------------------
# Find highly correlated feature pairs
# ------------------------------------------------------------

CORR_THRESHOLD = 0.95

upper_triangle = corr_matrix.where(

    np.triu(

        np.ones(corr_matrix.shape),

        k=1

    ).astype(bool)

)

high_corr_pairs = []

for col in upper_triangle.columns:

    for row in upper_triangle.index:

        value = upper_triangle.loc[row, col]

        if pd.notnull(value):

            if abs(value) >= CORR_THRESHOLD:

                high_corr_pairs.append(

                    (row, col, round(value, 3))

                )

print("\nHighly Correlated Pairs (>|0.95|)")

if len(high_corr_pairs) == 0:

    print("✓ None detected")

else:

    for f1, f2, corr in sorted(

        high_corr_pairs,

        key=lambda x: abs(x[2]),

        reverse=True

    ):

        print(f"{f1:30s} {f2:30s} {corr}")

# ------------------------------------------------------------
# Features suggested for removal
# ------------------------------------------------------------

to_drop = [

    col

    for col in upper_triangle.columns

    if any(

        upper_triangle[col].abs() >= CORR_THRESHOLD

    )

]

print("\nSuggested Features to Drop")

if len(to_drop) == 0:

    print("✓ None")

else:

    for feature in to_drop:

        print(feature)

# ------------------------------------------------------------
# Save correlation matrix
# ------------------------------------------------------------

corr_matrix.to_csv(

    os.path.join(

        MODEL_DIR,

        "correlation_matrix.csv"

    )

)

print("\n✓ correlation_matrix.csv saved")

# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

plt.figure(

    figsize=(12,10)

)

plt.imshow(

    corr_matrix,

    interpolation="nearest",

    aspect="auto"

)

plt.colorbar()

plt.xticks(

    range(len(corr_matrix.columns)),

    corr_matrix.columns,

    rotation=90,

    fontsize=8

)

plt.yticks(

    range(len(corr_matrix.columns)),

    corr_matrix.columns,

    fontsize=8

)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n")

print("=" * 70)

print("CORRELATION ANALYSIS COMPLETE")

print("=" * 70)

print(f"Total Features : {X_train.shape[1]}")

print(f"Highly Correlated Pairs : {len(high_corr_pairs)}")

print(f"Suggested Removals : {len(to_drop)}")


# In[13]:


print("=" * 70)
print("TARGET VALIDATION")
print("=" * 70)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# Unique classes
# ------------------------------------------------------------

classes = np.sort(y_train.unique())

print("\nClasses Detected")

print(classes)

# ------------------------------------------------------------
# Class counts
# ------------------------------------------------------------

print("\nTraining Class Distribution")

class_counts = y_train.value_counts().sort_index()

print(class_counts)

# ------------------------------------------------------------
# Class percentages
# ------------------------------------------------------------

print("\nTraining Class Distribution (%)")

class_percent = (

    y_train.value_counts(normalize=True)

    .sort_index()

    * 100

).round(2)

print(class_percent)

# ------------------------------------------------------------
# Compute balanced class weights
# ------------------------------------------------------------

weights = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=y_train

)

class_weights = {

    int(cls): float(weight)

    for cls, weight in zip(classes, weights)

}

print("\nComputed Class Weights")

for cls in class_weights:

    print(

        f"Class {cls} : {class_weights[cls]:.3f}"

    )

# ------------------------------------------------------------
# Save class weights
# ------------------------------------------------------------

joblib.dump(

    class_weights,

    os.path.join(

        MODEL_DIR,

        "class_weights.pkl"

    )

)

print("\n✓ class_weights.pkl saved")

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

assert len(classes) >= 2

assert y_train.isnull().sum() == 0

print("\n✓ Target validation passed")

print("\n")

print("=" * 70)

print("TARGET READY FOR MODEL TRAINING")

print("=" * 70)


# In[14]:


# ============================================================
# Cell 9 : LightGBM Training
# ============================================================

print("=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

import lightgbm as lgb
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# Create LightGBM model
# ------------------------------------------------------------

lgb_model = lgb.LGBMClassifier(

    objective="multiclass",

    num_class=len(np.unique(y_train)),

    n_estimators=300,

    learning_rate=0.05,

    max_depth=8,

    num_leaves=31,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1

)

# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

print("\nTraining model...\n")

lgb_model.fit(

    X_train,

    y_train

)

print("✓ Training completed")

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred = lgb_model.predict(X_test)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(

    y_test,

    y_pred

)

macro_f1 = f1_score(

    y_test,

    y_pred,

    average="macro"

)

weighted_f1 = f1_score(

    y_test,

    y_pred,

    average="weighted"

)

print("\nAccuracy")

print(round(accuracy, 4))

print("\nMacro F1")

print(round(macro_f1, 4))

print("\nWeighted F1")

print(round(weighted_f1, 4))

# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        y_pred,

        digits=4

    )

)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(

    y_test,

    y_pred

)

print("\nConfusion Matrix\n")

print(cm)

# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

joblib.dump(

    lgb_model,

    os.path.join(

        MODEL_DIR,

        "lightgbm_model.pkl"

    )

)

print("\n✓ lightgbm_model.pkl saved")

# ------------------------------------------------------------
# Store results
# ------------------------------------------------------------

lgb_results = {

    "accuracy": accuracy,

    "macro_f1": macro_f1,

    "weighted_f1": weighted_f1

}

print("\n")

print("=" * 70)

print("LIGHTGBM TRAINING COMPLETE")

print("=" * 70)


# In[20]:


# ============================================================
# Cell 10 : Model Validation & Leakage Check
# ============================================================

print("=" * 70)
print("MODEL VALIDATION & LEAKAGE CHECK")
print("=" * 70)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Training Accuracy
# ------------------------------------------------------------

train_pred = lgb_model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

print("\nTrain Accuracy")

print(round(train_accuracy, 4))

print("\nTest Accuracy")

print(round(accuracy, 4))

# ------------------------------------------------------------
# 2. Duplicate check
# ------------------------------------------------------------

duplicate_rows = X.duplicated().sum()

print("\nDuplicate Feature Rows")

print(duplicate_rows)

# ------------------------------------------------------------
# 3. Feature correlation with target
# ------------------------------------------------------------

temp = X.copy()

temp["severity_target"] = y

target_corr = (

    temp.corr(numeric_only=True)["severity_target"]

    .drop("severity_target")

    .sort_values(

        ascending=False

    )

)

print("\nTop 10 Feature Correlations with Target")

print(

    target_corr.head(10)

)

# ------------------------------------------------------------
# 4. Feature importance
# ------------------------------------------------------------

importance_df = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": lgb_model.feature_importances_

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 10 Feature Importances")

print(

    importance_df.head(10)

)

# ------------------------------------------------------------
# 5. 5-Fold Cross Validation
# ------------------------------------------------------------

print("\nRunning 5-Fold Cross Validation...")

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

cv_scores = cross_val_score(

    lgb_model,

    X,

    y,

    cv=cv,

    scoring="accuracy",

    n_jobs=-1

)

print("\nFold Accuracies")

for i, score in enumerate(cv_scores):

    print(

        f"Fold {i+1} : {score:.4f}"

    )

print("\nMean Accuracy")

print(round(cv_scores.mean(), 4))

print("\nStd Deviation")

print(round(cv_scores.std(), 6))

# ------------------------------------------------------------
# 6. Leakage warning
# ------------------------------------------------------------

print("\nLeakage Analysis")

high_corr = target_corr[

    target_corr.abs() > 0.90

]

if len(high_corr) == 0:

    print("✓ No feature has correlation > 0.90 with target.")

else:

    print("⚠ Highly correlated features detected:")

    print(high_corr)

# ------------------------------------------------------------
# 7. Final assessment
# ------------------------------------------------------------

print("\n")

print("=" * 70)

print("VALIDATION SUMMARY")

print("=" * 70)

print(f"Train Accuracy     : {train_accuracy:.4f}")

print(f"Test Accuracy      : {accuracy:.4f}")

print(f"CV Mean Accuracy   : {cv_scores.mean():.4f}")

print(f"CV Std Deviation   : {cv_scores.std():.6f}")

if abs(train_accuracy - cv_scores.mean()) < 0.02:

    print("\n✓ Model appears stable across folds.")

else:

    print("\n⚠ Large train-CV gap detected.")

if len(high_corr) == 0:

    print("✓ No obvious target leakage detected.")

else:

    print("⚠ Review highly correlated features carefully.")


# In[21]:


# ============================================================
# Cell 12 : Random Forest Training
# ============================================================

print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import os

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

rf_model = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced_subsample"

)

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\nTraining model...")

rf_model.fit(

    X_train,

    y_train

)

print("✓ Training completed")

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

rf_pred = rf_model.predict(

    X_test

)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

rf_accuracy = accuracy_score(

    y_test,

    rf_pred

)

rf_macro_f1 = f1_score(

    y_test,

    rf_pred,

    average="macro"

)

rf_weighted_f1 = f1_score(

    y_test,

    rf_pred,

    average="weighted"

)

print("\nAccuracy")

print(round(rf_accuracy,4))

print("\nMacro F1")

print(round(rf_macro_f1,4))

print("\nWeighted F1")

print(round(rf_weighted_f1,4))

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        rf_pred,

        digits=4

    )

)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        y_test,

        rf_pred

    )

)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

joblib.dump(

    rf_model,

    os.path.join(

        MODEL_DIR,

        "random_forest_model.pkl"

    )

)

print("\n✓ random_forest_model.pkl saved")

# ------------------------------------------------------------
# Store Results
# ------------------------------------------------------------

rf_results = {

    "accuracy": rf_accuracy,

    "macro_f1": rf_macro_f1,

    "weighted_f1": rf_weighted_f1

}

print("\n")

print("=" * 70)

print("RANDOM FOREST TRAINING COMPLETE")

print("=" * 70)


# In[23]:


# ============================================================
# Cell 13 : XGBoost Training
# ============================================================

print("=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib
import numpy as np
import os

# ------------------------------------------------------------
# Encode target labels
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y_train_xgb = label_encoder.fit_transform(
    y_train
)

y_test_xgb = label_encoder.transform(
    y_test
)

print("\nEncoded Classes")

print(label_encoder.classes_)

# ------------------------------------------------------------
# Build Model
# ------------------------------------------------------------

xgb_model = XGBClassifier(

    objective="multi:softprob",

    num_class=len(label_encoder.classes_),

    n_estimators=300,

    learning_rate=0.05,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1,

    eval_metric="mlogloss"

)

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print("\nTraining model...\n")

xgb_model.fit(

    X_train,

    y_train_xgb

)

print("✓ Training completed")

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

xgb_pred_encoded = xgb_model.predict(

    X_test

)

# ------------------------------------------------------------
# Decode predictions back to original labels
# ------------------------------------------------------------

xgb_pred = label_encoder.inverse_transform(

    xgb_pred_encoded.astype(int)

)

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

xgb_accuracy = accuracy_score(

    y_test,

    xgb_pred

)

xgb_macro_f1 = f1_score(

    y_test,

    xgb_pred,

    average="macro"

)

xgb_weighted_f1 = f1_score(

    y_test,

    xgb_pred,

    average="weighted"

)

print("\nAccuracy")

print(round(xgb_accuracy, 4))

print("\nMacro F1")

print(round(xgb_macro_f1, 4))

print("\nWeighted F1")

print(round(xgb_weighted_f1, 4))

# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        xgb_pred,

        digits=4

    )

)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        y_test,

        xgb_pred

    )

)

# ------------------------------------------------------------
# Save Model
# ------------------------------------------------------------

joblib.dump(

    xgb_model,

    os.path.join(

        MODEL_DIR,

        "xgboost_model.pkl"

    )

)

joblib.dump(

    label_encoder,

    os.path.join(

        MODEL_DIR,

        "xgb_label_encoder.pkl"

    )

)

print("\n✓ xgboost_model.pkl saved")

print("✓ xgb_label_encoder.pkl saved")

# ------------------------------------------------------------
# Store Results
# ------------------------------------------------------------

xgb_results = {

    "accuracy": xgb_accuracy,

    "macro_f1": xgb_macro_f1,

    "weighted_f1": xgb_weighted_f1

}

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n")

print("=" * 70)

print("XGBOOST TRAINING COMPLETE")

print("=" * 70)

print(f"Accuracy     : {xgb_accuracy:.4f}")

print(f"Macro F1     : {xgb_macro_f1:.4f}")

print(f"Weighted F1  : {xgb_weighted_f1:.4f}")


# In[24]:


# ============================================================
# Cell 14 : 5-Fold Cross Validation
# ============================================================

print("=" * 70)
print("5-FOLD STRATIFIED CROSS VALIDATION")
print("=" * 70)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Cross Validation Setup
# ------------------------------------------------------------

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

scoring = {

    "accuracy": "accuracy",

    "macro_f1": "f1_macro",

    "weighted_f1": "f1_weighted"

}

# ============================================================
# LightGBM
# ============================================================

print("\nEvaluating LightGBM...")

lgb_cv = cross_validate(

    lgb_model,

    X,

    y,

    cv=cv,

    scoring=scoring,

    n_jobs=-1,

    return_train_score=False

)

# ============================================================
# Random Forest
# ============================================================

print("Evaluating Random Forest...")

rf_cv = cross_validate(

    rf_model,

    X,

    y,

    cv=cv,

    scoring=scoring,

    n_jobs=-1,

    return_train_score=False

)

# ============================================================
# XGBoost
# ============================================================

print("Evaluating XGBoost...")

label_encoder = LabelEncoder()

y_xgb = label_encoder.fit_transform(y)

xgb_cv = cross_validate(

    xgb_model,

    X,

    y_xgb,

    cv=cv,

    scoring=scoring,

    n_jobs=-1,

    return_train_score=False

)

# ------------------------------------------------------------
# Build comparison table
# ------------------------------------------------------------

cv_results = pd.DataFrame({

    "Model": [

        "LightGBM",

        "Random Forest",

        "XGBoost"

    ],

    "Mean Accuracy": [

        lgb_cv["test_accuracy"].mean(),

        rf_cv["test_accuracy"].mean(),

        xgb_cv["test_accuracy"].mean()

    ],

    "Std Accuracy": [

        lgb_cv["test_accuracy"].std(),

        rf_cv["test_accuracy"].std(),

        xgb_cv["test_accuracy"].std()

    ],

    "Mean Macro F1": [

        lgb_cv["test_macro_f1"].mean(),

        rf_cv["test_macro_f1"].mean(),

        xgb_cv["test_macro_f1"].mean()

    ],

    "Mean Weighted F1": [

        lgb_cv["test_weighted_f1"].mean(),

        rf_cv["test_weighted_f1"].mean(),

        xgb_cv["test_weighted_f1"].mean()

    ]

})

# ------------------------------------------------------------
# Round values
# ------------------------------------------------------------

cv_results = cv_results.round(4)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print("\n")

print("=" * 70)

print("CROSS VALIDATION RESULTS")

print("=" * 70)

display(

    cv_results

)

# ------------------------------------------------------------
# Best Model
# ------------------------------------------------------------

best_model = cv_results.loc[

    cv_results["Mean Macro F1"].idxmax()

]

print("\nBest Model")

print(best_model["Model"])

print("\nBest Macro F1")

print(best_model["Mean Macro F1"])

# ------------------------------------------------------------
# Save for later cells
# ------------------------------------------------------------

cv_summary = cv_results.copy()

print("\n✓ Cross Validation Completed")

print("\n✓ cv_summary stored for Model Comparison Cell")


# In[25]:


# ============================================================
# Cell 15 : Model Comparison
# ============================================================

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

import pandas as pd
import joblib
import os

# ------------------------------------------------------------
# Ranking
# ------------------------------------------------------------

comparison_df = cv_summary.copy()

comparison_df = comparison_df.sort_values(

    by="Mean Macro F1",

    ascending=False

).reset_index(drop=True)

comparison_df.insert(

    0,

    "Rank",

    range(1, len(comparison_df)+1)

)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print("\nOverall Ranking\n")

display(comparison_df)

# ------------------------------------------------------------
# Best Model
# ------------------------------------------------------------

best_model_name = comparison_df.iloc[0]["Model"]

print("\nBest Model Selected")

print(best_model_name)

# ------------------------------------------------------------
# Save best model
# ------------------------------------------------------------

if best_model_name == "LightGBM":

    best_model = lgb_model

elif best_model_name == "Random Forest":

    best_model = rf_model

else:

    best_model = xgb_model

joblib.dump(

    best_model,

    os.path.join(

        MODEL_DIR,

        "best_model.pkl"

    )

)

print("\n✓ best_model.pkl saved")

# ------------------------------------------------------------
# Save comparison table
# ------------------------------------------------------------

comparison_df.to_csv(

    os.path.join(

        MODEL_DIR,

        "model_comparison.csv"

    ),

    index=False

)

print("✓ model_comparison.csv saved")

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n")

print("=" * 70)

print("MODEL SELECTION SUMMARY")

print("=" * 70)

print(f"Selected Model : {best_model_name}")

print(

    f"Mean Accuracy : "

    f"{comparison_df.iloc[0]['Mean Accuracy']:.4f}"

)

print(

    f"Mean Macro F1 : "

    f"{comparison_df.iloc[0]['Mean Macro F1']:.4f}"

)

print(

    f"Mean Weighted F1 : "

    f"{comparison_df.iloc[0]['Mean Weighted F1']:.4f}"

)

print("\nSelection Criterion : Highest Mean Macro F1")

# ------------------------------------------------------------
# Store for later cells
# ------------------------------------------------------------

selected_model_name = best_model_name
selected_model = best_model

print("\n✓ selected_model stored for SHAP analysis")


# In[27]:


# ============================================================
# Cell 16 : SHAP Explainability
# ============================================================

print("=" * 70)
print("SHAP EXPLAINABILITY")
print("=" * 70)

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# Create SHAP Explainer
# ------------------------------------------------------------

print("\nCreating SHAP Explainer...")

explainer = shap.TreeExplainer(

    selected_model

)

# ------------------------------------------------------------
# Use a subset for speed
# ------------------------------------------------------------

sample_size = min(

    1000,

    len(X_test)

)

X_sample = X_test.sample(

    n=sample_size,

    random_state=42

)

print(f"\nUsing {sample_size} samples")

# ------------------------------------------------------------
# Compute SHAP values
# ------------------------------------------------------------

print("\nComputing SHAP values...")

shap_values = explainer.shap_values(

    X_sample

)

print("✓ SHAP computation completed")

# ------------------------------------------------------------
# Handle multiclass output
# ------------------------------------------------------------

if isinstance(shap_values, list):

    shap_summary = np.mean(

        np.abs(

            np.stack(shap_values)

        ),

        axis=0

    )

else:

    shap_summary = np.abs(

        shap_values

    )

# ------------------------------------------------------------
# Mean SHAP importance
# ------------------------------------------------------------

# ------------------------------------------------------------
# Mean SHAP importance
# ------------------------------------------------------------

# SHAP Explanation object

if hasattr(shap_values, "values"):

    values = shap_values.values

else:

    values = shap_values

# multiclass

if len(values.shape) == 3:

    mean_shap = np.mean(

        np.abs(values),

        axis=(0,2)

    )

# binary/regression

else:

    mean_shap = np.mean(

        np.abs(values),

        axis=0

    )

importance = pd.DataFrame({

    "Feature": X_sample.columns,

    "Mean_SHAP": mean_shap

})

importance = importance.sort_values(

    by="Mean_SHAP",

    ascending=False

)

print("\nTop 15 SHAP Features\n")

display(

    importance.head(15)

)
# ------------------------------------------------------------
# Save SHAP importance
# ------------------------------------------------------------

importance.to_csv(

    os.path.join(

        MODEL_DIR,

        "shap_importance.csv"

    ),

    index=False

)

print("\n✓ shap_importance.csv saved")

# ------------------------------------------------------------
# SHAP Summary Plot
# ------------------------------------------------------------

print("\nGenerating Summary Plot...")

plt.figure(figsize=(10,8))

shap.summary_plot(

    shap_values,

    X_sample,

    show=False

)

plt.tight_layout()

plt.savefig(

    os.path.join(

        MODEL_DIR,

        "shap_summary.png"

    ),

    dpi=300

)

plt.show()

# ------------------------------------------------------------
# SHAP Bar Plot
# ------------------------------------------------------------

plt.figure(figsize=(10,8))

shap.summary_plot(

    shap_values,

    X_sample,

    plot_type="bar",

    show=False

)

plt.tight_layout()

plt.savefig(

    os.path.join(

        MODEL_DIR,

        "shap_bar.png"

    ),

    dpi=300

)

plt.show()

print("\n✓ shap_summary.png saved")

print("✓ shap_bar.png saved")

# ------------------------------------------------------------
# Store for next cell
# ------------------------------------------------------------

shap_importance = importance.copy()

print("\n")

print("=" * 70)

print("SHAP ANALYSIS COMPLETE")

print("=" * 70)


# In[28]:


# ============================================================
# Cell 17 : Feature Importance Analysis
# ============================================================

print("=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

import matplotlib.pyplot as plt
import pandas as pd
import os

# ------------------------------------------------------------
# LightGBM Importance
# ------------------------------------------------------------

lgb_imp = pd.DataFrame({

    "Feature": X_train.columns,

    "LightGBM": lgb_model.feature_importances_

})

# ------------------------------------------------------------
# Random Forest Importance
# ------------------------------------------------------------

rf_imp = pd.DataFrame({

    "Feature": X_train.columns,

    "RandomForest": rf_model.feature_importances_

})

# ------------------------------------------------------------
# XGBoost Importance
# ------------------------------------------------------------

xgb_imp = pd.DataFrame({

    "Feature": X_train.columns,

    "XGBoost": xgb_model.feature_importances_

})

# ------------------------------------------------------------
# Merge all
# ------------------------------------------------------------

importance_df = lgb_imp

importance_df = importance_df.merge(

    rf_imp,

    on="Feature"

)

importance_df = importance_df.merge(

    xgb_imp,

    on="Feature"

)

# ------------------------------------------------------------
# Merge SHAP if available
# ------------------------------------------------------------

if "shap_importance" in globals():

    shap_df = shap_importance.rename(

        columns={

            "Mean_SHAP": "SHAP"

        }

    )

    importance_df = importance_df.merge(

        shap_df,

        on="Feature",

        how="left"

    )

# ------------------------------------------------------------
# Sort by LightGBM importance
# ------------------------------------------------------------

importance_df = importance_df.sort_values(

    by="LightGBM",

    ascending=False

)

print("\nTop 15 Features\n")

display(

    importance_df.head(15)

)

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

importance_df.to_csv(

    os.path.join(

        MODEL_DIR,

        "feature_importance_summary.csv"

    ),

    index=False

)

print("\n✓ feature_importance_summary.csv saved")

# ------------------------------------------------------------
# Plot LightGBM
# ------------------------------------------------------------

plt.figure(figsize=(10,8))

top = importance_df.head(15)

plt.barh(

    top["Feature"],

    top["LightGBM"]

)

plt.gca().invert_yaxis()

plt.title("LightGBM Feature Importance")

plt.tight_layout()

plt.savefig(

    os.path.join(

        MODEL_DIR,

        "lightgbm_feature_importance.png"

    ),

    dpi=300

)

plt.show()

# ------------------------------------------------------------
# Plot Random Forest
# ------------------------------------------------------------

plt.figure(figsize=(10,8))

plt.barh(

    top["Feature"],

    top["RandomForest"]

)

plt.gca().invert_yaxis()

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(

    os.path.join(

        MODEL_DIR,

        "rf_feature_importance.png"

    ),

    dpi=300

)

plt.show()

# ------------------------------------------------------------
# Plot XGBoost
# ------------------------------------------------------------

plt.figure(figsize=(10,8))

plt.barh(

    top["Feature"],

    top["XGBoost"]

)

plt.gca().invert_yaxis()

plt.title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(

    os.path.join(

        MODEL_DIR,

        "xgb_feature_importance.png"

    ),

    dpi=300

)

plt.show()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n")

print("=" * 70)

print("FEATURE IMPORTANCE ANALYSIS COMPLETE")

print("=" * 70)

print("✓ LightGBM importance saved")

print("✓ Random Forest importance saved")

print("✓ XGBoost importance saved")

print("✓ Combined importance table saved")

# ------------------------------------------------------------
# Store for report generation
# ------------------------------------------------------------

final_feature_importance = importance_df.copy()


# In[29]:


# ============================================================
# Cell 18 : Save All Model Artifacts
# ============================================================

print("=" * 70)
print("SAVING MODEL ARTIFACTS")
print("=" * 70)

import os
import json
import joblib
from datetime import datetime

# ------------------------------------------------------------
# Artifact dictionary
# ------------------------------------------------------------

artifacts = {

    "best_model.pkl": selected_model,

    "lightgbm_model.pkl": lgb_model,

    "random_forest_model.pkl": rf_model,

    "xgboost_model.pkl": xgb_model,

    "feature_columns.pkl": FINAL_FEATURES,

    "class_weights.pkl": class_weights,

    "xgb_label_encoder.pkl": label_encoder

}

saved_files = []

# ------------------------------------------------------------
# Save artifacts
# ------------------------------------------------------------

for filename, obj in artifacts.items():

    path = os.path.join(

        MODEL_DIR,

        filename

    )

    joblib.dump(

        obj,

        path

    )

    saved_files.append(filename)

# ------------------------------------------------------------
# Create manifest
# ------------------------------------------------------------

manifest = {

    "created_at":

        datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        ),

    "selected_model":

        selected_model_name,

    "feature_count":

        len(FINAL_FEATURES),

    "training_samples":

        len(X_train),

    "testing_samples":

        len(X_test),

    "saved_files":

        saved_files

}

with open(

    os.path.join(

        MODEL_DIR,

        "artifact_manifest.json"

    ),

    "w"

) as f:

    json.dump(

        manifest,

        f,

        indent=4

    )

print("\nSaved Files\n")

for file in saved_files:

    print("✓", file)

print("\n✓ artifact_manifest.json saved")

print("\n")

print("=" * 70)

print("ARTIFACT STORAGE COMPLETE")

print("=" * 70)


# In[32]:

import sqlite3


conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(model_metadata)")

for row in cursor.fetchall():

    print(row)


# In[34]:


# ============================================================
# Cell 19 : Update model_metadata
# ============================================================

print("=" * 70)
print("UPDATING MODEL METADATA")
print("=" * 70)

import sqlite3
import json
from datetime import datetime

conn = sqlite3.connect(DATABASE_PATH)

cursor = conn.cursor()

# ------------------------------------------------------------
# Prepare values
# ------------------------------------------------------------

model_name = selected_model_name

model_version = "v1.0"

training_date = datetime.now().strftime(
    "%Y-%m-%d"
)

features_used = ",".join(FINAL_FEATURES)

n_features = len(FINAL_FEATURES)

weighted_f1 = float(
    comparison_df.iloc[0]["Mean Weighted F1"]
)

accuracy = float(
    comparison_df.iloc[0]["Mean Accuracy"]
)

best_params = json.dumps({

    "algorithm": model_name,

    "selected_by": "Highest Mean Macro F1"

})

training_rows = len(X_train)

notes = (

    "TrafficTwin severity prediction model. "

    "Validated using 5-fold stratified cross validation."

)

# ------------------------------------------------------------
# Insert metadata
# ------------------------------------------------------------

cursor.execute("""

INSERT INTO model_metadata(

model_name,

model_version,

training_date,

features_used,

n_features,

weighted_f1,

accuracy,

best_params,

training_rows,

notes

)

VALUES(

?,?,?,?,?,?,?,?,?,?

)

""",

(

model_name,

model_version,

training_date,

features_used,

n_features,

weighted_f1,

accuracy,

best_params,

training_rows,

notes

)

)

conn.commit()

# ------------------------------------------------------------
# Verify latest record
# ------------------------------------------------------------

latest = cursor.execute("""

SELECT

model_name,

model_version,

accuracy,

weighted_f1,

n_features,

training_rows,

created_at

FROM model_metadata

ORDER BY id DESC

LIMIT 1

""").fetchone()

print("\nLatest Metadata Entry\n")

print(latest)

conn.close()

print("\n")

print("=" * 70)

print("MODEL METADATA UPDATED")

print("=" * 70)


# In[35]:


# ============================================================
# Cell 20 : Generate Training Report
# ============================================================

print("=" * 70)
print("GENERATING TRAINING REPORT")
print("=" * 70)

import os
from datetime import datetime

report_path = os.path.join(

    MODEL_DIR,

    "training_report.txt"

)

# ------------------------------------------------------------
# Report Content
# ------------------------------------------------------------

report = []

report.append("=" * 70)
report.append("TRAFFICTWIN TRAINING REPORT")
report.append("=" * 70)

report.append("")
report.append(
    f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

report.append("")

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

report.append("-" * 70)
report.append("DATASET SUMMARY")
report.append("-" * 70)

report.append(f"Total Samples        : {len(X)}")
report.append(f"Training Samples     : {len(X_train)}")
report.append(f"Testing Samples      : {len(X_test)}")
report.append(f"Total Features       : {len(FINAL_FEATURES)}")

report.append("")

# ------------------------------------------------------------
# Target Distribution
# ------------------------------------------------------------

report.append("-" * 70)
report.append("TARGET DISTRIBUTION")
report.append("-" * 70)

target_dist = y.value_counts().sort_index()

for cls, count in target_dist.items():

    report.append(

        f"Severity {cls} : {count}"

    )

report.append("")

# ------------------------------------------------------------
# Cross Validation
# ------------------------------------------------------------

report.append("-" * 70)
report.append("CROSS VALIDATION RESULTS")
report.append("-" * 70)

for _, row in comparison_df.iterrows():

    report.append(

        f"{row['Model']:15s}"

        f" Accuracy={row['Mean Accuracy']:.4f}"

        f" MacroF1={row['Mean Macro F1']:.4f}"

        f" WeightedF1={row['Mean Weighted F1']:.4f}"

    )

report.append("")

# ------------------------------------------------------------
# Selected Model
# ------------------------------------------------------------

report.append("-" * 70)
report.append("SELECTED MODEL")
report.append("-" * 70)

report.append(

    f"Model Name      : {selected_model_name}"

)

report.append(

    f"Selection Basis : Highest Mean Macro F1"

)

report.append(

    f"Accuracy        : {comparison_df.iloc[0]['Mean Accuracy']:.4f}"

)

report.append(

    f"Macro F1        : {comparison_df.iloc[0]['Mean Macro F1']:.4f}"

)

report.append(

    f"Weighted F1     : {comparison_df.iloc[0]['Mean Weighted F1']:.4f}"

)

report.append("")

# ------------------------------------------------------------
# Top Features
# ------------------------------------------------------------

report.append("-" * 70)
report.append("TOP FEATURES")
report.append("-" * 70)

top_features = final_feature_importance.head(15)

for _, row in top_features.iterrows():

    report.append(

        f"{row['Feature']}"

    )

report.append("")

# ------------------------------------------------------------
# Saved Artifacts
# ------------------------------------------------------------

report.append("-" * 70)
report.append("GENERATED ARTIFACTS")
report.append("-" * 70)

artifact_list = [

    "best_model.pkl",

    "lightgbm_model.pkl",

    "random_forest_model.pkl",

    "xgboost_model.pkl",

    "feature_columns.pkl",

    "xgb_label_encoder.pkl",

    "artifact_manifest.json",

    "model_comparison.csv",

    "feature_importance_summary.csv",

    "training_report.txt"

]

for file in artifact_list:

    report.append(file)

report.append("")

# ------------------------------------------------------------
# Deployment Notes
# ------------------------------------------------------------

report.append("-" * 70)
report.append("DEPLOYMENT NOTES")
report.append("-" * 70)

report.append(
    "• Selected model should be loaded from best_model.pkl"
)

report.append(
    "• Input features must match feature_columns.pkl exactly"
)

report.append(
    "• XGBoost predictions require xgb_label_encoder.pkl"
)

report.append(
    "• Feature preprocessing must be identical to training"
)

report.append(
    "• Model validated using 5-fold stratified cross-validation"
)

report.append(
    "• LightGBM selected as final deployment model"
)

report.append("")

report.append("=" * 70)
report.append("END OF REPORT")
report.append("=" * 70)

# ------------------------------------------------------------
# Write Report
# ------------------------------------------------------------

with open(

    report_path,

    "w",

    encoding="utf-8"

) as f:

    f.write(

        "\n".join(report)

    )

print("\n✓ training_report.txt generated")

print("\nLocation")

print(report_path)

print("\n")

print("=" * 70)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal Selected Model")

print(selected_model_name)
print("\nMean Accuracy")
print(f"{comparison_df.iloc[0]['Mean Accuracy']:.4f}")
print("\nMean Macro F1")
print(f"{comparison_df.iloc[0]['Mean Macro F1']:.4f}")
print("\nMean Weighted F1")
print(f"{comparison_df.iloc[0]['Mean Weighted F1']:.4f}")
print("\n✓ TrafficTwin training pipeline completed.")

