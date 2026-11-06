
import numpy as np
import pandas as pd
#sklearn preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#Classification model
from sklearn.linear_model import LogisticRegression
#metrices
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('loan_dataset.csv')
print("Shape of dataset:", df.shape)
df.head()

#target labelencoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Loan_Status']=le.fit_transform(df['Loan_Status'])

df.drop( columns = ['Loan_ID'], inplace=True )

X = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']

numeric_features = X.select_dtypes(include = ['int64','float64']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

outlier_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
print("Outlier count (IQR method):")
for c in outlier_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"{c:20s}: {n_out}")

# Feature engineering
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']   # monthly repayment burden


for c in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income']:
    df[c] = np.log1p(df[c])

X = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']

numeric_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(exclude='number').columns.tolist()
print("Numeric    :", numeric_features)
print("Categorical:", categorical_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#for numerical features

num_transformer = Pipeline (
    steps = [
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)
#for cat
cat_transformer = Pipeline( steps = [
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
] )


#combine them
preprocessor = ColumnTransformer(
    transformers= [
        ('num',num_transformer,numeric_features),
        ('cat',cat_transformer,categorical_features)
    ]
    )

# Full ML pipeline = preprocessing + model
final_pipeline = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]
)
final_pipeline

final_pipeline.fit(X_train,y_train)
print("Model trained successfully on the training set.")
print("Training accuracy:", round(final_pipeline.score(X_train, y_train), 3))

from sklearn.model_selection import cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
print("Fold scores :", cv_scores.round(3))
print("Mean accuracy:", round(cv_scores.mean(), 3))
print("Std deviation:", round(cv_scores.std(), 3))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__class_weight': [None, 'balanced'],
    'model__solver': ['liblinear']
}


grid_search = GridSearchCV(final_pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest parameters :", grid_search.best_params_)
print("Best CV accuracy:", round(grid_search.best_score_, 3))


best_model = grid_search.best_estimator_

print("Baseline CV accuracy:", round(cv_scores.mean(), 3))
print("Tuned    CV accuracy:", round(grid_search.best_score_, 3))

if grid_search.best_score_ >= cv_scores.mean():
    print("\nTuned model selected as the final model.")
else:
    print("\nBaseline performs better; keeping default settings.")
    best_model = final_pipeline

print("Final model:", best_model.named_steps['model'])

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("F1-score :", round(f1_score(y_test, y_pred), 3))
print("ROC-AUC  :", round(roc_auc_score(y_test, y_prob), 3))


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Approved (1)']))
cm = confusion_matrix(y_test, y_pred)

import pickle


with open("loan_pipeline.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Best model saved as loan_pipeline.pkl")