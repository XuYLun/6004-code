# We will use the following packages

# NumPy for math operations, and Pandas for processing tabular data.
import numpy as np
import pandas as pd

# Plotly plotting package
import plotly.graph_objects as go
import plotly.express as px

# We use toy datasets in scikit-learn package
from sklearn.datasets import load_breast_cancer

# Tools in sklearn to select best model
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV

# Decision tree classifier in sklearn
from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree

# We use f1 score to test model performance
from sklearn.metrics import f1_score

# Import matplotlib.pyplot to visualize tree models
import matplotlib.pyplot as plt
# Load your CSV file
data = pd.read_csv('6004data.csv')

# 抽取其中5000个样本
data = data.sample(n=10000, random_state=42)

# Preprocessing steps (you may adjust these based on your dataset)
data['aki'] = data['aki'].apply(lambda x: 0 if (x == 0) else 1)
data['gender'] = data['gender'].replace({'F': 1, 'M': 0}).astype(int)
data = data.drop(['race', 'id'], axis=1)
data = data.fillna(data.mean())

# Extract features (X) and target (y)
X_raw = data.drop(columns=['aki'])  # Replace 'target_column_name' with the name of your target column
y_df = data['aki']  # Replace 'target_column_name' with the name of your target column

print(y_df.value_counts())
# Standardize the features
X_df = (X_raw - X_raw.mean()) / X_raw.std()

X_df = X_df[['admission_age','sbp_min','resp_rate_max' ,'bun_min' ,'gcs_verbal',
 'weight_admit']]

print(X_df.columns)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_df,y_df,test_size=0.3,random_state=10,
    stratify=y_df, shuffle=True
)

# We first build a shallow decision tree.
TreeModel = DTC(criterion='entropy',max_depth=1,random_state=15)
TreeModel.fit(X_train,y_train)

# Splitting rules can be visualized by using plot_tree in sklearn
plt.figure(figsize=(14,10))
plot_tree(
    TreeModel,
    filled=True,
    feature_names=X_df.columns,  # Updated to use the actual feature names
    class_names=['not ill','aid']
)
plt.show()

# The `max_depth` parameter is important for decision tree.
# We use `GridSearchCV` to select the best `max_depth`.

parameters = {'max_depth':np.arange(start=1,stop=10,step=1)}
print(parameters)

stratifiedCV = StratifiedKFold(n_splits=8)
TreeModel = DTC(criterion='entropy')
BestTree = GridSearchCV(
    TreeModel,
    param_grid=parameters,
    scoring='f1',
    cv=stratifiedCV
)
BestTree.fit(X_train,y_train)

print(BestTree.best_estimator_)

print(BestTree.best_score_)

y_pred = BestTree.predict(X_test)
print('F1 score on test set: {:.4f}'.format(f1_score(y_test,y_pred)))
pd.crosstab(y_test,y_pred)

from xgboost import XGBClassifier as XGBC

parameters = {
    'n_estimators':np.arange(start=2,stop=20,step=2),
    'max_depth':np.arange(start=2,stop=6,step=1),
    'learning_rate':np.arange(start=0.05,stop=0.4,step=0.05)
}

print(parameters)

stratifiedCV = StratifiedKFold(n_splits=8)
# XGBC: XGBoost classifier
XGBoostModel = XGBC()
BestXGBoost = GridSearchCV(
    XGBoostModel,
    param_grid=parameters,
    scoring='f1',
    cv=stratifiedCV,
    verbose=1,
    n_jobs=-1 # use all cpu cores to speedup grid search
)
BestXGBoost.fit(X_train,y_train)

print(BestXGBoost.best_params_)

print(BestXGBoost.best_score_)

y_pred = BestXGBoost.predict(X_test)
print('F1 score on test set: {:.4f}'.format(f1_score(y_test,y_pred)))
pd.crosstab(y_test,y_pred)

# Support Vector Classifier
from sklearn.svm import SVC

# 'C': strength of L2 regularization on linear SVM. Larger 'C' --> smaller regularization.
parameters = {
    'C':np.arange(start=1,stop=20,step=5)
}
stratifiedCV = StratifiedKFold(n_splits=8)
SVCModel = SVC(kernel='linear')
BestSVC = GridSearchCV(
    SVCModel,
    param_grid=parameters,
    scoring='f1',
    cv=stratifiedCV,
    verbose=1,
    n_jobs=-1
)
BestSVC.fit(X_train,y_train)

print(BestSVC.best_estimator_)

print(BestSVC.best_score_)

y_pred = BestSVC.predict(X_test)
print('F1 score on test set: {:.4f}'.format(f1_score(y_test,y_pred)))
pd.crosstab(y_test,y_pred)


nonlinear_models = {
    'DecisionTree':DTC(criterion='entropy'),
    'XGBoost':XGBC(),
    'SVM_rbf':SVC(kernel='rbf')
}

stratifiedCV = StratifiedKFold(n_splits=8)


params = {
    'DecisionTree':{
        'max_depth':np.arange(start=1,stop=10)
    },
    'XGBoost':{
        'n_estimators':np.arange(start=2,stop=20,step=2),
        'max_depth':np.arange(start=2,stop=6),
        'learning_rate':np.arange(start=0.05,stop=0.4,step=0.05)
    },
    'SVM_rbf':{
        'C':np.arange(0.5,5,step=0.5)
    }
}

records = {}

for model in nonlinear_models:
    BestParams = GridSearchCV(
        nonlinear_models[model],
        param_grid = params[model],
        scoring='f1',
        cv=stratifiedCV,
        n_jobs=-1
    )
    BestParams.fit(X_train,y_train)
    records[model] = BestParams
    print('For {} cross validation F1 score is {:.4f}'.format(model,BestParams.best_score_))

from sklearn.metrics import accuracy_score

# Decision Tree
TreeModel.fit(X_train, y_train)
y_pred_tree = TreeModel.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Accuracy:", accuracy_tree)

# XGBoost
BestXGBoost.fit(X_train, y_train)
y_pred_xgboost = BestXGBoost.predict(X_test)
accuracy_xgboost = accuracy_score(y_test, y_pred_xgboost)
print("XGBoost Accuracy:", accuracy_xgboost)

# Support Vector Classifier
BestSVC.fit(X_train, y_train)
y_pred_svc = BestSVC.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("SVC Accuracy:", accuracy_svc)

from sklearn.metrics import roc_auc_score

# Decision Tree
y_prob_tree = TreeModel.predict_proba(X_test)[:, 1]
auc_tree = roc_auc_score(y_test, y_prob_tree)
print("Decision Tree AUC:", auc_tree)

# XGBoost
y_prob_xgboost = BestXGBoost.predict_proba(X_test)[:, 1]
auc_xgboost = roc_auc_score(y_test, y_prob_xgboost)
print("XGBoost AUC:", auc_xgboost)

# Support Vector Classifier
y_prob_svc = BestSVC.decision_function(X_test)
auc_svc = roc_auc_score(y_test, y_prob_svc)
print("SVC AUC:", auc_svc)

