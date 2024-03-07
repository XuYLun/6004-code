import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
# We use AUROC and average precision (AP) scores from sklearn
from sklearn.metrics import roc_auc_score, average_precision_score

# Plotly plotting package
import plotly.graph_objects as go
import plotly.express as px
from torch import logit

# Load your CSV file
data = pd.read_csv('6004data.csv')

# 抽取其中5000个样本
#data = data.sample(n=5000, random_state=42)

# Preprocessing steps (you may adjust these based on your dataset)
data['aki'] = data['aki'].apply(lambda x: 0 if (x == 0) else 1)
data['gender'] = data['gender'].replace({'F': 1, 'M': 0}).astype(int)
data = data.drop(['race', 'id'], axis=1)
data = data.fillna(data.mean())

# Extract features (X) and target (y)
X_raw = data.drop(columns=['aki'])  # Replace 'target_column_name' with the name of your target column
y_df = data['aki']  # Replace 'target_column_name' with the name of your target column

# Standardize the features
X_df = (X_raw - X_raw.mean()) / X_raw.std()
print(X_df.describe())

# We convert dataframe to PyTorch tensor datatype,
# and then split it into training and testing parts.
X = torch.tensor(X_df.to_numpy(),dtype=torch.float32)
m,n = X.shape
y = torch.tensor(y_df.to_numpy(),dtype=torch.float32).reshape(m,1)

# We use an approx 6:4 train test splitting
cases = ['train','test']
case_list = np.random.choice(cases,size=X.shape[0],replace=True,p=[0.6,0.4])
X_train = X[case_list=='train']
X_test = X[case_list=='test']
y_train = y[case_list=='train']
y_test = y[case_list=='test']


h_L1 = torch.nn.Linear(
    in_features=n,
    out_features=1,
    bias=True
)
sigma = torch.nn.Sigmoid()

# Logistic model is linear+sigmoid
f_L1 = torch.nn.Sequential(
    h_L1,
    sigma
)

J_BCE = torch.nn.BCELoss()

GD_optimizer = torch.optim.Adam(lr=0.01,params=f_L1.parameters())

# Define L_1 regularization
def L1_reg(model,lbd):
    result = torch.tensor(0)
    for param in model.parameters(): # iterate over all parameters of our model
        result = result + param.abs().sum()

    return lbd*result


nIter = 1000
printInterval = 50
lbd = 0.03 # L1 reg strength

for i in range(nIter):
    GD_optimizer.zero_grad()
    pred = f_L1(X_train)
    loss = J_BCE(pred,y_train)
    (loss+L1_reg(f_L1,lbd)).backward()
    GD_optimizer.step()
    if i == 0 or ((i+1)%printInterval) == 0:
        print('Iter {}: average BCE loss is {:.3f}'.format(i+1,loss.item()))

with torch.no_grad():
    pred_test = f_L1(X_test)

auroc = roc_auc_score(y_test,pred_test)
ap = average_precision_score(y_test,pred_test)
print('On test dataset: AUROC {:.3f}, AP {:.3f}'.format(auroc,ap))

weight_L1 = h_L1.weight.detach().squeeze().clone()

# 获取最后训练完成的模型的权重
final_weights = h_L1.weight.detach().numpy()

# 保留绝对值大于0.05的权重对应的特征
selected_features = X_df.columns[np.abs(final_weights.squeeze()) > 0.03]

from sklearn.linear_model import LogisticRegression as logit # use build-in logistic regression model in sklearn
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.feature_selection import SequentialFeatureSelector as SFS

# sklearn support fitting model on pandas dataframe.
# We use same train test split as before

X_df_train = X_df.iloc[case_list=='train',:]
X_df_test = X_df.iloc[case_list=='test',:]
y_df_train = y_df.iloc[case_list=='train']
y_df_test = y_df.iloc[case_list=='test']

model = logit(penalty='l1',C=1/10,solver='liblinear') # c: 1/(strength of L1 regularization)

# Forward feature selection.
forward_selection = SFS(
    model, n_features_to_select=6, direction="forward"
).fit(X_df_train[selected_features], y_df_train)

# Backward feature selection.
backward_selection = SFS(
    model, n_features_to_select=6, direction="backward"
).fit(X_df_train[selected_features], y_df_train)

print(forward_selection.get_feature_names_out())
print(backward_selection.get_feature_names_out())

