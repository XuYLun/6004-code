import pandas as pd
import torch
import numpy as np
import plotly.express as px


# Load your CSV file
data = pd.read_csv('6004data.csv')

# Preprocessing steps (you may adjust these based on your dataset)
data['aki'] = data['aki'].apply(lambda x: 0 if (x == 0 ) else 1)
data['gender'] = data['gender'].replace({'F': 1, 'M': 0}).astype(int)
data = data.drop(['race', 'id'], axis=1)
data = data.fillna(data.mean())

# Extract features (X) and target (y)
X_raw = data.drop(columns=['aki'])  # Replace 'target_column_name' with the name of your target column
y_df = data['aki']  # Replace 'target_column_name' with the name of your target column

# Output information about X_raw
print(X_raw.info())

# Standardize the features
X_df = (X_raw - X_raw.mean()) / X_raw.std()
print(X_df.describe())

X_df = X_df[['admission_age', 'sbp_min', 'resp_rate_max', 'bun_min', 'gcs_verbal', 'weight_admit']]

# Convert dataframe to PyTorch tensor datatype,
# and then split it into training and testing parts.
X = torch.tensor(X_df.to_numpy(), dtype=torch.float32)
m, n = X.shape
y = torch.tensor(y_df.to_numpy(), dtype=torch.float32).reshape(m, 1)

# We use an approx 6:4 train test splitting
cases = ['train', 'test']
case_list = np.random.choice(cases, size=X.shape[0], replace=True, p=[0.6, 0.4])
X_train = X[case_list == 'train']
X_test = X[case_list == 'test']
y_train = y[case_list == 'train']
y_test = y[case_list == 'test']

h = torch.nn.Linear(
    in_features=n,
    out_features=1,
    bias=True
)
sigma = torch.nn.Sigmoid()

# Logistic model is linear+sigmoid
f = torch.nn.Sequential(
    h,
    sigma
)

J_BCE = torch.nn.BCELoss()
GD_optimizer = torch.optim.SGD(params=f.parameters(), lr=0.001)

nIter = 10000
printInterval = 1000

for i in range(nIter):
    GD_optimizer.zero_grad()
    pred = f(X_train)
    loss = J_BCE(pred, y_train)
    loss.backward()
    GD_optimizer.step()
    if i == 0 or ((i + 1) % printInterval) == 0:
        print('Iter {}: average BCE loss is {:.3f}'.format(i + 1, loss.item()))

# Test on test data

threshold = 0.5

with torch.no_grad():
    pred_test = f(X_test)

# The output has shape M*1. Use .squeeze to remove the trailing dimension with size 1.
binary_pred = np.where(pred_test.squeeze()>threshold,'AID','NOT ILL')
label = np.where(y_test.squeeze()>0.5,'AID','NOT ILL')
acc = (binary_pred==label).sum()/binary_pred.shape[0]
print('Accuracy on test dataset is {:.2f}%'.format(acc*100))

pd.crosstab(
    index=label,
    columns=binary_pred,
    rownames=['Label'],
    colnames=['Pred']
)

# Use plotly express (px) to visualize test results.
# px expects DataFrame input

pred_df = pd.DataFrame(
    {
        'pred_probability':pred_test.squeeze(),
        'label':label
    }
)
fig = px.scatter(data_frame=pred_df,y='pred_probability',color='label')
fig.show()