import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# 读取 CSV 文件
data = pd.read_csv('6004data.csv')

# 将指定列中的 0 值替换为 'zero'，非 0 值替换为 'non_zero'
data['aki'] = data['aki'].apply(lambda x: 0 if x == 0 else 1)

# 将 'F' 替换为 1，'M' 替换为 0，并将类型转换为整数
data['gender'] = data['gender'].replace({'F': 1, 'M': 0}).astype(int)

# 删除 'race''id' 列
data = data.drop(['race', 'id'], axis=1)

# 用每列的平均值填充缺失值
data = data.fillna(data.mean())

# 从数据中随机抽取5000行作为样本
sample_size = 5000  # 指定抽样数量
sample_data = data.sample(n=sample_size, random_state=42)

# 划分数据集为特征和标签
X_df = sample_data.drop('aki', axis=1)
y_df = sample_data['aki']

X_df = X_df[['admission_age', 'sbp_min', 'resp_rate_max', 'bun_min', 'gcs_verbal', 'weight_admit']]

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# 定义随机森林模型
rf_model = RandomForestClassifier()

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 定义网格搜索，减少交叉验证的折数为3
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3)

# 在训练集上拟合网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("调优后随机森林模型准确率:", accuracy)

# 计算AUC
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("调优后随机森林模型AUC:", auc)
