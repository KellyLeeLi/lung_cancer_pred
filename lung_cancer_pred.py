import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt

# 匯入資料
df = pd.read_csv('survey_lung_cancer.csv')
col_list = list(df.columns)
features = col_list[:-1]

print('data shape：', df.shape)
print('data information5：',df.info())
print(df.describe())


# 刪除重複資料
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print('data shape：', df.shape)
print('data information5：',df.info())
print(df.describe())


# 編碼 M=1, YES=1
le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])


# 繪圖分析
# Age
# sns.set_palette("pastel")
fig,ax = plt.subplots(1,3,figsize=(20,6))
# penguins = sns.load_dataset("penguins")
print(df['AGE'].info)
age = pd.DataFrame(df['AGE'])
sns.histplot(data=age, x="AGE", ax=ax[0])
sns.histplot(data=df, x="AGE", hue="LUNG_CANCER", kde=True, ax=ax[1])
sns.boxplot(x=df['LUNG_CANCER'], y=df['AGE'], ax=ax[2])



plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,linewidth=0.5,fmt='0.2f')



col_list.remove('AGE')

fig,ax = plt.subplots(15,2,figsize=(20, 90))
for i, j in enumerate(col_list):
    sns.countplot(data=df, x=j, ax=ax[i, 0])
    sns.countplot(data=df, x=j, hue='LUNG_CANCER', ax=ax[i, 1])

fig.tight_layout()
fig.subplots_adjust(right=0.8)


# 預測
from sklearn import linear_model, tree, neighbors, ensemble
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report


# 將1,2轉成0,1
print(df.head())

for i in col_list[1:14]:
    df[i]=df[i].map({1: 0, 2: 1})
print(df.head())


# 預測

# 分割訓練及測試資料
x = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.25, random_state=42) 

# Logistic Regression
logistic = linear_model.LogisticRegression(random_state=42)
logistic.fit(x_train, y_train)
lr_pre = logistic.predict(x_test)
print(classification_report(y_test, lr_pre))
# print(logistic.score(x_test, y_test))


lr_cm = confusion_matrix(y_test, lr_pre)
plt.figure(figsize=(6,4))
lr_s = sns.heatmap(lr_cm, annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
lr_s.set(xlabel='Predicted', ylabel='Actual', title='Logistic Regression')

# decision tree
dtree = tree.DecisionTreeClassifier(random_state=42, max_depth=8)
dtree.fit(x_train, y_train)
print(dtree.score(x_test, y_test))
dt_pre = dtree.predict(x_test)
print(classification_report(y_test, dt_pre))

  # 畫出決策樹
fig, ax = plt.subplots(figsize=(55, 20))
tree.plot_tree(dtree, ax=ax, fontsize=12,
               filled=True, rounded=True,
               feature_names=features)
plt.show()


  # 混淆矩陣
dt_cm = confusion_matrix(y_test, dt_pre)
plt.figure(figsize=(6,4))
dt_s = sns.heatmap(dt_cm, annot=True)
dt_s.set(xlabel='Predicted', ylabel='Actual', title='Decision Tree')



# KNN
k_range = range(1,10)
k_scores = []
for i in k_range:
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

x_ticks = [1,2,3,4,5,6,7,8,9,10]
plt.figure(figsize=(6,4))
plt.plot(k_range,k_scores)
plt.xlabel('Value of i for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.xticks(ticks=x_ticks)
plt.grid()
plt.show()


knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knn_pre = knn.predict(x_test)

print('KNN正確率：',knn.score(x_test, y_test))

  # 混淆矩陣
knn_cm = confusion_matrix(y_test, knn_pre)
plt.figure(figsize=(6,4))
dt_s = sns.heatmap(knn_cm, annot=True)
dt_s.set(xlabel='Predicted', ylabel='Actual', title='K Neighbors Classifier')


# SVC
from sklearn.svm import SVC
param_gird = {'C':[0.001, 0.01, 0.1, 1, 10, 100],
              'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
svc = RandomizedSearchCV(SVC(random_state=42), param_gird)
svc.fit(x_train, y_train)
svc_pre = svc.predict(x_test)
print(svc.best_params_)
print(classification_report(y_test, svc_pre))

  # 混淆矩陣
svc_cm = confusion_matrix(y_test, svc_pre)
plt.figure(figsize=(6,4))
svc_s = sns.heatmap(svc_cm, annot=True)
svc_s.set(xlabel='Predicted', ylabel='Actual', title='Support Vector Classifier')


# Random Forest Classifier
param_gird = {'n_estimators':[10, 30, 50, 75, 100, 120, 150, 200, 300, 500]}
rfc = RandomizedSearchCV(ensemble.RandomForestClassifier(random_state=42), param_gird)
rfc.fit(x_train, y_train)
rfc_pre = rfc.predict(x_test)
print(rfc.best_params_)
print(classification_report(y_test, rfc_pre))


  # 混淆矩陣
rfc_cm = confusion_matrix(y_test, rfc_pre)
plt.figure(figsize=(6,4))
rfc_s = sns.heatmap(rfc_cm, annot=True)
rfc_s.set(xlabel='Predicted', ylabel='Actual', title='Random Forest Classifier')


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
param_gird = {'n_estimators':[50, 75, 100, 125, 150, 200, 300],
              'learning_rate':[0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]}
gbc = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), param_gird)
gbc.fit(x_train, y_train)
gbc_pre = gbc.predict(x_test)
print(gbc.best_params_)
print(classification_report(y_test, gbc_pre))

  # 混淆矩陣
gbc_cm = confusion_matrix(y_test, gbc_pre)
plt.figure(figsize=(6,4))
gbc_s = sns.heatmap(gbc_cm, annot=True)
gbc_s.set(xlabel='Predicted', ylabel='Actual', title='Gradient Boosting Classifier')


# XGBoost Classifier (佔用的資源比LightGBM多)
from xgboost import XGBClassifier
param_gird = {'n_estimators':[50, 75, 100, 125, 150, 200, 300],
              'learning_rate':[0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1],
              'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
xgb = RandomizedSearchCV(XGBClassifier(random_state=42), param_gird)
xgb.fit(x_train, y_train)
xgb_pre = xgb.predict(x_test)
print(xgb.best_params_)
print(classification_report(y_test, xgb_pre))

  # 混淆矩陣
xgb_cm = confusion_matrix(y_test, xgb_pre)
plt.figure(figsize=(6,4))
xgb_s = sns.heatmap(xgb_cm, annot=True)
xgb_s.set(xlabel='Predicted', ylabel='Actual', title='XGBoost Classifier')



# LightGBM Classifier
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(x_train, y_train)
lgbm_pre = lgbm.predict(x_test)
print(classification_report(y_test, lgbm_pre))

  # 混淆矩陣
lgbm_cm = confusion_matrix(y_test, lgbm_pre)
plt.figure(figsize=(6,4))
lgbm_s = sns.heatmap(lgbm_cm, annot=True)
lgbm_s.set(xlabel='Predicted', ylabel='Actual', title='LightGBM Classifier')

