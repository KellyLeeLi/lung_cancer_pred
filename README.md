# lung_cancer_pred


## 資料來源
資料集來自kaggle [Lung Cancer](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer/data)


## 匯入套件

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model, tree, neighbors, ensemble
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
```


## 載入資料
```python
# 匯入資料
df = pd.read_csv('survey_lung_cancer.csv')
col_list = list(df.columns)
features = col_list[:-1]
```


## 查看資料集資訊
```python
print('data shape：', df.shape)
print('data information5：',df.info())
print(df.describe())
```
輸出結果：
```
data shape： (309, 16)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 309 entries, 0 to 308
Data columns (total 16 columns):
 #   Column                 Non-Null Count  Dtype 
---  ------                 --------------  ----- 
 0   GENDER                 309 non-null    object
 1   AGE                    309 non-null    int64 
 2   SMOKING                309 non-null    int64 
 3   YELLOW_FINGERS         309 non-null    int64 
 4   ANXIETY                309 non-null    int64 
 5   PEER_PRESSURE          309 non-null    int64 
 6   CHRONIC DISEASE        309 non-null    int64 
 7   FATIGUE                309 non-null    int64 
 8   ALLERGY                309 non-null    int64 
 9   WHEEZING               309 non-null    int64 
 10  ALCOHOL CONSUMING      309 non-null    int64 
 11  COUGHING               309 non-null    int64 
 12  SHORTNESS OF BREATH    309 non-null    int64 
 13  SWALLOWING DIFFICULTY  309 non-null    int64 
 14  CHEST PAIN             309 non-null    int64 
 15  LUNG_CANCER            309 non-null    object
dtypes: int64(14), object(2)
memory usage: 38.8+ KB
data information5： None
```
共16個欄位，最後一個欄位LUNG_CANCER是我們要的正確結果。


## 資料前處理

刪除重複的資料
```python
# 刪除重複資料
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print('data shape：', df.shape)
```
輸出結果：
```
0
data shape： (276, 16)
```

將文字編碼成數值以便後續分析使用
```python
# 編碼 M=1, YES=1
le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
```

## 繪出圖表

#### 繪圖看看年齡分布狀況
```python
# 繪圖分析
# Age
fig,ax = plt.subplots(1,3,figsize=(20,6))
print(df['AGE'].info)
age = pd.DataFrame(df['AGE'])
sns.histplot(data=age, x="AGE", ax=ax[0])
sns.histplot(data=df, x="AGE", hue="LUNG_CANCER", kde=True, ax=ax[1])
sns.boxplot(x=df['AGE'])
sns.boxplot(x=df['LUNG_CANCER'], y=df['AGE'], ax=ax[2])
```

輸出結果：
![fig00](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(00).png "fig00")


#### 繪製熱圖
``` python
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,linewidth=0.5,fmt='0.2f')
```

輸出結果：
![fig1](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(1).png "fig1")

可以透過熱圖看到特徵之間的相關性


#### 各個特徵與肺癌之間的關聯
``` python
col_list.remove('AGE')

fig,ax = plt.subplots(15,2,figsize=(20, 90))
for i, j in enumerate(col_list):
    sns.countplot(data=df, x=j, ax=ax[i, 0])
    sns.countplot(data=df, x=j, hue='LUNG_CANCER', ax=ax[i, 1])

fig.tight_layout()
fig.subplots_adjust(right=0.8)
```

輸出結果：
![fig2](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(2).png "fig2")

## 預測

```python
# 將1,2轉成0,1
print(df.head())

for i in col_list[1:14]:
    df[i]=df[i].map({1: 0, 2: 1})
print(df.head())
```
```python
# 分割訓練及測試資料
x = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.25, random_state=42) 
```

### Logistic Regression 邏輯式回歸 87%
```python
# Logistic Regression
logistic = linear_model.LogisticRegression(random_state=42)
logistic.fit(x_train, y_train)
lr_pre = logistic.predict(x_test)
print(classification_report(y_test, lr_pre))

lr_cm = confusion_matrix(y_test, lr_pre)
plt.figure(figsize=(6,4))
lr_s = sns.heatmap(lr_cm, annot=True)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
lr_s.set(xlabel='Predicted', ylabel='Actual', title='Logistic Regression')
```

輸出結果：
```
              precision    recall  f1-score   support

           0       1.00      0.31      0.47        13
           1       0.86      1.00      0.93        56

    accuracy                           0.87        69
   macro avg       0.93      0.65      0.70        69
weighted avg       0.89      0.87      0.84        69
```
正確率為87%


混淆矩陣：

![fig3](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(3).png "fig3")


### Decision Tree 決策樹 86%
```python
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
```

輸出結果：
```
              precision    recall  f1-score   support

           0       0.80      0.31      0.44        13
           1       0.86      0.98      0.92        56

    accuracy                           0.86        69
   macro avg       0.83      0.64      0.68        69
weighted avg       0.85      0.86      0.83        69
```
正確率為86%


決策樹：
![fig4](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(4).png "fig4")

混淆矩陣：

![fig5](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(5).png "fig5")



### K Nearest Neighbors K-近鄰演算法 81%
```python
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
```

輸出結果：
```
KNN正確率： 0.8115942028985508
```
正確率為81%


混淆矩陣：

![fig7](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(7).png "fig7")



### Support Vector Machine 支援向量機 81%
透過param_gird來協助找出參數C與gamma的最佳組合
```python
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
```


輸出結果：
```
{'gamma': 0.01, 'C': 10}
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        13
           1       0.81      1.00      0.90        56

    accuracy                           0.81        69
   macro avg       0.41      0.50      0.45        69
weighted avg       0.66      0.81      0.73        69
```
{'gamma': 0.01, 'C': 10}

正確率為81%



混淆矩陣：

![fig8](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(8).png "fig8")


### Random Forest Classifier 隨機森林 87%
透過param_gird來協助找出參數n_estimators最佳數值
```python
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
```


輸出結果：
```
{'n_estimators': 200}
              precision    recall  f1-score   support

           0       1.00      0.31      0.47        13
           1       0.86      1.00      0.93        56

    accuracy                           0.87        69
   macro avg       0.93      0.65      0.70        69
weighted avg       0.89      0.87      0.84        69
```
{'n_estimators': 200}

正確率為87%



混淆矩陣：

![fig9](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(9).png "fig9")



### Gradient Boosting Classifier 87%
透過param_gird來協助找出參數n_estimators和learning_rate的最佳組合
```python
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
```


輸出結果：
```
{'n_estimators': 75, 'learning_rate': 0.05}
              precision    recall  f1-score   support

           0       1.00      0.31      0.47        13
           1       0.86      1.00      0.93        56

    accuracy                           0.87        69
   macro avg       0.93      0.65      0.70        69
weighted avg       0.89      0.87      0.84        69
```
{'n_estimators': 75, 'learning_rate': 0.05}

正確率為87%



混淆矩陣：

![fig10](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(10).png "fig10")




### XGBoost Classifier 86%
透過param_gird來協助找出參數n_estimators、gamma和learning_rate的最佳組合
```python
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
```


輸出結果：
```
{'n_estimators': 125, 'learning_rate': 0.75, 'gamma': 1}
              precision    recall  f1-score   support

           0       1.00      0.23      0.38        13
           1       0.85      1.00      0.92        56

    accuracy                           0.86        69
   macro avg       0.92      0.62      0.65        69
weighted avg       0.88      0.86      0.82        69
```
{'n_estimators': 125, 'learning_rate': 0.75, 'gamma': 1}

正確率為86%



混淆矩陣：

![fig11](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(11).png "fig11")



### LightGBM Classifier 88%
佔用的資源XGBoost
準確率卻提高了
```python
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
```


輸出結果：
```
              precision    recall  f1-score   support

           0       1.00      0.38      0.56        13
           1       0.88      1.00      0.93        56

    accuracy                           0.88        69
   macro avg       0.94      0.69      0.74        69
weighted avg       0.90      0.88      0.86        69
```

正確率為88%



混淆矩陣：

![fig12](https://github.com/KellyLeeLi/lung_cancer_pred/blob/main/fig/Figure%202023-11-17%20100359%20(12).png "fig12")



## 未來持續改進
- 嘗試挑選參數，看是否能提升準確度
- 加入深度學習模型
- 發佈模型以提升實用性

