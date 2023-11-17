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

繪圖看看年齡與肺癌的關係
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
![fig0]( "fig0")
