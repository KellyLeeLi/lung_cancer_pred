# lung_cancer_pred

```python
# 匯入資料
df = pd.read_csv('survey_lung_cancer.csv')
col_list = list(df.columns)
features = col_list[:-1]

print('data shape：', df.shape)
print('data information5：',df.info())
print(df.describe())
```
