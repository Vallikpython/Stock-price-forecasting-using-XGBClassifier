import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./archive (1)/coin_Ethereum.csv')
df = df.drop(['Name'], axis=1)
df = df.drop(['Symbol'], axis=1)
df = df.drop(['SNo'], axis=1)


plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Ethereum Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

print(df.isnull().sum())

features = ['Open', 'High', 'Low', 'Close', 'Volume']

# Создаем несколько графиков histplot
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.histplot(df[col])
plt.show()
# Создаем несколько графиков boxplot
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()

# Убираем время из столбца Date
df['Date'] = df['Date'].str.split(' ').str [0]
splitted = df['Date'].str.split('-', expand=True)

# Создаем новые столбцы
df['day'] = splitted[2].astype('int')
df['month'] = splitted[1].astype('int')
df['year'] = splitted[0].astype('int')


# Создаем столбец квартал.
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)



data_grouped = df.drop('Date', axis=1).groupby('year').mean()
# Из приведенных гистограм можно сделать вывод что цена на Ethereum в трое выросла в 2021 году.
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()



df.drop('Date', axis=1).groupby('is_quarter_end').mean()
# Добавляем еще столбцы, которые помогут в обучении
df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
# Целевая функция, которая является сигналом, стоит ли покупать.
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Строим круговую диаграму что бы проверить сбалансирована ли целевая функция
plt.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 10)) 

# Поскольку нас интересуют только сильно 
# коррелированные объекты, мы будем визуализировать 
# нашу тепловую карту только в соответствии с этим критерием.
sb.heatmap(df.drop(['Date', 'Marketcap'], axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()



features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Создаем экземпляр класса
scaler = StandardScaler()
# Нормализуем данные

features = scaler.fit_transform(features)
# Разделяем данные на тестовые и обучающие
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

# Создаем модель
models = XGBClassifier()

# Обучаем модель.
models.fit(X_train, Y_train)


print('Точность обучения : ', metrics.roc_auc_score(
  Y_train, models.predict_proba(X_train)[:,1]))
print('Точность валидации : ', metrics.roc_auc_score(
  Y_valid, models.predict_proba(X_valid)[:,1]))


# Строим матрицу ошибок
ConfusionMatrixDisplay.from_estimator(models, X_valid, Y_valid)
plt.show()
