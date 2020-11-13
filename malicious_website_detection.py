import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('dataset.csv')

df = df.drop(['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'], axis = 1)

df = df.iloc[:, 1:]

df.isnull().sum()

df.fillna(df.mean(), inplace=True)


df.iloc[:, 3] = df.iloc[:, 3].astype(str)

from sklearn.preprocessing import LabelEncoder
le2 = LabelEncoder()
le3 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()


df.iloc[:, 2] = le2.fit_transform(df.iloc[:, 2])

df.iloc[:, 3] = le3.fit_transform(df.iloc[:, 3])

df.iloc[:, 5] = le5.fit_transform(df.iloc[:, 5])

df.iloc[:, 6] = le6.fit_transform(df.iloc[:, 6])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [2, 3, 5, 6])], remainder='passthrough')


df_enc = ct.fit_transform(df)

df_enc.shape

df_enc = pd.DataFrame(df_enc.toarray())

X = df_enc.iloc[:, :-1]
y = df_enc.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred))



















