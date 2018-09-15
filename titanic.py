import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
import matplotlib.pyplot as plt


train = pd.read_csv('C:/Users/Aditya/Downloads/TiTanic/train.csv') 
test = pd.read_csv('C:/Users/Aditya/Downloads/TiTanic/test.csv')

train.head()
test.head()

train.describe()
test.describe()
train.columns.values
test.columns.values

train.isna().head()
test.isna().head()


train.isna().sum()
test.isna().sum()


train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace =True)


train.isna().sum()
test.isna().sum()

train['Ticket'].head()
test['Cabin'].head()

train[['Pclass','Survived']].groupby(['Pclass'],as_index = False).mean().sort_values(by='Survived',ascending= False)

train[['Sex','Survived']].groupby(['Sex'],as_index = False).mean().sort_values(by = 'Survived', ascending =False)


train[['SibSp','Survived']].groupby(['SibSp'],as_index = False).mean().sort_values(by='Survived', ascending=False)


g= sns.FacetGrid(train,col='Survived')
g.map(plt.hist,'Age',bins=20)


grid = sns.FacetGrid(train,col='Survived',row='Pclass',size=2.2, aspect= 1.6)
grid.map(plt.hist, "Age", bins=20)
grid.add_legend()


train.info()
test.info()


train = train.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
test = test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])

labelEncoder = LabelEncoder()
labelEncoder.fit(test['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])


X = np.array(train.drop(['Survived'],axis =1).astype('float'))
y = np.array(train['Survived'])

train.info()

kmean = KMeans(n_clusters=2)
kmean.fit(X)
    
y[i]
len(X)


correct = 0 
for i in range(len(X)):
    #consert simple dataframe into array means each row will treat as array
    predict_me = np.array(X[i].astype(float))
#    print(predict_me)
#    print(len(predict_me))
    predict_me = predict_me.reshape(-1,len(predict_me))
    #treat every row as specific dataframe
#    print(predict_me)
    prediction = kmean.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
        
print(correct/len(X))






















