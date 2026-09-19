"""
Leak-Free Machine Learning Pipelines & Hyperparameter Tuning
A best-practices machine learning engineering project demonstrating leak-free data preprocessing, ColumnTransformer architecture, and systematic GridSearchCV hyperparameter optimization in Scikit-Learn.

Original Kaggle Notebook: https://www.kaggle.com/code/lazer999/0-80-hyperparameter-tuning-with-pipelines
Author: Muhammad Musa Khan (Kaggle Master: https://kaggle.com/lazer999)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# --- Smart Dataset Path Resolution ---
def _resolve_data_path(file_path):
    """Checks local and data/ directories if dataset path is missing."""
    if os.path.exists(file_path):
        return file_path
    base = os.path.basename(file_path)
    candidates = [
        base,
        os.path.join("data", base),
        os.path.join("..", "data", base),
        file_path.replace("/kaggle/input/", "data/"),
        file_path.replace("../input/", "data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path

# --- Pipeline Execution ---

# --- Cell 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set(style='darkgrid', font_scale=2)
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn import set_config

# --- Cell 2 ---
set_config(display='diagram')

# --- Cell 3 ---
df_train = pd.read_csv('../input/spaceship-titanic/train.csv')
df_test = pd.read_csv('../input/spaceship-titanic/test.csv')

df_train.head()

# --- Cell 4 ---
r1,c1 = df_train.shape
print('The training data has {} rows and {} columns'.format(r1,c1))
r2,c2 = df_test.shape
print('The validation data has {} rows and {} columns'.format(r2,c2))

# --- Cell 5 ---
df_train.info()

# --- Cell 6 ---
df_train.describe()

# --- Cell 7 ---
df_test.describe()

# --- Cell 8 ---
# Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
df_train[['Deck','Num','Side']] = df_train.Cabin.str.split('/',expand=True)
df_test[['Deck','Num','Side']] = df_test.Cabin.str.split('/',expand=True)

# --- Cell 9 ---
df_train['total_spent']= df_train['RoomService']+ df_train['FoodCourt']+ df_train['ShoppingMall']+ df_train['Spa']+ df_train['VRDeck']
df_test['total_spent']=df_test['RoomService']+df_test['FoodCourt']+df_test['ShoppingMall']+df_test['Spa']+df_test['VRDeck']

# --- Cell 10 ---
df_train['AgeGroup'] = 0
for i in range(6):
    df_train.loc[(df_train.Age >= 10*i) & (df_train.Age < 10*(i + 1)), 'AgeGroup'] = i
# Same for test data
df_test['AgeGroup'] = 0
for i in range(6):
    df_test.loc[(df_test.Age >= 10*i) & (df_test.Age < 10*(i + 1)), 'AgeGroup'] = i

# --- Cell 11 ---
plt.figure(figsize=(10,6))
sns.countplot(df_train.Deck,hue=df_train.Transported);

# --- Cell 12 ---
df_train['Num'].nunique()

# --- Cell 13 ---
plt.figure(figsize=(10,5))
sns.countplot(df_train.Side,hue=df_train.Transported)
plt.legend(loc=4);

# --- Cell 14 ---
plt.figure(figsize=(10,6))
sns.countplot(y=df_train['AgeGroup'],hue=df_train['Transported']);

# --- Cell 15 ---
df_train.head()

# --- Cell 16 ---
X=df_train.drop('Transported',axis=1)
y = df_train['Transported']

# --- Cell 17 ---
X['Num'] = pd.to_numeric(X['Num'])

# --- Cell 18 ---
X=X.drop(['PassengerId','Name'],axis=1)

# --- Cell 19 ---
cat_cols=X.select_dtypes('object').columns.to_list()
cat_cols

# --- Cell 20 ---
num_cols=X.select_dtypes(exclude='object').columns.to_list()
num_cols

# --- Cell 21 ---
numeric_preprocessor = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaling',StandardScaler()),
])

# --- Cell 22 ---
categorical_preprocessor = Pipeline(steps=[
    ('encoder',OneHotEncoder(handle_unknown='ignore')),  
    ('imputer',SimpleImputer(strategy='constant')),
    
])

# --- Cell 23 ---
preprocessor = ColumnTransformer([
    ('categorical',categorical_preprocessor,cat_cols),
    ('numeric',numeric_preprocessor,num_cols)

])

# --- Cell 24 ---
Pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',GradientBoostingClassifier())])

# --- Cell 25 ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)

# --- Cell 26 ---
Pipe.fit(X_train,y_train)

# --- Cell 27 ---
pred=Pipe.predict(X_val)

# --- Cell 28 ---
pred=Pipe.predict(X_train)
pred_y=Pipe.predict(X_val)
print('Train accuracy ',accuracy_score(y_train.values,pred))
print('Validation accuracy',accuracy_score(y_val.values,pred_y))

# --- Cell 29 ---
# you can try more parameters, but hell it takes a lot of time.
param_grid={'model__n_estimators':[500,1000],'model__learning_rate':[0.1,0.2],'model__verbose':[1],'model__max_depth':[2,3]}
from sklearn.model_selection import GridSearchCV
gcv=GridSearchCV(Pipe,param_grid=param_grid,cv=5,scoring="roc_auc")

# --- Cell 30 ---
gcv.fit(X,y)

# --- Cell 31 ---
params = gcv.best_params_
params

# --- Cell 32 ---
gcv.best_score_

# --- Cell 33 ---
Hyper_Pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',GradientBoostingClassifier(n_estimators=500,max_depth=3, random_state=1)),
])
Hyper_Pipe.fit(X_train,y_train)

# --- Cell 34 ---
pred=Hyper_Pipe.predict(X_train)
pred_y=Hyper_Pipe.predict(X_val)
print('Train accuracy ',accuracy_score(y_train.values,pred))
print('Validation accuracy',accuracy_score(y_val.values,pred_y))

# --- Cell 35 ---
confusion_matrix(pred_y,y_val.values)

# --- Cell 36 ---
y_pred = Hyper_Pipe.predict(df_test)

sub=pd.DataFrame({'Transported':y_pred.astype(bool)},index=df_test['PassengerId'])

sub.head()

# --- Cell 37 ---
sub.to_csv('submission')

# --- Cell 38 ---
pd.read_csv('../input/spaceship-titanic/sample_submission.csv')



if __name__ == "__main__":
    print("Pipeline execution complete.")
