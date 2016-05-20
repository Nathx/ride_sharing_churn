
# coding: utf-8

# In[165]:

import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.cross_validation import KFold, cross_val_score
#get_ipython().magic(u'matplotlib inline')


# In[166]:


def get_score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return (y_pred == y_test).mean()


def plot_feat_importances():
    gbm = xgboost.XGBClassifier(silent=False, seed=8).fit(X_train, y_train)
    plot = xgboost.plot_importance(gbm)
    ticks = plot.set_yticklabels(df_xgb.columns)

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], yerr=std[indices], color='lightblue')
    ticks = plt.yticks(range(len(indices)), df_xgb.columns)


def plotting_perfect_ratings():
    df_xgb['perfect_rating_of_driver'] = (df_xgb.avg_rating_of_driver == 5)
    df_xgb.groupby('perfect_rating_of_driver').mean().churn.plot(kind='bar')
    plt.title('Perfect rating of driver')

    df_xgb['perfect_rating_by_driver'] = (df_xgb.avg_rating_by_driver == 5)
    df_xgb.groupby('perfect_rating_by_driver').mean().churn.plot(kind='bar')
    plt.title('Perfect rating by driver')


# In[151]:

df_origin = pd.read_csv('data/churn.csv')


# In[164]:

df = df_origin.copy()

df.last_trip_date = pd.to_datetime(df.last_trip_date)

df['days_since_last_trip'] = (df.last_trip_date.max() - df.last_trip_date) / np.timedelta64(1, 'D')
df['churn'] = (df['days_since_last_trip'] > 30)

df.signup_date = pd.to_datetime(df.signup_date)
df['days_since_signup'] = (df.last_trip_date.max() - df.signup_date) / np.timedelta64(1, 'D')
df = df.drop(['days_since_last_trip', 'last_trip_date', 'signup_date'], axis=1)

df_phone = pd.get_dummies(df['phone'])
frames = [df, df_phone]
df = pd.concat(frames, axis=1)
df = df.drop(['phone', 'Android'], axis=1)

df_city = pd.get_dummies(df['city'])
frames = [df, df_city]
df = pd.concat(frames, axis=1)
df = df.drop('city', axis=1)
df = df.drop(['Winterfell'], axis=1).copy()




df_xgb = df.copy()
# Missing values
df_xgb.avg_rating_by_driver = df_xgb.avg_rating_by_driver.fillna(df_xgb.avg_rating_by_driver.mean())
df_xgb.avg_rating_of_driver = df_xgb.avg_rating_of_driver.fillna(df_xgb.avg_rating_of_driver.mean())

y = df_xgb.pop('churn')
X = df_xgb.values
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=8, test_size=.2)

gbm = xgboost.XGBClassifier(silent=False, seed=8,n_estimators=300)
cross_val_score(gbm, X_train, y_train, cv=5).mean()

#cross_val_score(X_train, X_test, y_train, y_test, 5)


# In[ ]:
