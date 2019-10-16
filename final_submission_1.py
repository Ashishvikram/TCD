#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# # Importing libraries
import pandas as pd


# In[2]:


import numpy as np
from sklearn import linear_model
from sklearn.base import TransformerMixin


# In[3]:


#Reading the file
df = pd.read_csv('C:/Users/Vikram/income_pred.csv')


# In[4]:


#cleaning the data, fill missing values
df = df.replace('#N/A',np.nan)
df = df.replace('0', np.nan)
df = df.replace('#', np.nan)
df = df[df['Income in EUR'] >= 0]

#Imputing NA values with mean values

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

X = pd.DataFrame(df)
xt = DataFrameImputer().fit_transform(X)
df=xt


# In[5]:


#Removing the income and instance column
X = df.drop('Income in EUR',axis=1)
X = X.drop('Instance',axis=1)
X = pd.get_dummies(X, columns=["Profession","University Degree","Country","Year of Record","Gender","Hair Color"], prefix=["prof","unv","cou","yea","gen","hair"],drop_first=True)

y = df[['Income in EUR']]



# In[6]:


# Splitting the data between training and Testing and Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)


# In[7]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
y_train.shape


# In[8]:


# getting all hot_econded columns "unv","cou","yea","gen","Prof","hair","age","size of city","body Height"
#Calculate RSME and store predictions to CSV
#X = X.drop('yea_1999.4192859130778', axis=1)
prof_col = [col for col in X if col.startswith('prof')]
hair_col = [col for col in X if col.startswith('hair')]
edu_col = [col for col in X if col.startswith('unv')]
country_col = [col for col in X if col.startswith('cou')]
year_col = [col for col in X if col.startswith('yea')]
gender_col = [col for col in X if col.startswith('gen')]
size_city_col = ['Size of City']
age_col = ['Age']
#glass_col = ['Wears Glasses']
height_col = ['Body Height [cm]']
#print(gender_col)
#Combining all the columns
combined_col = edu_col + country_col + year_col + gender_col + hair_col + size_city_col + age_col + prof_col + height_col
print(len(combined_col))
reg = LinearRegression()
reg.fit(X_train[combined_col], y_train)
y_predicted = reg.predict(X_train[combined_col])
print("rms: %.2f" % sqrt(mean_squared_error(y_train, y_predicted)))


# In[10]:



test = pd.read_csv('E:/test.csv')
test = test.replace('#N/A', np.nan)
test = test.replace('0', np.nan)
test = test.drop('Income', axis=1)
test.fillna(method='ffill', inplace=True)
test = pd.get_dummies(test, columns=["Profession","University Degree","Country","Year of Record","Gender","Hair Color"], prefix=["prof","unv","cou","yea","gen","hair"],drop_first=True)
t_prof_col = [col for col in test if col.startswith('prof')]
t_hair_col = [col for col in X if col.startswith('hair')]
t_country_col = [col for col in test if col.startswith('cou')]
t_edu_col = [col for col in test if col.startswith('edu')]
t_year_col = [col for col in test if col.startswith('yea')]
t_gender_col = [col for col in test if col.startswith('gen')]
t_size_city_col = ['Size of City']
t_age_col = ['Age']
t_height_col = ['Body Height [cm]']
t_combined_col = t_edu_col + t_country_col + t_year_col + t_gender_col + t_hair_col + t_size_city_col + t_age_col + t_height_col + t_prof_col
missing = np.setdiff1d(combined_col,t_combined_col)
print(len(missing))
for i in range(len(missing)):
    test[missing[i]] = np.nan
test = test.fillna(0)
print(test.isnull().sum().sum())
test_y_predicted = reg.predict(test[combined_col])
print(test_y_predicted)
np.savetxt("output6.csv", test_y_predicted, delimiter=",")


# In[ ]:





# In[ ]:




