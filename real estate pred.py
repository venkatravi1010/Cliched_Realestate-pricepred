#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = [20, 10]


# In[20]:


df1 = pd.read_csv("Bengaluru_House_Data.csv.xls")
df1.head()


# In[21]:


df1.shape


# In[22]:


df1.groupby('area_type')['area_type'].agg('count')


# In[25]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis= 'columns')
df2.head()


# In[26]:


df2.isnull().sum()


# In[27]:


df3 = df2.dropna()
df3.isnull().sum()


# In[29]:


df3.shape


# In[30]:


df3['size'].unique()


# In[32]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[33]:


df3.head()


# In[34]:


df3['bhk'].unique()


# In[35]:


df3[df3.bhk>20]


# In[36]:


df3.total_sqft.unique()


# In[37]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[40]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[41]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[42]:


convert_sqft_to_num('2166')


# In[43]:


convert_sqft_to_num('2100 - 2850')


# In[44]:


convert_sqft_to_num('34.46Sq. Meter')


# In[46]:


df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head(3)


# In[47]:


df4.loc[30]


# In[77]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[81]:


df5.location.unique()


# In[83]:


df5.location = df5.location.apply(lambda x: x.strip())

location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[87]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[89]:


df5.location.unique()


# In[90]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
df5.location.unique()


# In[91]:


df5.head(10)


# In[93]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[94]:


df5.shape


# In[95]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape
            


# In[97]:


df6.price_per_sqft.describe()


# In[98]:


def remove_pps_outlier(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft) 
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df], ignore_index = True)
    return df_out
df7 = remove_pps_outlier(df6)
df7.shape
                           
    


# In[99]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[100]:


plot_scatter_chart(df7,"Hebbal")


# In[101]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[102]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[103]:


plot_scatter_chart(df8,"Hebbal")


# In[104]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[105]:


df8.bath.unique()


# In[106]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[107]:


df8[df8.bath>10]


# In[108]:


df8[df8.bath>df8.bhk+2]


# In[109]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[110]:


df9.head(2)


# In[111]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# In[112]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[114]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[115]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[116]:


df12.shape


# In[117]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[118]:


X.shape


# In[119]:


y = df12.price
y.head(3)


# In[121]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[122]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[123]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[124]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[126]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[128]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[129]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




