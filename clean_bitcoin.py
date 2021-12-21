#!/usr/bin/env python
# coding: utf-8

# In[191]:


import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go


# In[192]:


df_bitcoin = pd.read_json('gs://eecs6893_final/final_data/bitcoin_price.json', orient='split')
df_tweet1 = pd.read_csv('gs://eecs6893_final/final_data/live_tweet.csv', names=['sentiment', 'time'])
df_tweet2 = pd.read_csv('gs://eecs6893_final/final_data/live_tweet(1).csv', names=['sentiment', 'time'])


# In[193]:


df_bitcoin
df_bitcoin.to_csv('gs://eecs6893_final/final_data/bitcoin.csv')


# In[194]:


df_bitcoin['time'] = df_bitcoin['datetime'].apply(lambda x: x[2:-3])
df_bitcoin.drop('datetime', axis=1, inplace=True)


# In[195]:


df_bitcoin = df_bitcoin[['last', 'time']]
df_bitcoin


# In[196]:


df_tweet2


# In[197]:


df_tweet = pd.concat([df_tweet1, df_tweet2])
df_tweet


# In[199]:


df = pd.merge(df_bitcoin, df_tweet, on='time', how='inner')
df.to_csv('gs://eecs6893_final/final_data/df.csv')
df2 = df


# In[200]:


df2['s2'] = df['sentiment']**2
df2['s3'] = df['sentiment']**3


# In[201]:


plt.plot(df['sentiment'])


# In[202]:


plt.plot(df['last'])


# In[203]:


from pyspark.sql.types import *

# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)


# In[204]:


df = pandas_to_spark(df)


# In[205]:


df.show()


# ### Baseline Model: LinearRegression

# In[206]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[207]:


df_regression = df[['sentiment', 's2', 's3', 'last']] 
vector_assembler = VectorAssembler(inputCols=['sentiment', 's2', 's3'], outputCol="features")
vec_df = vector_assembler.transform(df_regression)
vec_df = vec_df.drop('sentiment', 's2', 's3')


# In[208]:


lr = LinearRegression(featuresCol='features', labelCol='last', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(vec_df)


# In[209]:


lr_pred = lrModel.transform(vec_df)


# In[210]:


print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# In[211]:


pd.DataFrame(lr_pred)


# In[212]:


lr_pred.show()


# In[ ]:




