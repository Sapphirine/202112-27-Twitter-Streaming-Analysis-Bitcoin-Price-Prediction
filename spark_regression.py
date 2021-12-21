#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import plotly.offline as py
import plotly.graph_objs as go


# In[52]:


df_bitcoin = pd.read_json('bitcoin_price_new.json', orient='split')
df_tweet = pd.read_csv('live_tweet (3).csv', names=['bitcoin_price_crypto', 'bitcoin_price', 'price_crypto', 'elon_musk_crypto',
             'cryptocurrency', 'dogecoin', 'Ethereum_ETH', 'Litecoin_LTC', 'blockchain', 'time'])


# In[53]:


df_bitcoin.head()


# In[54]:


df_bitcoin['time'] = df_bitcoin['datetime'].apply(lambda x: x[5:10])
df_bitcoin.drop('datetime', axis=1, inplace=True)


# In[55]:


df_tweet.head()


# In[56]:


df_tweet['time'] = df_tweet['time'].apply(lambda x: str(x))


# In[57]:


df_tweet['time'] = df_tweet['time'].apply(lambda x: x[3:8])
df_tweet.head()


# In[58]:


df_bitcoin = df_bitcoin[['last', 'time']]
df_bitcoin


# In[59]:


df = pd.merge(df_bitcoin, df_tweet, on='time', how='inner')
#df.to_csv('gs://eecs6893_final/final_data/df.csv')
df.head()


# In[60]:


a = df.groupby('time')['last'].mean().reset_index()
a


# In[62]:


plt.plot(a['time'], a['last'])


# In[76]:


a = df.groupby('time')['elon_musk_crypto'].mean().reset_index()
b = df.groupby('time')['dogecoin'].mean().reset_index()


# In[79]:


fig, ax = plt.subplots()
ax.plot(a['time'], a['elon_musk_crypto'], label=['elon musk bitcoin'])
ax.plot(b['time'], b['dogecoin'], label=['crypto bitcoin'])
ax.legend(loc='best', )


# In[12]:


plt.plot(df['bitcoin_price_crypto'])


# In[13]:


plt.plot(df['bitcoin_price'])


# In[14]:


plt.plot(df['elon_musk_crypto'])


# In[15]:


plt.plot(df['dogecoin'])


# In[11]:


fig, ax = plt.subplots()
plt.plot(df['time'], df['last'])


# In[17]:


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


# In[18]:


df = pandas_to_spark(df)


# In[19]:


df.show()


# ### Baseline Model: LinearRegression

# In[20]:


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[21]:


df_regression = df[['bitcoin_price_crypto', 'bitcoin_price', 'price_crypto', 'elon_musk_crypto',
             'cryptocurrency', 'dogecoin', 'Ethereum_ETH', 'Litecoin_LTC', 'blockchain','last']]
vector_assembler = VectorAssembler(inputCols=['bitcoin_price_crypto', 'bitcoin_price', 'price_crypto', 'elon_musk_crypto',
             'cryptocurrency', 'dogecoin', 'Ethereum_ETH', 'Litecoin_LTC', 'blockchain'], outputCol="features", handleInvalid = "skip")
vec_df = vector_assembler.transform(df_regression)
vec_df = vec_df.select(['features', 'last'])
vec_df.show(3)


# In[22]:


lr = LinearRegression(featuresCol='features', labelCol='last', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(vec_df)


# In[23]:


print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))


# # Remember to include model coefficients

# In[24]:


trainingSummary = lrModel.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)


# ### Generalized Linear Regression

# In[25]:


from pyspark.ml.regression import GeneralizedLinearRegression
glr = GeneralizedLinearRegression(featuresCol='features', labelCol='last', 
                                  family="gamma", link="identity", maxIter=15, regParam=0.3)
glrModel = glr.fit(vec_df)


# In[26]:


print("Coefficients: " + str(glrModel.coefficients))
print("Intercept: " + str(glrModel.intercept))


# In[27]:


summary = glrModel.summary
summary
#print("RMSE: %f" % gtrainingSummary.rootMeanSquaredError)
#print("r2: %f" % gtrainingSummary.r2)


# In[28]:


print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
print("T Values: " + str(summary.tValues))
print("P Values: " + str(summary.pValues))
print("Dispersion: " + str(summary.dispersion))
print("Null Deviance: " + str(summary.nullDeviance))
print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
print("Deviance: " + str(summary.deviance))
print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()


# ## Decision Tree

# In[32]:


from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol="features", labelCol='last')
dtModel = dt.fit(vec_df)


# In[33]:


from pyspark.ml.evaluation import RegressionEvaluator
pred = dtModel.transform(vec_df)
evaluator = RegressionEvaluator(
    labelCol="last", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(pred)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[34]:


tree.export_graphviz(dtModel,
                     out_file="tree.dot",
                     feature_names = fn, 
                     class_names=cn,
                     filled = True)


# In[35]:


df_tree = df_tree.dropna()
X = df_tree.loc[:, ['bitcoin_price_crypto', 'bitcoin_price', 'price_crypto', 'elon_musk_crypto',
             'cryptocurrency', 'dogecoin', 'Ethereum_ETH', 'Litecoin_LTC', 'blockchain']]
y = df_tree[['last']]


# In[36]:


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz
clf = DecisionTreeRegressor(max_depth=3)
clf.fit(X, y)


# In[37]:


from IPython.display import display
display(graphviz.Source(export_graphviz(clf)))


# In[39]:


fn = ['bitcoin_price_crypto', 'bitcoin_price', 'price_crypto', 'elon_musk_crypto',
             'cryptocurrency', 'dogecoin', 'Ethereum_ETH', 'Litecoin_LTC', 'blockchain']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,4), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               filled = True)


# ## Random Forest

# In[40]:


from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator


# In[43]:


featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(vec_df)


# In[44]:


rf = RandomForestRegressor(featuresCol="features", labelCol='last')
rfModel = rf.fit(vec_df)


# In[47]:


pred_rf = rfModel.transform(vec_df)
evaluator = RegressionEvaluator(
    labelCol="last", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(pred_rf)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


# In[ ]:




