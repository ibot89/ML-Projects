
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime


```python
# Only used in KNN
df_knn = pd.read_csv("C:/Users/Botev/Desktop/logreg/HR_comma_sep.csv")
```


```python
df = pd.read_csv("C:/Users/Botev/Desktop/logreg/HR_comma_sep.csv")
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14994</th>
      <td>0.40</td>
      <td>0.57</td>
      <td>2</td>
      <td>151</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0.37</td>
      <td>0.48</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>0.37</td>
      <td>0.53</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>0.11</td>
      <td>0.96</td>
      <td>6</td>
      <td>280</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14999 entries, 0 to 14998
    Data columns (total 10 columns):
    satisfaction_level       14999 non-null float64
    last_evaluation          14999 non-null float64
    number_project           14999 non-null int64
    average_montly_hours     14999 non-null int64
    time_spend_company       14999 non-null int64
    Work_accident            14999 non-null int64
    left                     14999 non-null int64
    promotion_last_5years    14999 non-null int64
    sales                    14999 non-null object
    salary                   14999 non-null object
    dtypes: float64(2), int64(6), object(2)
    memory usage: 1.1+ MB
    


```python

```


```python
df[(df["left"]==1) & (df["promotion_last_5years"]==0)].count()#3552 people who left didnt have a promotion in the last 5 years
```




    satisfaction_level       3552
    last_evaluation          3552
    number_project           3552
    average_montly_hours     3552
    time_spend_company       3552
    Work_accident            3552
    left                     3552
    promotion_last_5years    3552
    sales                    3552
    salary                   3552
    dtype: int64




```python

```


```python
df[(df["left"]==1) & (df["Work_accident"]==0)].count()#out of the 3571 who left 3402 didnt have a work accient 
```




    satisfaction_level       3402
    last_evaluation          3402
    number_project           3402
    average_montly_hours     3402
    time_spend_company       3402
    Work_accident            3402
    left                     3402
    promotion_last_5years    3402
    sales                    3402
    salary                   3402
    dtype: int64




```python

```


```python
df[df["left"]==1].count()#3571 left in total
```




    satisfaction_level       3571
    last_evaluation          3571
    number_project           3571
    average_montly_hours     3571
    time_spend_company       3571
    Work_accident            3571
    left                     3571
    promotion_last_5years    3571
    sales                    3571
    salary                   3571
    dtype: int64




```python

```


```python
df[df["Work_accident"]==1].count()#2169 people in total didnt have a work accident 
```




    satisfaction_level       2169
    last_evaluation          2169
    number_project           2169
    average_montly_hours     2169
    time_spend_company       2169
    Work_accident            2169
    left                     2169
    promotion_last_5years    2169
    sales                    2169
    salary                   2169
    dtype: int64




```python

```


```python
df[(df["salary"]=="low") & (df["left"]==1)].count()#2172 people with low  salaries left 
```




    satisfaction_level       2172
    last_evaluation          2172
    number_project           2172
    average_montly_hours     2172
    time_spend_company       2172
    Work_accident            2172
    left                     2172
    promotion_last_5years    2172
    sales                    2172
    salary                   2172
    dtype: int64




```python

```


```python

```


```python

```


```python

```


```python
df[(df["salary"]=="medium") & (df["left"]==1)].count()#1317 people with medium salaries left 
```




    satisfaction_level       1317
    last_evaluation          1317
    number_project           1317
    average_montly_hours     1317
    time_spend_company       1317
    Work_accident            1317
    left                     1317
    promotion_last_5years    1317
    sales                    1317
    salary                   1317
    dtype: int64




```python
df[(df["salary"]=="high") & (df["left"]==1)].count()#82 people with high salaries left 
```




    satisfaction_level       82
    last_evaluation          82
    number_project           82
    average_montly_hours     82
    time_spend_company       82
    Work_accident            82
    left                     82
    promotion_last_5years    82
    sales                    82
    salary                   82
    dtype: int64




```python

#Conclusion:
#The higher the salary the lower the chances that an employee will leave.
```


```python

```


```python

```


```python
#Plots
```


```python
df.corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>satisfaction_level</th>
      <td>1.000000</td>
      <td>0.105021</td>
      <td>-0.142970</td>
      <td>-0.020048</td>
      <td>-0.100866</td>
      <td>0.058697</td>
      <td>-0.388375</td>
      <td>0.025605</td>
    </tr>
    <tr>
      <th>last_evaluation</th>
      <td>0.105021</td>
      <td>1.000000</td>
      <td>0.349333</td>
      <td>0.339742</td>
      <td>0.131591</td>
      <td>-0.007104</td>
      <td>0.006567</td>
      <td>-0.008684</td>
    </tr>
    <tr>
      <th>number_project</th>
      <td>-0.142970</td>
      <td>0.349333</td>
      <td>1.000000</td>
      <td>0.417211</td>
      <td>0.196786</td>
      <td>-0.004741</td>
      <td>0.023787</td>
      <td>-0.006064</td>
    </tr>
    <tr>
      <th>average_montly_hours</th>
      <td>-0.020048</td>
      <td>0.339742</td>
      <td>0.417211</td>
      <td>1.000000</td>
      <td>0.127755</td>
      <td>-0.010143</td>
      <td>0.071287</td>
      <td>-0.003544</td>
    </tr>
    <tr>
      <th>time_spend_company</th>
      <td>-0.100866</td>
      <td>0.131591</td>
      <td>0.196786</td>
      <td>0.127755</td>
      <td>1.000000</td>
      <td>0.002120</td>
      <td>0.144822</td>
      <td>0.067433</td>
    </tr>
    <tr>
      <th>Work_accident</th>
      <td>0.058697</td>
      <td>-0.007104</td>
      <td>-0.004741</td>
      <td>-0.010143</td>
      <td>0.002120</td>
      <td>1.000000</td>
      <td>-0.154622</td>
      <td>0.039245</td>
    </tr>
    <tr>
      <th>left</th>
      <td>-0.388375</td>
      <td>0.006567</td>
      <td>0.023787</td>
      <td>0.071287</td>
      <td>0.144822</td>
      <td>-0.154622</td>
      <td>1.000000</td>
      <td>-0.061788</td>
    </tr>
    <tr>
      <th>promotion_last_5years</th>
      <td>0.025605</td>
      <td>-0.008684</td>
      <td>-0.006064</td>
      <td>-0.003544</td>
      <td>0.067433</td>
      <td>0.039245</td>
      <td>-0.061788</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
correlation = df.corr()#how are each of the features related to each other 
```


```python
plt.figure(figsize=(10,10))
sns.heatmap(correlation,vmax=1, square=True,annot=True,cmap='cubehelix')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xf568da0>




![png](output_29_1.png)



```python
sns.factorplot(x="left",y="average_montly_hours",data=df,kind="bar")
#little to no difference between the montly hours. Those who left worked a bit more 
```




    <seaborn.axisgrid.FacetGrid at 0x10837358>




![png](output_30_1.png)



```python
sns.factorplot(x="left",y="satisfaction_level",data=df,kind="bar")
#deffinate relatonship. Those who left had a significantly lower satisfaction level compared to those who didnt
```




    <seaborn.axisgrid.FacetGrid at 0xa634d68>




![png](output_31_1.png)



```python
sns.factorplot(x="left",y="promotion_last_5years",data=df,kind="bar")
#Deffinate relationship. 
```




    <seaborn.axisgrid.FacetGrid at 0x13bfb3c8>




![png](output_32_1.png)



```python
#Peple who were promoted after 5 years are considerably less likelly to leave than those who werent. 
#Those who's satisfaction level was higher were less likelly to stay 
```


```python
g = sns.FacetGrid(df, col = 'left')
g.map(sns.boxplot, 'time_spend_company')
```




    <seaborn.axisgrid.FacetGrid at 0x110bbef0>




![png](output_34_1.png)



```python
g = sns.FacetGrid(df, col = 'left')
g.map(sns.boxplot, 'average_montly_hours')
```




    <seaborn.axisgrid.FacetGrid at 0x11e5ee80>




![png](output_35_1.png)



```python
#shows that those who didnt leave worked around 190 hours a month. Those who left worked more. 
```


```python
g = sns.FacetGrid(df, col = 'left')
g.map(sns.boxplot, 'number_project')
```




    <seaborn.axisgrid.FacetGrid at 0x13960ac8>




![png](output_37_1.png)



```python
#Shows that those who didnt leave had between 3 and 4 projects. 
```


```python
sns.factorplot(x= 'time_spend_company', y = 'left', data = df, size = 10)
#Shows that those who had worked for exactly 5 years who the most prone leaving.
```




    <seaborn.axisgrid.FacetGrid at 0x11edf5f8>




![png](output_39_1.png)



```python
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.barplot(x="sales",y="time_spend_company",hue="left",data=df,ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x133de4a8>




![png](output_40_1.png)



```python

```


```python
# Logistic regression - sklearn
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Normalise all numerical columns 
```


```python
df["sat_level_norm"] = (df["satisfaction_level"] - df["satisfaction_level"].mean())/(df["satisfaction_level"].max()-df["satisfaction_level"].min())
```


```python
df["last_evaluation_norm"] = (df["last_evaluation"] - df["last_evaluation"].mean())/(df["last_evaluation"].max()-df["last_evaluation"].min())
```


```python
df["average_montly_hours_norm"] = (df["average_montly_hours"] - df["average_montly_hours"].mean())/(df["average_montly_hours"].max()-df["average_montly_hours"].min())
```


```python
df["time_spend_company_norm"] = (df["time_spend_company"] - df["time_spend_company"].mean())/(df["time_spend_company"].max()-df["time_spend_company"].min())
```


```python
df["number_project_norm"] = (df["number_project"] - df["number_project"].mean())/(df["number_project"].max()-df["number_project"].min())
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
      <th>sat_level_norm</th>
      <th>last_evaluation_norm</th>
      <th>average_montly_hours_norm</th>
      <th>time_spend_company_norm</th>
      <th>number_project_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.290784</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.205677</td>
      <td>0.224841</td>
      <td>0.284812</td>
      <td>0.312721</td>
      <td>0.239389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.256091</td>
      <td>0.331540</td>
      <td>0.062721</td>
      <td>0.639389</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.117765</td>
      <td>0.240466</td>
      <td>0.102569</td>
      <td>0.187721</td>
      <td>0.239389</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["sales_by_number"] = df["sales"].apply(lambda x: 1 if x=="sales" else None)
```


```python
#encode the sales column 
```


```python
def job_sub_num(j_title):
    if j_title=="sales":
        return 1
    elif j_title=="accounting":
        return 2
    elif j_title=="hr":
        return 3
    elif j_title=="technical":
        return 4
    elif j_title=="support":
        return 5
    elif j_title=="management":
        return 6
    elif j_title=="IT":
        return 7
    elif j_title=="product_mng":
        return 8
    elif j_title=="marketing":
        return 9
    else:
        return 10
```


```python
df["sales_by_number2"] = df["sales"].apply(job_sub_num)
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
      <th>sat_level_norm</th>
      <th>last_evaluation_norm</th>
      <th>average_montly_hours_norm</th>
      <th>time_spend_company_norm</th>
      <th>number_project_norm</th>
      <th>sales_by_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.290784</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.205677</td>
      <td>0.224841</td>
      <td>0.284812</td>
      <td>0.312721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.256091</td>
      <td>0.331540</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.117765</td>
      <td>0.240466</td>
      <td>0.102569</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.337659</td>
      <td>-0.224534</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.10</td>
      <td>0.77</td>
      <td>6</td>
      <td>247</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.563553</td>
      <td>0.084216</td>
      <td>0.214718</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.92</td>
      <td>0.85</td>
      <td>5</td>
      <td>259</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.337546</td>
      <td>0.209216</td>
      <td>0.270793</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.89</td>
      <td>1.00</td>
      <td>5</td>
      <td>224</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.443591</td>
      <td>0.107241</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.42</td>
      <td>0.53</td>
      <td>2</td>
      <td>142</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.211905</td>
      <td>-0.290784</td>
      <td>-0.275936</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.54</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.275159</td>
      <td>-0.308646</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.11</td>
      <td>0.81</td>
      <td>6</td>
      <td>305</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.146716</td>
      <td>0.485746</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.84</td>
      <td>0.92</td>
      <td>4</td>
      <td>234</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.249633</td>
      <td>0.318591</td>
      <td>0.153970</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.41</td>
      <td>0.55</td>
      <td>2</td>
      <td>148</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.259534</td>
      <td>-0.247899</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.36</td>
      <td>0.56</td>
      <td>2</td>
      <td>137</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.277839</td>
      <td>-0.243909</td>
      <td>-0.299301</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.38</td>
      <td>0.54</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.275159</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.45</td>
      <td>0.47</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.384534</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.78</td>
      <td>0.99</td>
      <td>4</td>
      <td>255</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.183699</td>
      <td>0.427966</td>
      <td>0.252101</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.45</td>
      <td>0.51</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.322034</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.76</td>
      <td>0.89</td>
      <td>5</td>
      <td>262</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.161721</td>
      <td>0.271716</td>
      <td>0.284812</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.11</td>
      <td>0.83</td>
      <td>6</td>
      <td>282</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.177966</td>
      <td>0.378269</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.38</td>
      <td>0.55</td>
      <td>2</td>
      <td>147</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.259534</td>
      <td>-0.252572</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.09</td>
      <td>0.95</td>
      <td>6</td>
      <td>304</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.574542</td>
      <td>0.365466</td>
      <td>0.481073</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.46</td>
      <td>0.57</td>
      <td>2</td>
      <td>139</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.167949</td>
      <td>-0.228284</td>
      <td>-0.289955</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.40</td>
      <td>0.53</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.290784</td>
      <td>-0.201170</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.89</td>
      <td>0.92</td>
      <td>5</td>
      <td>242</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.318591</td>
      <td>0.191354</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.82</td>
      <td>0.87</td>
      <td>4</td>
      <td>239</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.227655</td>
      <td>0.240466</td>
      <td>0.177335</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.40</td>
      <td>0.49</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.353284</td>
      <td>-0.308646</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.41</td>
      <td>0.46</td>
      <td>2</td>
      <td>128</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.400159</td>
      <td>-0.341357</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.38</td>
      <td>0.50</td>
      <td>2</td>
      <td>132</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.337659</td>
      <td>-0.322665</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14969</th>
      <td>0.43</td>
      <td>0.46</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.200916</td>
      <td>-0.400159</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14970</th>
      <td>0.78</td>
      <td>0.93</td>
      <td>4</td>
      <td>225</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.183699</td>
      <td>0.334216</td>
      <td>0.111914</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14971</th>
      <td>0.39</td>
      <td>0.45</td>
      <td>2</td>
      <td>140</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.244872</td>
      <td>-0.415784</td>
      <td>-0.285282</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14972</th>
      <td>0.11</td>
      <td>0.97</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.396716</td>
      <td>0.509111</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14973</th>
      <td>0.36</td>
      <td>0.52</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.277839</td>
      <td>-0.306409</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14974</th>
      <td>0.36</td>
      <td>0.54</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.277839</td>
      <td>-0.275159</td>
      <td>-0.224534</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14975</th>
      <td>0.10</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.563553</td>
      <td>0.115466</td>
      <td>0.509111</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14976</th>
      <td>0.40</td>
      <td>0.47</td>
      <td>2</td>
      <td>136</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.384534</td>
      <td>-0.303974</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14977</th>
      <td>0.81</td>
      <td>0.85</td>
      <td>4</td>
      <td>251</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>0.216666</td>
      <td>0.209216</td>
      <td>0.233410</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14978</th>
      <td>0.40</td>
      <td>0.47</td>
      <td>2</td>
      <td>144</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.384534</td>
      <td>-0.266590</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14979</th>
      <td>0.09</td>
      <td>0.93</td>
      <td>6</td>
      <td>296</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.574542</td>
      <td>0.334216</td>
      <td>0.443690</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14980</th>
      <td>0.76</td>
      <td>0.89</td>
      <td>5</td>
      <td>238</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>high</td>
      <td>0.161721</td>
      <td>0.271716</td>
      <td>0.172662</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14981</th>
      <td>0.73</td>
      <td>0.93</td>
      <td>5</td>
      <td>162</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.128754</td>
      <td>0.334216</td>
      <td>-0.182478</td>
      <td>0.062721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14982</th>
      <td>0.38</td>
      <td>0.49</td>
      <td>2</td>
      <td>137</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.255861</td>
      <td>-0.353284</td>
      <td>-0.299301</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14983</th>
      <td>0.72</td>
      <td>0.84</td>
      <td>5</td>
      <td>257</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>0.117765</td>
      <td>0.193591</td>
      <td>0.261447</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14984</th>
      <td>0.40</td>
      <td>0.56</td>
      <td>2</td>
      <td>148</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.243909</td>
      <td>-0.247899</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14985</th>
      <td>0.91</td>
      <td>0.99</td>
      <td>5</td>
      <td>254</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>0.326557</td>
      <td>0.427966</td>
      <td>0.247428</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14986</th>
      <td>0.85</td>
      <td>0.85</td>
      <td>4</td>
      <td>247</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.260623</td>
      <td>0.209216</td>
      <td>0.214718</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14987</th>
      <td>0.90</td>
      <td>0.70</td>
      <td>5</td>
      <td>206</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.315568</td>
      <td>-0.025159</td>
      <td>0.023129</td>
      <td>0.062721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14988</th>
      <td>0.46</td>
      <td>0.55</td>
      <td>2</td>
      <td>145</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>-0.167949</td>
      <td>-0.259534</td>
      <td>-0.261917</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14989</th>
      <td>0.43</td>
      <td>0.57</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>-0.200916</td>
      <td>-0.228284</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14990</th>
      <td>0.89</td>
      <td>0.88</td>
      <td>5</td>
      <td>228</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.256091</td>
      <td>0.125933</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14991</th>
      <td>0.09</td>
      <td>0.81</td>
      <td>6</td>
      <td>257</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.574542</td>
      <td>0.146716</td>
      <td>0.261447</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14992</th>
      <td>0.40</td>
      <td>0.48</td>
      <td>2</td>
      <td>155</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.368909</td>
      <td>-0.215188</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14993</th>
      <td>0.76</td>
      <td>0.83</td>
      <td>6</td>
      <td>293</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>0.161721</td>
      <td>0.177966</td>
      <td>0.429671</td>
      <td>0.312721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14994</th>
      <td>0.40</td>
      <td>0.57</td>
      <td>2</td>
      <td>151</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.228284</td>
      <td>-0.233880</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0.37</td>
      <td>0.48</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.368909</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>0.37</td>
      <td>0.53</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.290784</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>0.11</td>
      <td>0.96</td>
      <td>6</td>
      <td>280</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.381091</td>
      <td>0.368924</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.201170</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>14999 rows × 16 columns</p>
</div>




```python
df.drop("sales_by_number",axis=1,inplace = True)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-10-15ae52fe21e2> in <module>()
    ----> 1 df.drop("sales_by_number",axis=1,inplace = True)
    

    C:\Users\Botev\Anaconda\lib\site-packages\pandas\core\generic.pyc in drop(self, labels, axis, level, inplace, errors)
       1905                 new_axis = axis.drop(labels, level=level, errors=errors)
       1906             else:
    -> 1907                 new_axis = axis.drop(labels, errors=errors)
       1908             dropped = self.reindex(**{axis_name: new_axis})
       1909             try:
    

    C:\Users\Botev\Anaconda\lib\site-packages\pandas\indexes\base.pyc in drop(self, labels, errors)
       3260             if errors != 'ignore':
       3261                 raise ValueError('labels %s not contained in axis' %
    -> 3262                                  labels[mask])
       3263             indexer = indexer[~mask]
       3264         return self.delete(indexer)
    

    ValueError: labels ['sales_by_number'] not contained in axis



```python
df.rename(columns={'sales_by_number2':'sales_by_number'},inplace = True)
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
      <th>sat_level_norm</th>
      <th>last_evaluation_norm</th>
      <th>average_montly_hours_norm</th>
      <th>time_spend_company_norm</th>
      <th>number_project_norm</th>
      <th>sales_by_number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.290784</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.205677</td>
      <td>0.224841</td>
      <td>0.284812</td>
      <td>0.312721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.256091</td>
      <td>0.331540</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.117765</td>
      <td>0.240466</td>
      <td>0.102569</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.337659</td>
      <td>-0.224534</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.10</td>
      <td>0.77</td>
      <td>6</td>
      <td>247</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.563553</td>
      <td>0.084216</td>
      <td>0.214718</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.92</td>
      <td>0.85</td>
      <td>5</td>
      <td>259</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.337546</td>
      <td>0.209216</td>
      <td>0.270793</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.89</td>
      <td>1.00</td>
      <td>5</td>
      <td>224</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.443591</td>
      <td>0.107241</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.42</td>
      <td>0.53</td>
      <td>2</td>
      <td>142</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.211905</td>
      <td>-0.290784</td>
      <td>-0.275936</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.54</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.275159</td>
      <td>-0.308646</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.11</td>
      <td>0.81</td>
      <td>6</td>
      <td>305</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.146716</td>
      <td>0.485746</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.84</td>
      <td>0.92</td>
      <td>4</td>
      <td>234</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.249633</td>
      <td>0.318591</td>
      <td>0.153970</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.41</td>
      <td>0.55</td>
      <td>2</td>
      <td>148</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.259534</td>
      <td>-0.247899</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.36</td>
      <td>0.56</td>
      <td>2</td>
      <td>137</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.277839</td>
      <td>-0.243909</td>
      <td>-0.299301</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.38</td>
      <td>0.54</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.275159</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.45</td>
      <td>0.47</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.384534</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.78</td>
      <td>0.99</td>
      <td>4</td>
      <td>255</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.183699</td>
      <td>0.427966</td>
      <td>0.252101</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.45</td>
      <td>0.51</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.322034</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.76</td>
      <td>0.89</td>
      <td>5</td>
      <td>262</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.161721</td>
      <td>0.271716</td>
      <td>0.284812</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.11</td>
      <td>0.83</td>
      <td>6</td>
      <td>282</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.177966</td>
      <td>0.378269</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.38</td>
      <td>0.55</td>
      <td>2</td>
      <td>147</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.259534</td>
      <td>-0.252572</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.09</td>
      <td>0.95</td>
      <td>6</td>
      <td>304</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.574542</td>
      <td>0.365466</td>
      <td>0.481073</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.46</td>
      <td>0.57</td>
      <td>2</td>
      <td>139</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.167949</td>
      <td>-0.228284</td>
      <td>-0.289955</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.40</td>
      <td>0.53</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.290784</td>
      <td>-0.201170</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.89</td>
      <td>0.92</td>
      <td>5</td>
      <td>242</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.318591</td>
      <td>0.191354</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.82</td>
      <td>0.87</td>
      <td>4</td>
      <td>239</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.227655</td>
      <td>0.240466</td>
      <td>0.177335</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.40</td>
      <td>0.49</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.353284</td>
      <td>-0.308646</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.41</td>
      <td>0.46</td>
      <td>2</td>
      <td>128</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.400159</td>
      <td>-0.341357</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.38</td>
      <td>0.50</td>
      <td>2</td>
      <td>132</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.337659</td>
      <td>-0.322665</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14969</th>
      <td>0.43</td>
      <td>0.46</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.200916</td>
      <td>-0.400159</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14970</th>
      <td>0.78</td>
      <td>0.93</td>
      <td>4</td>
      <td>225</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.183699</td>
      <td>0.334216</td>
      <td>0.111914</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14971</th>
      <td>0.39</td>
      <td>0.45</td>
      <td>2</td>
      <td>140</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.244872</td>
      <td>-0.415784</td>
      <td>-0.285282</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14972</th>
      <td>0.11</td>
      <td>0.97</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.396716</td>
      <td>0.509111</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14973</th>
      <td>0.36</td>
      <td>0.52</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.277839</td>
      <td>-0.306409</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14974</th>
      <td>0.36</td>
      <td>0.54</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.277839</td>
      <td>-0.275159</td>
      <td>-0.224534</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14975</th>
      <td>0.10</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.563553</td>
      <td>0.115466</td>
      <td>0.509111</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14976</th>
      <td>0.40</td>
      <td>0.47</td>
      <td>2</td>
      <td>136</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.384534</td>
      <td>-0.303974</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14977</th>
      <td>0.81</td>
      <td>0.85</td>
      <td>4</td>
      <td>251</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>0.216666</td>
      <td>0.209216</td>
      <td>0.233410</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14978</th>
      <td>0.40</td>
      <td>0.47</td>
      <td>2</td>
      <td>144</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.384534</td>
      <td>-0.266590</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14979</th>
      <td>0.09</td>
      <td>0.93</td>
      <td>6</td>
      <td>296</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.574542</td>
      <td>0.334216</td>
      <td>0.443690</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14980</th>
      <td>0.76</td>
      <td>0.89</td>
      <td>5</td>
      <td>238</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>high</td>
      <td>0.161721</td>
      <td>0.271716</td>
      <td>0.172662</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14981</th>
      <td>0.73</td>
      <td>0.93</td>
      <td>5</td>
      <td>162</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.128754</td>
      <td>0.334216</td>
      <td>-0.182478</td>
      <td>0.062721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14982</th>
      <td>0.38</td>
      <td>0.49</td>
      <td>2</td>
      <td>137</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.255861</td>
      <td>-0.353284</td>
      <td>-0.299301</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14983</th>
      <td>0.72</td>
      <td>0.84</td>
      <td>5</td>
      <td>257</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>0.117765</td>
      <td>0.193591</td>
      <td>0.261447</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14984</th>
      <td>0.40</td>
      <td>0.56</td>
      <td>2</td>
      <td>148</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.243909</td>
      <td>-0.247899</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14985</th>
      <td>0.91</td>
      <td>0.99</td>
      <td>5</td>
      <td>254</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>0.326557</td>
      <td>0.427966</td>
      <td>0.247428</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14986</th>
      <td>0.85</td>
      <td>0.85</td>
      <td>4</td>
      <td>247</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.260623</td>
      <td>0.209216</td>
      <td>0.214718</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14987</th>
      <td>0.90</td>
      <td>0.70</td>
      <td>5</td>
      <td>206</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.315568</td>
      <td>-0.025159</td>
      <td>0.023129</td>
      <td>0.062721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14988</th>
      <td>0.46</td>
      <td>0.55</td>
      <td>2</td>
      <td>145</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>-0.167949</td>
      <td>-0.259534</td>
      <td>-0.261917</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14989</th>
      <td>0.43</td>
      <td>0.57</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>-0.200916</td>
      <td>-0.228284</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14990</th>
      <td>0.89</td>
      <td>0.88</td>
      <td>5</td>
      <td>228</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.256091</td>
      <td>0.125933</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14991</th>
      <td>0.09</td>
      <td>0.81</td>
      <td>6</td>
      <td>257</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.574542</td>
      <td>0.146716</td>
      <td>0.261447</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14992</th>
      <td>0.40</td>
      <td>0.48</td>
      <td>2</td>
      <td>155</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.368909</td>
      <td>-0.215188</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14993</th>
      <td>0.76</td>
      <td>0.83</td>
      <td>6</td>
      <td>293</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>0.161721</td>
      <td>0.177966</td>
      <td>0.429671</td>
      <td>0.312721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14994</th>
      <td>0.40</td>
      <td>0.57</td>
      <td>2</td>
      <td>151</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.228284</td>
      <td>-0.233880</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0.37</td>
      <td>0.48</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.368909</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>0.37</td>
      <td>0.53</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.290784</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>0.11</td>
      <td>0.96</td>
      <td>6</td>
      <td>280</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.381091</td>
      <td>0.368924</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.201170</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>14999 rows × 16 columns</p>
</div>




```python
#encode the salary column 
```


```python
def salary_encode(salary):
    if salary =="low":
        return 1
    elif salary=="medium":
        return 2
    else:
        return 3
```


```python
df["salary_encoded"] = df["salary"].apply(salary_encode)
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
      <th>sat_level_norm</th>
      <th>last_evaluation_norm</th>
      <th>average_montly_hours_norm</th>
      <th>time_spend_company_norm</th>
      <th>number_project_norm</th>
      <th>sales_by_number</th>
      <th>sales_by_number</th>
      <th>salary_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.290784</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.205677</td>
      <td>0.224841</td>
      <td>0.284812</td>
      <td>0.312721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.256091</td>
      <td>0.331540</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.117765</td>
      <td>0.240466</td>
      <td>0.102569</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.337659</td>
      <td>-0.224534</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.10</td>
      <td>0.77</td>
      <td>6</td>
      <td>247</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.563553</td>
      <td>0.084216</td>
      <td>0.214718</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.92</td>
      <td>0.85</td>
      <td>5</td>
      <td>259</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.337546</td>
      <td>0.209216</td>
      <td>0.270793</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.89</td>
      <td>1.00</td>
      <td>5</td>
      <td>224</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.443591</td>
      <td>0.107241</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.42</td>
      <td>0.53</td>
      <td>2</td>
      <td>142</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.211905</td>
      <td>-0.290784</td>
      <td>-0.275936</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.54</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.275159</td>
      <td>-0.308646</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.11</td>
      <td>0.81</td>
      <td>6</td>
      <td>305</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.146716</td>
      <td>0.485746</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.84</td>
      <td>0.92</td>
      <td>4</td>
      <td>234</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.249633</td>
      <td>0.318591</td>
      <td>0.153970</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.41</td>
      <td>0.55</td>
      <td>2</td>
      <td>148</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.259534</td>
      <td>-0.247899</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.36</td>
      <td>0.56</td>
      <td>2</td>
      <td>137</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.277839</td>
      <td>-0.243909</td>
      <td>-0.299301</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.38</td>
      <td>0.54</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.275159</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.45</td>
      <td>0.47</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.384534</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.78</td>
      <td>0.99</td>
      <td>4</td>
      <td>255</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.183699</td>
      <td>0.427966</td>
      <td>0.252101</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.45</td>
      <td>0.51</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.322034</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.76</td>
      <td>0.89</td>
      <td>5</td>
      <td>262</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.161721</td>
      <td>0.271716</td>
      <td>0.284812</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.11</td>
      <td>0.83</td>
      <td>6</td>
      <td>282</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.177966</td>
      <td>0.378269</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.38</td>
      <td>0.55</td>
      <td>2</td>
      <td>147</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.259534</td>
      <td>-0.252572</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.09</td>
      <td>0.95</td>
      <td>6</td>
      <td>304</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.574542</td>
      <td>0.365466</td>
      <td>0.481073</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.46</td>
      <td>0.57</td>
      <td>2</td>
      <td>139</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.167949</td>
      <td>-0.228284</td>
      <td>-0.289955</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.40</td>
      <td>0.53</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.290784</td>
      <td>-0.201170</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.89</td>
      <td>0.92</td>
      <td>5</td>
      <td>242</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.318591</td>
      <td>0.191354</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.82</td>
      <td>0.87</td>
      <td>4</td>
      <td>239</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.227655</td>
      <td>0.240466</td>
      <td>0.177335</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.40</td>
      <td>0.49</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.353284</td>
      <td>-0.308646</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.41</td>
      <td>0.46</td>
      <td>2</td>
      <td>128</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.400159</td>
      <td>-0.341357</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.38</td>
      <td>0.50</td>
      <td>2</td>
      <td>132</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.337659</td>
      <td>-0.322665</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14969</th>
      <td>0.43</td>
      <td>0.46</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.200916</td>
      <td>-0.400159</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14970</th>
      <td>0.78</td>
      <td>0.93</td>
      <td>4</td>
      <td>225</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.183699</td>
      <td>0.334216</td>
      <td>0.111914</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14971</th>
      <td>0.39</td>
      <td>0.45</td>
      <td>2</td>
      <td>140</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.244872</td>
      <td>-0.415784</td>
      <td>-0.285282</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14972</th>
      <td>0.11</td>
      <td>0.97</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.396716</td>
      <td>0.509111</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14973</th>
      <td>0.36</td>
      <td>0.52</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.277839</td>
      <td>-0.306409</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14974</th>
      <td>0.36</td>
      <td>0.54</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.277839</td>
      <td>-0.275159</td>
      <td>-0.224534</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14975</th>
      <td>0.10</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.563553</td>
      <td>0.115466</td>
      <td>0.509111</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14976</th>
      <td>0.40</td>
      <td>0.47</td>
      <td>2</td>
      <td>136</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.384534</td>
      <td>-0.303974</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14977</th>
      <td>0.81</td>
      <td>0.85</td>
      <td>4</td>
      <td>251</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>0.216666</td>
      <td>0.209216</td>
      <td>0.233410</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14978</th>
      <td>0.40</td>
      <td>0.47</td>
      <td>2</td>
      <td>144</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.384534</td>
      <td>-0.266590</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14979</th>
      <td>0.09</td>
      <td>0.93</td>
      <td>6</td>
      <td>296</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.574542</td>
      <td>0.334216</td>
      <td>0.443690</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14980</th>
      <td>0.76</td>
      <td>0.89</td>
      <td>5</td>
      <td>238</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>high</td>
      <td>0.161721</td>
      <td>0.271716</td>
      <td>0.172662</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14981</th>
      <td>0.73</td>
      <td>0.93</td>
      <td>5</td>
      <td>162</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.128754</td>
      <td>0.334216</td>
      <td>-0.182478</td>
      <td>0.062721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14982</th>
      <td>0.38</td>
      <td>0.49</td>
      <td>2</td>
      <td>137</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.255861</td>
      <td>-0.353284</td>
      <td>-0.299301</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14983</th>
      <td>0.72</td>
      <td>0.84</td>
      <td>5</td>
      <td>257</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>0.117765</td>
      <td>0.193591</td>
      <td>0.261447</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14984</th>
      <td>0.40</td>
      <td>0.56</td>
      <td>2</td>
      <td>148</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.243909</td>
      <td>-0.247899</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14985</th>
      <td>0.91</td>
      <td>0.99</td>
      <td>5</td>
      <td>254</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>0.326557</td>
      <td>0.427966</td>
      <td>0.247428</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14986</th>
      <td>0.85</td>
      <td>0.85</td>
      <td>4</td>
      <td>247</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.260623</td>
      <td>0.209216</td>
      <td>0.214718</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>NaN</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14987</th>
      <td>0.90</td>
      <td>0.70</td>
      <td>5</td>
      <td>206</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.315568</td>
      <td>-0.025159</td>
      <td>0.023129</td>
      <td>0.062721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14988</th>
      <td>0.46</td>
      <td>0.55</td>
      <td>2</td>
      <td>145</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>-0.167949</td>
      <td>-0.259534</td>
      <td>-0.261917</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14989</th>
      <td>0.43</td>
      <td>0.57</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>-0.200916</td>
      <td>-0.228284</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14990</th>
      <td>0.89</td>
      <td>0.88</td>
      <td>5</td>
      <td>228</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.256091</td>
      <td>0.125933</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14991</th>
      <td>0.09</td>
      <td>0.81</td>
      <td>6</td>
      <td>257</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.574542</td>
      <td>0.146716</td>
      <td>0.261447</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14992</th>
      <td>0.40</td>
      <td>0.48</td>
      <td>2</td>
      <td>155</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.368909</td>
      <td>-0.215188</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14993</th>
      <td>0.76</td>
      <td>0.83</td>
      <td>6</td>
      <td>293</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>0.161721</td>
      <td>0.177966</td>
      <td>0.429671</td>
      <td>0.312721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14994</th>
      <td>0.40</td>
      <td>0.57</td>
      <td>2</td>
      <td>151</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.228284</td>
      <td>-0.233880</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0.37</td>
      <td>0.48</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.368909</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>0.37</td>
      <td>0.53</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.290784</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>0.11</td>
      <td>0.96</td>
      <td>6</td>
      <td>280</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.381091</td>
      <td>0.368924</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.201170</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>14999 rows × 18 columns</p>
</div>




```python

```


```python

```


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
```


```python
X = df[["Work_accident","promotion_last_5years","sat_level_norm","last_evaluation_norm","average_montly_hours_norm","time_spend_company_norm","sales_by_number","number_project_norm","salary_encoded"]]
Y = df["left"]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
```


```python
logmodel = LogisticRegression(C=100)
logmodel.fit(X_train,y_train)
```




    LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
predictions = logmodel.predict(X_test)
```


```python
print(classification_report(y_test,predictions))#f1 score weighted average of the precision and recall
```

                 precision    recall  f1-score   support
    
              0       0.82      0.93      0.87      4588
              1       0.60      0.36      0.45      1412
    
    avg / total       0.77      0.79      0.77      6000
    
    


```python
score = logmodel.score(X_test, y_test)
```


```python
print "Log regression score is:%0.4f"%(score)
```

    Log regression score is:0.7915
    


```python
#no normalising 
X1 = df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","sales_by_number","salary_encoded"]]
Y1 = df["left"]
```


```python
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.4, random_state=42)
```


```python
logmodel = LogisticRegression(C=100)
logmodel.fit(X_train1,y_train1)
```




    LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
predictions1 = logmodel.predict(X_test1)
```


```python
print(classification_report(y_test1,predictions1))
```

                 precision    recall  f1-score   support
    
              0       0.83      0.92      0.87      4595
              1       0.59      0.37      0.45      1405
    
    avg / total       0.77      0.79      0.77      6000
    
    


```python
print "Log regression score is:%0.4f"%(score)
```

    Log regression score is:0.7923
    


```python
accuracy_score(y_test,predictions)#same as score 
```




    0.79233333333333333




```python
print('Variance score: %.2f' % r2_score(y_test1,predictions1))
#negative r2 - means that the model is even worse than the worst model assumed (which is the absolute mean model).
```

    Variance score: -0.16
    


```python
d1_map = y_test - predictions
d2_map = y_test - y_test.mean()
r2_map = 1 - d1_map.dot(d1_map) / d2_map.dot(d2_map)
```


```python
r2_map
```




    -0.15864516622992464




```python
#r squared cannot be used for logistic regression. Other versions of r squared exist
```


```python
plt.scatter(y_test,predictions)#Not important 
```




    <matplotlib.collections.PathCollection at 0x12d1ac18>




![png](output_84_1.png)



```python
#drop the encoded columns 
X2 = df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years"]]
Y2 = df["left"]
```


```python
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.4, random_state=42)
```


```python
logmodel = LogisticRegression(C=100)
logmodel.fit(X_train2,y_train2)
```




    LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
predictions2 = logmodel.predict(X_test2)
```


```python
print(classification_report(y_test1,predictions1))
```

                 precision    recall  f1-score   support
    
              0       0.83      0.92      0.87      4595
              1       0.59      0.37      0.45      1405
    
    avg / total       0.77      0.79      0.77      6000
    
    


```python
print "Log regression score is:%0.4f"%(score)
```

    Log regression score is:0.7915
    


```python
print('Variance score: %.2f' % r2_score(y_test2,predictions2))
```

    Variance score: -0.31
    


```python
plt.scatter(y_test2,predictions2)
```




    <matplotlib.collections.PathCollection at 0x11a1d080>




![png](output_92_1.png)



```python
#Log reg no sklearn - grad descent
X_new = df[["satisfaction_level","last_evaluation","Work_accident","promotion_last_5years","number_project_norm","time_spend_company_norm","average_montly_hours_norm","salary_encoded"]]
Y_new = df["left"]
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
      <th>sat_level_norm</th>
      <th>last_evaluation_norm</th>
      <th>average_montly_hours_norm</th>
      <th>time_spend_company_norm</th>
      <th>number_project_norm</th>
      <th>sales_by_number</th>
      <th>salary_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.290784</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.205677</td>
      <td>0.224841</td>
      <td>0.284812</td>
      <td>0.312721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.256091</td>
      <td>0.331540</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.117765</td>
      <td>0.240466</td>
      <td>0.102569</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.41</td>
      <td>0.50</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.337659</td>
      <td>-0.224534</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.10</td>
      <td>0.77</td>
      <td>6</td>
      <td>247</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.563553</td>
      <td>0.084216</td>
      <td>0.214718</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.92</td>
      <td>0.85</td>
      <td>5</td>
      <td>259</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.337546</td>
      <td>0.209216</td>
      <td>0.270793</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.89</td>
      <td>1.00</td>
      <td>5</td>
      <td>224</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.443591</td>
      <td>0.107241</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.42</td>
      <td>0.53</td>
      <td>2</td>
      <td>142</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.211905</td>
      <td>-0.290784</td>
      <td>-0.275936</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.45</td>
      <td>0.54</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.275159</td>
      <td>-0.308646</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.11</td>
      <td>0.81</td>
      <td>6</td>
      <td>305</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.146716</td>
      <td>0.485746</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.84</td>
      <td>0.92</td>
      <td>4</td>
      <td>234</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.249633</td>
      <td>0.318591</td>
      <td>0.153970</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.41</td>
      <td>0.55</td>
      <td>2</td>
      <td>148</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.259534</td>
      <td>-0.247899</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.36</td>
      <td>0.56</td>
      <td>2</td>
      <td>137</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.277839</td>
      <td>-0.243909</td>
      <td>-0.299301</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.38</td>
      <td>0.54</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.275159</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.45</td>
      <td>0.47</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.384534</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.78</td>
      <td>0.99</td>
      <td>4</td>
      <td>255</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.183699</td>
      <td>0.427966</td>
      <td>0.252101</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.45</td>
      <td>0.51</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.178938</td>
      <td>-0.322034</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.76</td>
      <td>0.89</td>
      <td>5</td>
      <td>262</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.161721</td>
      <td>0.271716</td>
      <td>0.284812</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.11</td>
      <td>0.83</td>
      <td>6</td>
      <td>282</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.177966</td>
      <td>0.378269</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.38</td>
      <td>0.55</td>
      <td>2</td>
      <td>147</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.259534</td>
      <td>-0.252572</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.09</td>
      <td>0.95</td>
      <td>6</td>
      <td>304</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.574542</td>
      <td>0.365466</td>
      <td>0.481073</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.46</td>
      <td>0.57</td>
      <td>2</td>
      <td>139</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.167949</td>
      <td>-0.228284</td>
      <td>-0.289955</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.40</td>
      <td>0.53</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.290784</td>
      <td>-0.201170</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.89</td>
      <td>0.92</td>
      <td>5</td>
      <td>242</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.318591</td>
      <td>0.191354</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.82</td>
      <td>0.87</td>
      <td>4</td>
      <td>239</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.227655</td>
      <td>0.240466</td>
      <td>0.177335</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.40</td>
      <td>0.49</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.353284</td>
      <td>-0.308646</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.41</td>
      <td>0.46</td>
      <td>2</td>
      <td>128</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
      <td>-0.222894</td>
      <td>-0.400159</td>
      <td>-0.341357</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.38</td>
      <td>0.50</td>
      <td>2</td>
      <td>132</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.337659</td>
      <td>-0.322665</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14969</th>
      <td>0.43</td>
      <td>0.46</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.200916</td>
      <td>-0.400159</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14970</th>
      <td>0.78</td>
      <td>0.93</td>
      <td>4</td>
      <td>225</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.183699</td>
      <td>0.334216</td>
      <td>0.111914</td>
      <td>0.187721</td>
      <td>0.039389</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14971</th>
      <td>0.39</td>
      <td>0.45</td>
      <td>2</td>
      <td>140</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.244872</td>
      <td>-0.415784</td>
      <td>-0.285282</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14972</th>
      <td>0.11</td>
      <td>0.97</td>
      <td>6</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.396716</td>
      <td>0.509111</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14973</th>
      <td>0.36</td>
      <td>0.52</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.277839</td>
      <td>-0.306409</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14974</th>
      <td>0.36</td>
      <td>0.54</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
      <td>-0.277839</td>
      <td>-0.275159</td>
      <td>-0.224534</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14975</th>
      <td>0.10</td>
      <td>0.79</td>
      <td>7</td>
      <td>310</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.563553</td>
      <td>0.115466</td>
      <td>0.509111</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14976</th>
      <td>0.40</td>
      <td>0.47</td>
      <td>2</td>
      <td>136</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.384534</td>
      <td>-0.303974</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14977</th>
      <td>0.81</td>
      <td>0.85</td>
      <td>4</td>
      <td>251</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>0.216666</td>
      <td>0.209216</td>
      <td>0.233410</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14978</th>
      <td>0.40</td>
      <td>0.47</td>
      <td>2</td>
      <td>144</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>hr</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.384534</td>
      <td>-0.266590</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14979</th>
      <td>0.09</td>
      <td>0.93</td>
      <td>6</td>
      <td>296</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.574542</td>
      <td>0.334216</td>
      <td>0.443690</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14980</th>
      <td>0.76</td>
      <td>0.89</td>
      <td>5</td>
      <td>238</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>high</td>
      <td>0.161721</td>
      <td>0.271716</td>
      <td>0.172662</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14981</th>
      <td>0.73</td>
      <td>0.93</td>
      <td>5</td>
      <td>162</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.128754</td>
      <td>0.334216</td>
      <td>-0.182478</td>
      <td>0.062721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14982</th>
      <td>0.38</td>
      <td>0.49</td>
      <td>2</td>
      <td>137</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.255861</td>
      <td>-0.353284</td>
      <td>-0.299301</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14983</th>
      <td>0.72</td>
      <td>0.84</td>
      <td>5</td>
      <td>257</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>0.117765</td>
      <td>0.193591</td>
      <td>0.261447</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14984</th>
      <td>0.40</td>
      <td>0.56</td>
      <td>2</td>
      <td>148</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>-0.233883</td>
      <td>-0.243909</td>
      <td>-0.247899</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14985</th>
      <td>0.91</td>
      <td>0.99</td>
      <td>5</td>
      <td>254</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
      <td>0.326557</td>
      <td>0.427966</td>
      <td>0.247428</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14986</th>
      <td>0.85</td>
      <td>0.85</td>
      <td>4</td>
      <td>247</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.260623</td>
      <td>0.209216</td>
      <td>0.214718</td>
      <td>0.312721</td>
      <td>0.039389</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14987</th>
      <td>0.90</td>
      <td>0.70</td>
      <td>5</td>
      <td>206</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>0.315568</td>
      <td>-0.025159</td>
      <td>0.023129</td>
      <td>0.062721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14988</th>
      <td>0.46</td>
      <td>0.55</td>
      <td>2</td>
      <td>145</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>-0.167949</td>
      <td>-0.259534</td>
      <td>-0.261917</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14989</th>
      <td>0.43</td>
      <td>0.57</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>-0.200916</td>
      <td>-0.228284</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14990</th>
      <td>0.89</td>
      <td>0.88</td>
      <td>5</td>
      <td>228</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>0.304579</td>
      <td>0.256091</td>
      <td>0.125933</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14991</th>
      <td>0.09</td>
      <td>0.81</td>
      <td>6</td>
      <td>257</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.574542</td>
      <td>0.146716</td>
      <td>0.261447</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14992</th>
      <td>0.40</td>
      <td>0.48</td>
      <td>2</td>
      <td>155</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.368909</td>
      <td>-0.215188</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14993</th>
      <td>0.76</td>
      <td>0.83</td>
      <td>6</td>
      <td>293</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>0.161721</td>
      <td>0.177966</td>
      <td>0.429671</td>
      <td>0.312721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14994</th>
      <td>0.40</td>
      <td>0.57</td>
      <td>2</td>
      <td>151</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.233883</td>
      <td>-0.228284</td>
      <td>-0.233880</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0.37</td>
      <td>0.48</td>
      <td>2</td>
      <td>160</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.368909</td>
      <td>-0.191824</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14996</th>
      <td>0.37</td>
      <td>0.53</td>
      <td>2</td>
      <td>143</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.290784</td>
      <td>-0.271263</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>0.11</td>
      <td>0.96</td>
      <td>6</td>
      <td>280</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.552564</td>
      <td>0.381091</td>
      <td>0.368924</td>
      <td>0.062721</td>
      <td>0.439389</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14998</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>158</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.201170</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>14999 rows × 17 columns</p>
</div>




```python
#function to perform one hot encoding for the salary column. The sales column is droped.
#same encoding can be used to deal with the sales column as well. 
def encode(X,Y):
    X_mat = X.as_matrix()#conver to matrixes 
    Y_mat = Y.as_matrix()
    N,D = X_mat.shape
    X2 = np.zeros((N,D+3))# introducing a second matrix. This matrix has 3 more column than the original so that the encoding 
    #can be performed
    X2[:,0:D-1] = X_mat[:,0:D-1]
    
    
    for n in xrange(N):
        i=int(X_mat[n,D-1])
        X2[n,D-1+i]=1
    
    return X2,Y_mat
```


```python
X_enc,Y_enc = encode(X_new,Y_new)
```


```python
X_enc,Y_enc = shuffle(X_enc,Y_enc)
```


```python
X_enc.shape
```




    (14999L, 11L)




```python
len(X_enc)
```




    14999




```python
D = X_enc.shape[1]
```


```python
w = np.random.rand(D)
b=0
```


```python
#spliting into training and testing sets.
X_train = X_enc[:-4500]
Y_train = Y_enc[:-4500]
X_test = X_enc[-4500:]
Y_test = Y_enc[-4500:]
```


```python

```


```python

```


```python
def sigmoid(z):
    return 1/(1+np.exp(-z))
```


```python
def forward(X,w,b):
    return sigmoid(X.dot(w)+b)
```


```python
#gives you an idea of how well the algorithm is performing 
def classification_rate(Y,P):
    return np.mean(Y==P)
```


```python
# in log regression cross entropy is essentially the error that the algorithm has
def cross_entropy(T,py):
    return -np.mean(T*np.log(py)+(1-T)*np.log(1-py))
```


```python
learning_rate = 0.0001
training_cost = []
test_cost = []
```


```python
for i in xrange(50000):
    py_train = forward(X_train,w,b)
    py_test = forward(X_test,w,b)
    ctrain = cross_entropy(Y_train,py_train)
    ctest = cross_entropy(Y_test,py_test)
    training_cost.append(ctrain)
    test_cost.append(ctest)
    
    
    
    w -= learning_rate*X_train.T.dot(py_train - Y_train)
    b -= learning_rate*(py_train - Y_train).sum()
    
    if i%1000==0:
        print i,ctrain,ctest
    
    
print "Final Classification training rate:",classification_rate(Y_train,np.round(py_train))
print "Final Classification Testing rate:",classification_rate(Y_test,np.round(py_test))
    
legend1, = plt.plot(training_cost,label = "training_cost")
legend2, = plt.plot(test_cost,label = "test_cost")
plt.legend([legend1,legend2])
plt.show()
```

    0 1.07781599159 1.07577401016
    1000 0.428028896518 0.436562750264
    2000 0.42792930709 0.436286515962
    3000 0.427921855453 0.436226030664
    4000 0.42792106906 0.436209239552
    5000 0.427920980473 0.436203960286
    6000 0.427920970313 0.436202213743
    7000 0.427920969141 0.436201625309
    8000 0.427920969005 0.436201425816
    9000 0.42792096899 0.436201358041
    10000 0.427920968988 0.436201335
    11000 0.427920968988 0.436201327164
    12000 0.427920968988 0.436201324499
    13000 0.427920968988 0.436201323593
    14000 0.427920968988 0.436201323285
    15000 0.427920968988 0.43620132318
    16000 0.427920968988 0.436201323144
    17000 0.427920968988 0.436201323132
    18000 0.427920968988 0.436201323128
    19000 0.427920968988 0.436201323127
    20000 0.427920968988 0.436201323126
    21000 0.427920968988 0.436201323126
    22000 0.427920968988 0.436201323126
    23000 0.427920968988 0.436201323126
    24000 0.427920968988 0.436201323126
    25000 0.427920968988 0.436201323126
    26000 0.427920968988 0.436201323126
    27000 0.427920968988 0.436201323126
    28000 0.427920968988 0.436201323126
    29000 0.427920968988 0.436201323126
    30000 0.427920968988 0.436201323126
    31000 0.427920968988 0.436201323126
    32000 0.427920968988 0.436201323126
    33000 0.427920968988 0.436201323126
    34000 0.427920968988 0.436201323126
    35000 0.427920968988 0.436201323126
    36000 0.427920968988 0.436201323126
    37000 0.427920968988 0.436201323126
    38000 0.427920968988 0.436201323126
    39000 0.427920968988 0.436201323126
    40000 0.427920968988 0.436201323126
    41000 0.427920968988 0.436201323126
    42000 0.427920968988 0.436201323126
    43000 0.427920968988 0.436201323126
    44000 0.427920968988 0.436201323126
    45000 0.427920968988 0.436201323126
    46000 0.427920968988 0.436201323126
    47000 0.427920968988 0.436201323126
    48000 0.427920968988 0.436201323126
    49000 0.427920968988 0.436201323126
    Final Classification training rate: 0.789789503762
    Final Classification Testing rate: 0.789333333333
    


![png](output_110_1.png)



```python
#KNN implementation using sklearn 
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
      <th>sat_level_norm</th>
      <th>last_evaluation_norm</th>
      <th>average_montly_hours_norm</th>
      <th>time_spend_company_norm</th>
      <th>number_project_norm</th>
      <th>sales_by_number</th>
      <th>sales_by_number</th>
      <th>salary_encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.255861</td>
      <td>-0.290784</td>
      <td>-0.205843</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>0.205677</td>
      <td>0.224841</td>
      <td>0.284812</td>
      <td>0.312721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>-0.552564</td>
      <td>0.256091</td>
      <td>0.331540</td>
      <td>0.062721</td>
      <td>0.639389</td>
      <td>1.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>0.117765</td>
      <td>0.240466</td>
      <td>0.102569</td>
      <td>0.187721</td>
      <td>0.239389</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>-0.266850</td>
      <td>-0.306409</td>
      <td>-0.196497</td>
      <td>-0.062279</td>
      <td>-0.360611</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_knn = df[["satisfaction_level","last_evaluation","Work_accident","promotion_last_5years","number_project_norm","time_spend_company_norm","average_montly_hours_norm","salary_encoded"]]
Y_knn = df["left"]
```


```python
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_knn, Y_knn, test_size=0.5, random_state=42)
```


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn = KNeighborsClassifier(n_neighbors=1)
```


```python
knn.fit(X_train_k,y_train_k)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=1, p=2,
               weights='uniform')




```python
pred = knn.predict(X_test_k)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test_k,pred))
```

    [[5593  140]
     [ 100 1667]]
    


```python
print(classification_report(y_test_k,pred))
```

                 precision    recall  f1-score   support
    
              0       0.98      0.98      0.98      5733
              1       0.92      0.94      0.93      1767
    
    avg / total       0.97      0.97      0.97      7500
    
    


```python
#optimising for k 
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_k,y_train_k)
    pred_i = knn.predict(X_test_k)
    error_rate.append(np.mean(pred_i !=y_test_k))
    
```


```python
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,linestyle = 'dashed',marker = 'o',markersize = 10)
plt.xlabel('K')
plt.ylabel('error_size')
```




    <matplotlib.text.Text at 0xdd48f60>




![png](output_122_1.png)



```python
from sortedcontainers import SortedList
```

    
    


```python
class Knn(object):
    def __init__(self,k):
        self.k = k
    
    
    def fit(self,X,Y):
        self.X = X
        self.Y = Y
        
    def predict(self,X):
        
        y = np.zeros(len(X))
        
        for i,x in enumerate(X):#enumerate over the testing data
            sl = SortedList(load = self.k)
            for j,xt in enumerate(self.X):#enumerate over the training data
                d = x-xt
                dif = d.dot(d)
                if len(sl)<self.k:
                    sl.add((dif,self.Y[j] ))
                else:
                    
                    if dif<sl[-1][0]:
                        del sl[-1]
                        sl.add((dif,self.Y[j]))
                        
                        
                        
            votes={}
            for _,v in sl:
                votes[v] = votes.get(v,0)+1
            max_count = 0
            dom_class = -1
            
            for cl,val in votes.iteritems():
                if val>max_count:
                    max_count = val
                    dom_class=cl
            
            y[i] = dom_class
        
        return y 
                

    def score(self,X,Y):
        P = self.predict(X)
        return np.mean(P==Y)
```


```python
if __name__ == '__main__':
    for k in [1,2,3,4,5]:
        Ntrain = 7500
        X_train_knn = X_knn[:Ntrain]
        Y_train_knn = Y_knn[:Ntrain]
        X_test_knn = X_knn[Ntrain:]
        Y_test_knn = Y_knn[Ntrain:]
        
        
        X_train_knn = X_train_knn.as_matrix()
        Y_train_knn = Y_train_knn.as_matrix()
        X_test_knn = X_test_knn.as_matrix()
        Y_test_knn = Y_test_knn.as_matrix()
        
        knn_new = Knn(k)
        
        t0 = datetime.now()
        knn_new.fit(X_train_knn,Y_train_knn)
        print "Training Accuracy"
        knn_new.score(X_train_knn,Y_train_knn)
        print "Training time:",(datetime.now()-t0)
        
        
        t0 = datetime.now()
        print "Testing Accuracy:",knn_new.score(X_test_knn,Y_test_knn)
        print "Testing time:",(datetime.now()-t0)
        
```

    Training Accuracy
    Training time: 0:02:52.364000
    Testing Accuracy: 0.96399519936
    Testing time: 0:02:53.357000
    Training Accuracy
    Training time: 0:03:26.720000
    Testing Accuracy: 0.963861848246
    Testing time: 0:03:29.507000
    Training Accuracy
    Training time: 0:03:28.720000
    Testing Accuracy: 0.955594079211
    Testing time: 0:03:31.480000
    Training Accuracy
    Training time: 0:03:23.987000
    Testing Accuracy: 0.958394452594
    Testing time: 0:03:21.230000
    Training Accuracy
    Training time: 0:03:31.657000
    Testing Accuracy: 0.952660354714
    Testing time: 0:03:27.640000
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
