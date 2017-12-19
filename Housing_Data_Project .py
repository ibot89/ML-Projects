
# coding: utf-8

# In[1]:


#House Sales Project 
import numpy as np 
import pandas as pd 


# In[2]:


df = pd.read_csv('C:\Users\Botev\Desktop\kc_house_data.csv')


# In[8]:


df.head()


# In[7]:


df.info()#shows how many rows and columns 


# In[8]:


df['price'].mean()# average price of a house 


# In[11]:


df['price'].max()#most expensive house


# In[12]:


df['price'].min()#least expensive house 


# In[10]:


df['condition'].mean()#average condition 


# In[ ]:





# In[13]:


df['date']


# In[21]:


#how many houses were sold in each year 
sum(df['date'].apply(lambda x: x[:4])=='2014')#houses sold in 2014


# In[23]:


sum(df['date'].apply(lambda x: x[:4])=='2015')#sold in 2015


# In[29]:


df.loc[df['date'].apply(lambda x: x[:4])=='2015']['bedrooms']#number of bedrooms for the houses sold in 2015


# In[31]:


df.loc[df['date'].apply(lambda x: x[:4])=='2015']['bedrooms'].mean()


# In[33]:


df.loc[df['date'].apply(lambda x: x[:4])=='2014']['bedrooms'].mean()#number of bedrooms for the houses sold in 2015


# In[ ]:


#number of bedrooms is very similar between the two years. 


# In[52]:


#how many have been renovated
sum(df.loc[df['date'].apply(lambda x: x[:4])=='2014']['yr_renovated']!=0)# a lot have been renovated.


# In[57]:


sum(df.loc[df['date'].apply(lambda x: x[:4])=='2015']['yr_renovated']!=0)#renovated for 2015 


# In[3]:


#graphs 
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#number of rooms vs price 
sns.jointplot(x='bedrooms',y ='price',data = df,size = 10)


# In[9]:


sns.boxplot(x='bedrooms',y='price',data = df)#bedrooms vs price


# In[10]:


sns.lmplot(x='bedrooms',y='price',data = df)#bedrooms vs price - linear plot


# In[ ]:


#NO OBVIOUS RELATIONSHIP BETWEEN THE NUMBER OF BEDROOMS AND THE PRICE


# In[12]:


sns.lmplot(x='bedrooms',y='price',data = df,col='floors',row = 'bathrooms')# BAD GRaphs


# In[19]:


# sqft_living vs price
sns.lmplot(x='sqft_living',y='price',data = df,size=10)


# In[16]:


#Heatmap of correlations. Needs to be enlarged!!!!!!!!!!
sns.heatmap(df.corr(),annot=True)


# In[20]:


sns.jointplot(x='sqft_lot',y='price',data=df,size = 7,kind = 'reg')


# In[21]:


# Joint Plots
sns.jointplot(x='sqft_above',y='price',data=df,size = 5,kind = 'reg')
sns.jointplot(x='sqft_basement',y='price',data=df,size = 5,kind = 'reg')
sns.jointplot(x='yr_built',y='price',data=df,size = 5,kind = 'reg')
sns.jointplot(x='yr_renovated',y='price',data=df,size = 5,kind = 'reg')
sns.jointplot(x='sqft_living15',y='price',data=df,size = 5,kind = 'reg')
sns.jointplot(x='sqft_lot15',y='price',data=df,size = 5,kind = 'reg')


# In[ ]:


# Joint plots show clear relationship particularlly between sqft_above and sqft_living 


# In[20]:


df['yr_ren'] = df['yr_renovated'].apply(lambda x:'Renovated' if x >0 else 'Not Renovated')


# In[21]:


# CELL NOT USED
#droping zeroes on the year renovated and zeroes. Add 2 more columns. Tried it, doesnt work well.
df['sqft_basement_no_zeroes'] = df['sqft_basement'].apply(lambda x: x if x>0 else None)
df['yr_renovated_no_zeroes'] = df['yr_renovated'].apply(lambda x: x if x>0 else None)


# In[22]:


sns.jointplot(x='yr_renovated_no_zeroes',y='price',data=df,dropna=True,size=5,kind='reg')
sns.jointplot(x='sqft_basement_no_zeroes',y='price',data=df,dropna=True,size=5,kind='reg')


# In[19]:


#categorical variables 
sns.boxplot(y='waterfront',x='price',data = df,fliersize = 3,width = 0.8,orient='h',showmeans = True,linewidth=2.5)


# In[4]:


#setting the categorical variables 
df['yr_renovated_cat'] = df['yr_renovated'].apply(lambda x:1 if x>0 else 0)
df['sqft_basement_cat'] = df['sqft_basement'].apply(lambda x:1 if x>0 else 0)


# In[26]:


sns.boxplot(x='price',y='yr_renovated_cat',data = df,orient = 'h')


# In[27]:


sns.boxplot(x='price',y='sqft_basement_cat',data = df,orient = 'h')


# In[30]:


#Relationship between bedrooms and price 
sns.boxplot(y = 'bedrooms', x = 'price', data = df,width = 0.8,orient = 'h',showmeans = True)


# In[31]:


#Definate relationship between bathrooms and price
sns.boxplot(y = 'bathrooms', x = 'price', data = df,width = 0.8,orient = 'h',showmeans = True)


# In[32]:


#Floors
sns.boxplot(y = 'floors', x = 'price', data = df,width = 0.8,orient = 'h',showmeans = True)


# In[33]:


#Definate relationship 
sns.boxplot(y = 'view', x = 'price', data = df,width = 0.8,orient = 'h',showmeans = True)


# In[34]:


sns.boxplot(y = 'condition', x = 'price', data = df,width = 0.8,orient = 'h',showmeans = True)


# In[35]:


#Definate relationship
sns.boxplot(y = 'grade', x = 'price', data = df,width = 0.8,orient = 'h',showmeans = True)


# In[27]:


df.head()


# In[29]:


#Tried to see if there is a correlation b/n sqft_lot15  vs price and whether or not house is renovated
sns.lmplot(x='sqft_lot15',y='price',data = df,hue='yr_ren')


# In[30]:


sns.lmplot(x='sqft_living15',y='price',data = df,hue='yr_ren')#indicates stronger correlation between price,sqft_ling
# when Renovated


# In[32]:


g = sns.PairGrid(df,vars = ['sqft_living15','sqft_lot15','sqft_above'],hue = 'yr_ren')
g.map(plt.scatter)


# In[7]:


#linear regression based on 'sqft_living15','sqft_lot15','sqft_above'
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[6]:


X = df[['sqft_living15','sqft_lot15','sqft_above']]
y = df['price']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[8]:


lm = LinearRegression()


# In[15]:


fit = lm.fit(X_train,y_train)#fir the training dataset. Fit is sucessful 
fit


# In[10]:


print(lm.intercept_)# corresponds to theta0 from Andrew NG 


# In[11]:


#visualising coefficients 
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])


# In[12]:


coeff_df# the three thetas 


# In[13]:


#prediction
prediction = lm.predict(X_test)


# In[22]:


print "Accuracy:%d"%(int(fit.score(X_test,y_test)*100))+'%'#very low accuracy 


# In[21]:


plt.scatter(y_test,prediction)#not a great prediction model 


# In[25]:


sns.distplot((y_test - prediction),bins=50)


# In[ ]:


#Lin regresion using sklearn with two features 


# In[4]:


X1 = df[['sqft_living15','sqft_above']]
y1 = df['price']


# In[24]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.4, random_state=101)


# In[25]:


lm1 = LinearRegression()


# In[28]:


#fir the training dataset. Fit is sucessful 
fit1 = lm1.fit(X1_train,y1_train)
fit1


# In[33]:


print "Accuracy:%d"%(int(fit1.score(X1_test,y1_test)*100))+"%"


# In[19]:


print(lm1.intercept_)


# In[20]:


coeff_df1 = pd.DataFrame(lm1.coef_,X1.columns,columns=['Coefficients_new'])


# In[21]:


coeff_df1


# In[22]:


prediction1 = lm1.predict(X1_test)


# In[23]:


plt.scatter(y1_test,prediction1)


# In[24]:


sns.distplot((y1_test - prediction1),bins=50)


# In[30]:


#print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y1_test,prediction1)))


# In[5]:


df.head()


# In[35]:


df.head()


# In[40]:


#run the lin regression on most of the features 
#feature_cols = [ u'age_of_house',  u'bedrooms', u'bathrooms', u'sqft_living',
#      u'sqft_lot', u'floors', u'waterfront', u'view', u'condition', u'grade',
#     u'sqft_above', u'sqft_basement', u'yr_built_cat', u'yr_renovated_cat']
df.columns
#X2 = df[feature_cols]
#y2 = df['price']


# In[ ]:


#Linear Regression using 15 features. Best fit


# In[5]:


X2 = df[['bedrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement',
        'yr_built','sqft_basement_cat','yr_renovated_cat','sqft_living15','sqft_lot15']]
y2 = df['price']


# In[44]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.4, random_state=101)


# In[45]:


lm2 = LinearRegression()


# In[47]:


fit2 = lm2.fit(X2_train,y2_train)
fit2


# In[48]:


print "Accuracy:%d"%(int(fit2.score(X2_test,y2_test)*100))+"%"


# In[49]:


print(lm2.intercept_)


# In[52]:


coeff_df2 = pd.DataFrame(lm2.coef_,X2.columns,columns=['Coefficients_new'])
coeff_df2


# In[53]:


prediction2 = lm2.predict(X2_test)


# In[54]:


plt.scatter(y2_test,prediction2)


# In[55]:


sns.distplot((y2_test - prediction2),bins=50)


# In[5]:


X1


# In[ ]:


#--------------- Attempts to calc linear regression using sklearn - All Fails ------------------------


# In[6]:


cols = df.shape[1]  
X3 = X1.iloc[:,0:cols-1]  


# In[7]:


X3 = np.matrix(X3.values) 


# In[8]:


X3


# In[9]:


y3 = np.matrix(y1.values)  


# In[10]:


y3


# In[11]:


theta = np.matrix(np.array([0,0]))  


# In[12]:


def computeCost(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# In[13]:


computeCost(X3, y3, theta)# can compute the cost


# In[ ]:


#Cant compute the grad descent.
# Grad descent uses 2 for loops


# In[14]:


def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# In[ ]:


alpha = 0.01  
iters = 100

# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X3, y3, theta, alpha, iters)  
g  


# In[12]:


#matrixes
mat = np.matrix(np.array([[1,2],[4,5]]))


# In[13]:


mat


# In[15]:


mat.shape[0]


# In[20]:


mat[1:]


# In[ ]:


a = 


# In[13]:


#martix implementation 
X2


# In[ ]:


# Grad descent using matrixes - one for loop Fail again


# In[29]:


X_new = pd.DataFrame.as_matrix(X2)


# In[23]:


X_new


# In[27]:


X2.values[:,1]


# In[31]:


y_new = y2.values


# In[32]:


y_new


# In[14]:


theta = np.zeros(15)
theta


# In[38]:


def computeCost(X, y, theta):  
    #inner = np.power(((X * theta.T) - y), 2)
    inner = np.power(((X.dot(theta)) - y), 2)
    return np.sum(inner) / (2 * len(X))


# In[39]:


computeCost(X_new,y_new,theta)


# In[45]:


def gradientDescent1(X, y, theta =[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], alpha=0.01, num_iters=1500):
#def gradientDescent1(X, y, theta, alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    magic = alpha/m
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - (magic*(X.T.dot(h-y)))
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)


# In[ ]:


theta , Cost_J = gradientDescent1(X_new,y_new,theta=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],alpha=0.01,num_iters=1500)


# In[43]:





# In[ ]:


#----------------  NEw Lin regression try ----------------------------------------------------------------------------


# In[6]:


df['ones'] = 1


# In[7]:


X_newidea = df[['bedrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement',
        'yr_built','sqft_basement_cat','yr_renovated_cat','sqft_living15','sqft_lot15','ones']]
y_newidea = df['price']


# In[8]:


X_nimatrix = X_newidea.as_matrix()


# In[9]:


#Useless but wanted to call the variables differently from the start
X = X_nimatrix
Y = y_newidea


# In[29]:


l2 = 0.5
w_map = np.linalg.solve(l2*np.eye(16) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)


# In[30]:


d1_map = Y - Yhat_map
d2_map = Y - Y.mean()
r2_map = 1 - d1_map.dot(d1_map) / d2_map.dot(d2_map)


# In[31]:


r2_map


# In[23]:


w = np.linalg.solve( X.T.dot(X), X.T.dot(Y) )# w are all the parameters
Yhat = X.dot(w)


# In[26]:


w.size# 15 parameters in total plus one more because of the 'ones' column added 


# In[27]:


Yhat.size


# In[28]:


w# these are the parameters 


# In[30]:


d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)


# In[32]:


r2# 65% same as before while using sklearn 


# In[ ]:


#---------------------------Derivation-----------------------------------------------------
#i tells us which sample 
#j tells us which feature
#der - derivative
#W - parameters, 1 parameter per feature
#D - number of features
#N number of samples
#X input matrix N by D matrix
#Yhat prediction  - column vector with size D by 1
#y - output column vector size N by 1
# for multiclass lin regression we have: Yhat = b + W1*X1 + W2*X2 ....... WD*XD. WE can append b into W by adding a column of 1s
#into X, so we get: Yhat = W0*X0 + W1*X1.........WD*XD = W.T*X(in a matrix form)
# Error function = E = sum(y - Yhat)^2 (summation is always from i=1 to N). The goal is make the error function equal to 0.
#Thus we need to take the derivative of E with respect for Wj(derivative of the error function for each feature)
#and make it equal to 0.
#der(E with respect to Wj ) = sum(2(yi - W.TXi).())


# In[41]:


Yhat.shape


# In[46]:


Yhat1 = Yhat.T


# In[47]:


Yhat1.shape


# In[50]:


plt.scatter(Y,Yhat)


# In[51]:


sns.distplot((Y - Yhat),bins=50)


# In[ ]:


#One more try this time iliminating the categorical variables


# In[11]:


X_newidea2 = df[['bedrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement',
        'yr_built','sqft_living15','sqft_lot15','ones']]
y_newidea2 = df['price']


# In[12]:


X_nimatrix2 = X_newidea2.as_matrix()


# In[13]:


X2 = X_nimatrix2
Y2 = y_newidea2


# In[60]:


w2 = np.linalg.solve( X2.T.dot(X2), X2.T.dot(Y2) )
Yhat2 = X2.dot(w2)


# In[61]:


w2.size


# In[64]:


d1_n = Y - Yhat2
d2_n = Y2 - Y2.mean()
r2_n = 1 - d1_n.dot(d1_n) / d2_n.dot(d2_n)


# In[66]:


r2_n# the categorical variables the way they are right now have almost no effect on the r-squared 


# In[ ]:


#Grad descent again alpha param increased to 0.09 using just a for loop. Computer freezes again


# In[23]:


alpha = 0.09


# In[39]:


for t in range(0,100):
    Y2hat = X2.dot(theta)
    Y2hat = np.array(Y2hat)
    theta = theta - alpha*X2.T*((Y2hat - Y2))


# In[32]:


Y2hat


# In[33]:


theta = []


# In[38]:


np.array(X2.dot(theta))


# In[1]:


# From here down pointless DO NOT READ


# In[34]:


theta = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]


# In[16]:


X_newidea2 = df[['bedrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement',
        'yr_built','sqft_living15','sqft_lot15']]
y_newidea2 = df['price']


# In[17]:


X_nimatrix2 = X_newidea2.as_matrix()


# In[18]:


X2 = X_nimatrix2
Y2 = y_newidea2


# In[20]:


X2.shape

