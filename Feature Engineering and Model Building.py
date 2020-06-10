############# Customer Lifetime Value Prediction ################## 
            
        # FEATURE ENGINEERING AND MODEL BUILDING  # 
##################################################################
# Load Libraries #
##################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")
##################################################################
# Load Data #
##################################################################

df = pd.read_csv("Preprocessed.csv")
df
##################################################################
# Data Preprocessing #
##################################################################

df.isna().sum()

df['Income'] = df['Income'].fillna(df['Income'].mode()[0])

plt.rcParams["figure.figsize"] = 15,10
df.hist()

def outlier(x):
    high=0
    q1 = x.quantile(.25)
    q3 = x.quantile(.75)
    iqr = q3-q1
    low = q1-1.5*iqr
    high += q3+1.5*iqr
    outlier = (x.loc[(x < low) | (x > high)])
    return(outlier)

q1 =df['Customer_Lifetime_Value'].quantile(.25)
q3 = df['Customer_Lifetime_Value'].quantile(.75)
iqr = q3-q1

df_out = df[~((df['Customer_Lifetime_Value'] < (q1 - 1.5 *iqr))  |  (df['Customer_Lifetime_Value'] > (q3+ 1.5 * iqr)))]
print(df_out)
##################################################################
# FEATURE ENGINEERING #
##################################################################


cat_var = [var for var in df.columns if df[var].dtypes == "object"]
cat_var = df[cat_var]
print(cat_var)

for var in cat_var.columns:
    print("==========================")
    print(cat_var[var].value_counts())
    
def encode_Coverage(Coverage):
    if Coverage == "Basic":
        return 1
    elif Coverage == "Extended":
        return 2
    elif Coverage == "Premium":
        return 3
cat_var['Coverage'] = cat_var['Coverage'].apply(encode_Coverage)

def encode_Education(Education):
    if Education == "Bachelor":
        return 2
    elif Education == "College":
        return 2
    elif Education == "High School or Below":
        return 1
    elif Education == "Master":
        return 3
    elif Education == "Doctor":
        return 4
cat_var['Education'] = cat_var['Education'].apply(encode_Education)

def encode_EmploymentStatus(EmploymentStatus):
    if EmploymentStatus == "Employed":
        return 5
    elif EmploymentStatus == "Unemployed":
        return 4
    elif EmploymentStatus == "Medical Leave":
        return 3
    elif EmploymentStatus == "Disabled":
        return 2
    elif EmploymentStatus == "Retired":
        return 1
cat_var['EmploymentStatus'] = cat_var['EmploymentStatus'].apply(encode_EmploymentStatus)

def encode_Policy(Policy):
    if Policy == "Personal L3":
        return 2
    elif Policy == "Personal L2":
        return 2
    elif Policy == "Personal L1":
        return 2
    elif Policy == "Corporate L3":
        return 3
    elif Policy == "Corporate L2":
        return 3
    elif Policy == "Corporate L1":
        return 3
    elif Policy == "Special L2":
        return 1
    elif Policy == "Special L3":
        return 1
    elif Policy == "Special L1":
        return 1
cat_var['Policy'] = cat_var['Policy'].apply(encode_Policy)

def encode_Location_Code(Location_Code):
    if Location_Code == "Suburban":
        return 3
    elif Location_Code == "Rural":
        return 2
    elif Location_Code == "Urban":
        return 1
cat_var['Location_Code'] = cat_var['Location_Code'].apply(encode_Location_Code)


cat_var['Gender'] = pd.get_dummies(cat_var['Gender'])

cat_var['Marital_Status'] = pd.get_dummies(cat_var['Marital_Status'])

def encode_Renew_Offer_Type(Renew_Offer_Type):
    if Renew_Offer_Type == "Offer1":
        return 4
    elif Renew_Offer_Type == "Offer2":
        return 3
    elif Renew_Offer_Type == "Offer3":
        return 2
    elif Renew_Offer_Type == "Offer4":
        return 1
cat_var['Renew_Offer_Type'] = cat_var['Renew_Offer_Type'].apply(encode_Renew_Offer_Type)


def encode_Sales_Channel(Sales_Channel):
    if Sales_Channel == "Agent":
        return 4
    elif Sales_Channel == "Branch":
        return 3
    elif Sales_Channel == "Call Center":
        return 2
    elif Sales_Channel == "Web":
        return 1
cat_var['Sales_Channel'] = cat_var['Sales_Channel'].apply(encode_Sales_Channel)

def encode_Vehicle_Class(Vehicle_Class):
    if Vehicle_Class == "Four-Door Car":
        return 6
    elif Vehicle_Class == "Two-Door Car":
        return 5
    elif Vehicle_Class == "SUV":
        return 4
    elif Vehicle_Class == "Sports Car":
        return 3
    elif Vehicle_Class == "Luxury SUV":
        return 2
    elif Vehicle_Class == "Luxury Car":
        return 1
cat_var['Vehicle_Class'] = cat_var['Vehicle_Class'].apply(encode_Vehicle_Class)


def encode_Policy_Type(Policy_Type):
    if Policy_Type == "Personal Auto":
        return 2
    elif Policy_Type == "Corporate Auto":
        return 3
    elif Policy_Type == "Special Auto":
        return 1
cat_var['Policy_Type'] = cat_var['Policy_Type'].apply(encode_Policy_Type)

num_var = [var for var in df.columns if df[var].dtypes != "object"]
num_var = df[num_var]
num_var


df_new = pd.concat([num_var,cat_var],axis=1)
df_new

mean_encode = df_new.groupby('Income')['Customer_Lifetime_Value'].mean()

df_new.loc[:,'Income'] = df_new.Income.map(mean_encode)

mean = df_new['Customer_Lifetime_Value'].mean()

agg =  df_new.groupby('Income')['Customer_Lifetime_Value'].agg(['count','mean'])

counts = agg['count']
means =agg['mean']
weight =100

smooth = (counts * means + weight * means) / (counts + weight)

df_new.loc[:,'Income'] = df_new.Income.map(smooth)

#--------------------------------------------------------------#
mean_encode = df_new.groupby('Location_Geo')['Customer_Lifetime_Value'].mean()

df_new.loc[:,'Location_Geo'] = df_new.Location_Geo.map(mean_encode)

mean = df_new['Customer_Lifetime_Value'].mean()

agg =  df_new.groupby('Location_Geo')['Customer_Lifetime_Value'].agg(['count','mean'])

counts = agg['count']
means =agg['mean']
weight =100

smooth = (counts * means + weight * means) / (counts + weight)

df_new.loc[:,'Location_Geo'] = df_new.Location_Geo.map(smooth)
#------------------------------------------------------------#

df_version1 = df_new.drop(['CustomerID'],axis=1)
df_version1

##################################################################
# TRAIN TEST SPLIT #
##################################################################

x = df_version1.drop(['Customer_Lifetime_Value'],axis=1)
y = df_version1['Customer_Lifetime_Value']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=40)
##################################################################
# FEATURE SELECTION #
##################################################################

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
estimator = RandomForestRegressor()
selector = RFE(estimator,6,step=1)
selector = selector.fit(x_train,y_train)
selector.ranking_

x = x.drop(['Marital_Status','Policy','Location_Code','Gender','Months_Since_Last_Claim'],axis=1)
##################################################################
# FEATURE SCALING #
##################################################################

x = x.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
##################################################################
# MODEL BUILDING #
##################################################################

import statsmodels.api as sm
model2 =sm.OLS(y_train,x_train).fit()

model2.summary()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr_y_pred=lr.predict(x_test)

from sklearn import metrics
lr_RMSE = np.sqrt(metrics.mean_squared_error(y_test,lr_y_pred))
lr_RMSE

from sklearn.metrics import r2_score
lr_r2_score = r2_score(y_test,lr_y_pred)
lr_r2_score

import pickle
pickle.dump(lr,open('linear_model.pkl','wb'))

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1000,random_state=3)
reg=regressor.fit(x_train,y_train)

rf_y_pred=regressor.predict(x_test)


from sklearn import metrics
rf_RMSE = np.sqrt(metrics.mean_squared_error(y_test,rf_y_pred))
rf_RMSE

from sklearn.metrics import r2_score
rf_r2_score = r2_score(y_test,rf_y_pred)
rf_r2_score


import pickle
pickle.dump(regressor,open('rf_model.pkl','wb'))


