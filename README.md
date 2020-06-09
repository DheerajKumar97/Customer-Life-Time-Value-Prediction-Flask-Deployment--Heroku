# CUSTOMER LIFE TIME VALUE PREDICTION

Heroku App Link 1 : https://customerlifetimevaluepred.herokuapp.com/

### This project is designed to predict the CUSTOMER LIFE TIME VALUE  using Regression Analysis with Python, FLASK, HTML, SQL 
### A highly comprehensive analysis with all data cleaning, exploration, visualization, feature selection, model building, evaluation and MLR assumptions validity steps explained in detail.


# The Python file has following sections:


### In Customer Lifetime Value (Exploratory Data Analysis).py

1- **Data Preprocessing** and some **Exploratory Data Analysis** to understand the data

2- **Data cleaning**



### In Customer Lifetime Value (Feature Engineering).py

1- Data preparation: **Feature Engineering and Scaling**

2- Feature Selection using **RFE and Model Building**

3- **Regression Assumptions** Validation and **Outlier Removal**

4- Rebuilding the Model Post Outlier Removal: Feature Selection  & RFE

5- **Removing Multicollinearity**, Model Re-evaluation and Assumptions Validation

## Details of Variables  [Response Variable ==> Customer Life Time Value]

![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Details%20of%20Variables.png)

## OLS REGRESSION MODEL OUTPUT 

![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/OLS%20Rgression%20Results.png)

# Sample EDA VISUALIZATIONS
###############################################################################################
## Gender__vs__Policy Type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Gender%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) When compared to Male, Female gender has taken more Policies
             (2) In all three Policies, the most prefered or Taken Policy Type is Personal Auto Policy for both Male and Female
###############################################################################################
## Marital Status__vs__Policy Type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Marital_status%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all three categories is Personal Auto Policy
             (2) In all three Categories more Policy takers are Married People
###############################################################################################
## Employment Status__vs__Policy Type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/EmploymentStatus%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all five categories is Personal Auto Policy
             (2) In all three Categories more Policy takers are Employed People
###############################################################################################
## Employed_People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Employed_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all three categories is Personal Auto Policy
             (2) In all three Categories more Policy takers are Employed and  Married People
###############################################################################################
## UnEmployed_People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/UnEmployed_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all three categories is Personal Auto Policy
             (2) In all three Categories more Policy takers are UnEmployed and  Single People
###############################################################################################
## Employed Male People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Employed_Male_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all three categories is Personal Auto Policy
             (2) In all three Categories more Policy takers are Employed and Married Male People
###############################################################################################
## Employed FeMale People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Employed_FeMale_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all three categories is Personal Auto Policy
             (2) In all three Categories more Policy takers are Employed and Married FeMale People
###############################################################################################
## UnEmployed Male People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/UnEmployed_Male_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all three categories is Personal Auto Policy
             (2) In all three Categories more Policy takers are UnEmployed and Single Male People
###############################################################################################
## UnEmployed FeMale People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/UnEmployed_FeMale_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all three categories is Personal Auto Policy
             (2) In all three Categories more Policy takers are UnEmployed and Single FeMale People
###############################################################################################
## Married People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Married_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all five categories is Personal Auto Policy
             (2) In all five Categories more Policy takers are Married and Employed People
###############################################################################################
## Married Male People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Married_Male_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all five categories is Personal Auto Policy
             (2) In all five Categories more Policy takers are Married and Employed Male People
###############################################################################################
## Married FeMale People vs Policy_type
![Alt Text](https://github.com/DheerajKumar97/Customer-Life-Time-Value-Prediction-Flask-Deployment--Heroku/blob/master/CLTP%20Analysis%20Output/Married_FeMale_People%20vs%20Policy_type.png)

### From this tabulation our insight will be 
             (1) Most prefered or Taken Policy Type for all five categories is Personal Auto Policy
             (2) In all five Categories more Policy takers are Married and Employed FeMale People
###############################################################################################
