# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:37:39 2025

@author: SUHEL
"""

##IMporting the required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso
warnings.filterwarnings('ignore')


###Reading the Train and Test Datasets
df_train=pd.read_csv('C:\\Users\\SUHEL\\Downloads\\train_bigmart.csv')
df_test=pd.read_csv('C:\\Users\\SUHEL\\Downloads\\test_bigmart.csv')

###Checking for the duplicates
df_train.duplicated().any()
df_test.duplicated().any()

####EDA Techniques

###Checking the datsets size information
df_train.shape
df_test.shape

###This helps in getting the know the numerical columns statistics such as min,max,average etc
df_train.describe()
df_test.describe()

##Checking for the null values in each column
df_train.isnull().sum()
df_test.isnull().sum()

###Univariate Imputation
mean_weight = df_train['Item_Weight'].mean()
median_weight = df_train['Item_Weight'].median()
df_train['Item_Weight_mean'] = df_train['Item_Weight'].fillna(mean_weight)
df_train['Item_Weight_median'] = df_train['Item_Weight'].fillna(median_weight)
# The interpolate() function fills in the missing (NaN) values in a column by estimating values based on the neighboring values.

df_train['Item_Weight_interploate'] = df_train['Item_Weight'].interpolate(method="linear")


print("Original Weight variable variance", df_train['Item_Weight'].var())
print("Product Weight variance after mean imputation", df_train['Item_Weight_mean'].var())
print("Product Weight variance after median imputation", df_train['Item_Weight_median'].var())
print("Product Weight variance after median imputation", df_train['Item_Weight_interploate'].var())



df_train['Item_Weight'].plot(kind= "kde", label="Original")
df_train['Item_Weight_mean'].plot(kind= "kde", label= "Mean")
df_train['Item_Weight_median'].plot(kind= "kde", label= "Median")
df_train['Item_Weight_interploate'].plot(kind = "kde", label = "interpolate")


plt.legend()
plt.show()

##From the Plot, its clearlyy visible that we should the fill nan values using Interpolation methood
##Hence dropping the remaining columns
df_train.drop(columns=['Item_Weight','Item_Weight_mean','Item_Weight_median'],inplace=True)
##Remaing the columns
df_train.rename(columns={'Item_Weight_interploate':'Item_Weight'},inplace=True)

###Same analysis needs to be done of test dataset and validation needs to be done
###Univariate Imputation
mean_weight = df_test['Item_Weight'].mean()
median_weight = df_test['Item_Weight'].median()
df_test['Item_Weight_mean'] = df_test['Item_Weight'].fillna(mean_weight)
df_test['Item_Weight_median'] = df_test['Item_Weight'].fillna(median_weight)
# The interpolate() function fills in the missing (NaN) values in a column by estimating values based on the neighboring values.

df_test['Item_Weight_interploate'] = df_test['Item_Weight'].interpolate(method="linear")


print("Original Weight variable variance", df_test['Item_Weight'].var())
print("Product Weight variance after mean imputation", df_test['Item_Weight_mean'].var())
print("Product Weight variance after median imputation", df_test['Item_Weight_median'].var())
print("Product Weight variance after median imputation", df_test['Item_Weight_interploate'].var())



df_test['Item_Weight'].plot(kind= "kde", label="Original")
df_test['Item_Weight_mean'].plot(kind= "kde", label= "Mean")
df_test['Item_Weight_median'].plot(kind= "kde", label= "Median")
df_test['Item_Weight_interploate'].plot(kind = "kde", label = "interpolate")

plt.legend()
plt.show()

##From the Plot, its clearlyy visible that we should the fill nan values using Interpolation methood
##Hence dropping the remaining columns
df_test.drop(columns=['Item_Weight','Item_Weight_mean','Item_Weight_median'],inplace=True)
#Renaming the columns
df_test.rename(columns={'Item_Weight_interploate':'Item_Weight'},inplace=True)


# ------------------------------------------------

# Taking the Groupby Mean of 'Outlet_Size' for each Outlet_Type
outlet_size_mode_train = df_train.groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode()[0])

# Impute missing 'Outlet_Size' in train
df_train['Outlet_Size'] = df_train.apply(
    lambda row: outlet_size_mode_train[row['Outlet_Type']] if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'],
    axis=1
)

# Taking the Groupby Mean of 'Outlet_Size' for each Outlet_Type in Test Data
outlet_size_mode_test = df_train.groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode()[0])


# Impute missing 'Outlet_Size' in test
df_test['Outlet_Size'] = df_test.apply(
    lambda row: outlet_size_mode_test[row['Outlet_Type']] if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'],
    axis=1
)

###Item Visibility means the visibility of each Item, hence it can be 0
# Replace 0 values with NaN first

df_train['Item_Visibility_interpolate'] = df_train['Item_Visibility'].replace(0,np.nan).interpolate(method='linear')

df_train['Item_Visibility'].plot(kind="kde", label="Original")
df_train['Item_Visibility_interpolate'].plot(kind="kde", color='red', label="Interpolate")
df_train.drop(columns=['Item_Visibility'],inplace=True)
df_train.rename(columns={'Item_Visibility_interpolate':'Item_Visibility'},inplace=True)


plt.legend()
plt.show()

##For Test data
df_test['Item_Visibility_interpolate'] = df_test['Item_Visibility'].replace(0,np.nan).interpolate(method='linear')

df_test['Item_Visibility'].plot(kind="kde", label="Original")
df_test['Item_Visibility_interpolate'].plot(kind="kde", color='red', label="Interpolate")
df_test.drop(columns=['Item_Visibility'],inplace=True)


plt.legend()
plt.show()
df_test.rename(columns={'Item_Visibility_interpolate':'Item_Visibility'},inplace=True)


###Creating the copy of cleaned data
##For Train Dataset
df_train_clean=df_train.copy()

##For Test DataSet
df_test_clean=df_test.copy()

###Feature Engineering 
###Below Function helps in getting to know about the count of each categorical column
##Now checking the Categorical variable count distribution

def plot_categorical_distribution(df):
    """
    Plots the distribution of categorical variables using Plotly.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Iterate through each categorical column and plot the distribution
    for col in categorical_cols:
        # Get the count of each category
        count_data = df[col].value_counts().reset_index()
        count_data.columns = [col, 'Count']
        
        # Create a Plotly bar chart
        fig = px.bar(count_data, x=col, y='Count', 
                     title=f'Distribution of {col}',
                     labels={col: col, 'Count': 'Count'},
                     color='Count', color_continuous_scale='Viridis')

        # Show the plot
        fig.show()

    print("\nCategorical Distribution Plots Completed!")

###Creating the plots for each dataset
    
plot_categorical_distribution(df_train_clean)

plot_categorical_distribution(df_test_clean)


##From the Bar Chart, we got to know that Item content column needs to be cleaned
##For the Item Fat Content, we need to align all the similar type into one
##Like Reg-->Regular,Low fat-->Low Fat etc

##Train Dataset
df_train_clean['Item_Fat_Content'] = df_train_clean['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

##Test Dataset
df_test_clean['Item_Fat_Content'] = df_test_clean['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

### Combining them captures the relationships between the outlet's type, size, and location more meaningfully

##For Train Data
df_train_clean['Outlet_Type_Size_Location'] = df_train_clean['Outlet_Type'] + "_" + df_train_clean['Outlet_Size'] + "_" + df_train_clean['Outlet_Location_Type']
df_test_clean['Outlet_Type_Size_Location'] = df_test_clean['Outlet_Type'] + "_" + df_test_clean['Outlet_Size'] + "_" + df_test_clean['Outlet_Location_Type']

##For Test Data
df_train_clean.drop(columns=['Outlet_Type','Outlet_Size','Outlet_Location_Type'], inplace=True)
df_test_clean.drop(columns=['Outlet_Type','Outlet_Size','Outlet_Location_Type'], inplace=True)


###Adding the new Feature to capture Item Visibility per Item Weight which helps in finding out the insights related to Item Weight/Item Visibility
##For Train Dataset
df_train_clean['Weight/Visibility']=df_train_clean['Item_Weight']/df_train_clean['Item_Visibility']
##For Test Dataset
df_test_clean['Weight/Visibility']=df_test_clean['Item_Weight']/df_train_clean['Item_Visibility']

##For any model to predict correctly, its will be useful to convert the year into the age
##As the dataset is till 2013, consider the same to calculate the Outlet Age

df_train_clean['Outlet_Age']=2013-df_train_clean['Outlet_Establishment_Year']
df_test_clean['Outlet_Age']=2013-df_test_clean['Outlet_Establishment_Year']


##Dropping Establisment Year as it may increase the multicolinearity
df_train_clean.drop(columns=['Outlet_Establishment_Year'],inplace=True)
df_test_clean.drop(columns=['Outlet_Establishment_Year'],inplace=True)

#The original Item_Identifier looked like random codes: 'FDX07', 'NCD19', etc.

#But actually, those first two letters encode the product category (Food, Non-Consumable, Drinks)

#Get the first two characters of ID as they depict the Type of Product 

##Train Data
df_train_clean['Item_Type_Category'] = df_train_clean['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
df_train_clean['Item_Type_Category'] = df_train_clean['Item_Type_Category'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
df_train_clean['Item_Type_Category'].value_counts()

##Test Data
df_test_clean['Item_Type_Category'] = df_test_clean['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
df_test_clean['Item_Type_Category'] = df_test_clean['Item_Type_Category'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})

# 2. Calculate and Map Category-wise MRP Mean for Train data

category_mrp_mean_train = df_train_clean.groupby('Item_Type_Category')['Item_MRP'].mean()
df_train_clean['Category_MRP_Mean'] = df_train_clean['Item_Type_Category'].map(category_mrp_mean_train)

## Now for Test data
category_mrp_mean_test = df_test_clean.groupby('Item_Type_Category')['Item_MRP'].mean()
df_test_clean['Category_MRP_Mean'] = df_test_clean['Item_Type_Category'].map(category_mrp_mean_test)

##Calculating the Outlet Group Mean 

# 2. Calculate mean MRP for each Outlet Group in Train Data
outlet_group_mrp_mean_train = df_train_clean.groupby('Outlet_Type_Size_Location')['Item_MRP'].mean()

# 3. Map it back to the dataset
df_train_clean['Outlet_Group_MRP_Mean'] = df_train_clean['Outlet_Type_Size_Location'].map(outlet_group_mrp_mean_train)


# 2. Calculate mean MRP for each Outlet Group in Test Data
outlet_group_mrp_mean_test = df_test_clean.groupby('Outlet_Type_Size_Location')['Item_MRP'].mean()

# 3. Map it back to the dataset
df_test_clean['Outlet_Group_MRP_Mean'] = df_test_clean['Outlet_Type_Size_Location'].map(outlet_group_mrp_mean_test)


# 1. Calculate mean Item Visibility for each Outlet for Train Data
outlet_visibility_mean_train = df_train_clean.groupby('Outlet_Identifier')['Item_Visibility'].mean()

# 2. Map the mean visibility back to the main data
df_train_clean['Outlet_Visibility_Mean'] = df_train_clean['Outlet_Identifier'].map(outlet_visibility_mean_train)

# 3. (Optional) Create a deviation feature
df_train_clean['Item_Outlet_Visibility_Deviation'] = df_train_clean['Item_Visibility'] - df_train_clean['Outlet_Visibility_Mean']


# 1. Calculate mean Item Visibility for each Outlet for Test Data
outlet_visibility_mean_test = df_train_clean.groupby('Outlet_Identifier')['Item_Visibility'].mean()

# 2. Map the mean visibility back to the main data
df_test_clean['Outlet_Visibility_Mean'] = df_test_clean['Outlet_Identifier'].map(outlet_visibility_mean_test)

# 3. (Optional) Create a deviation feature
df_test_clean['Item_Outlet_Visibility_Deviation'] = df_test_clean['Item_Visibility'] - df_test_clean['Outlet_Visibility_Mean']

##Checking the co relation of each column bby plotingg heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_train_clean.corr(), annot=True, cmap='coolwarm', linewidths=1)
plt.title("Correlation Heatmap")
plt.show()

##During EDA, its was found that Unique Identifier is nothing but the Unique Id,as it wont be adding much value while training, we need to drop in both Train and Test

##As we will be using it in the submission file, lets save only Test Identifier and later we can combine with Predcited Values,as order wont be changing

test_item_identifier = df_test_clean['Item_Identifier'].copy()

##Droping it both in Train and Test Datasets

df_train_clean.drop('Item_Identifier', axis=1, inplace=True)
df_test_clean.drop('Item_Identifier', axis=1, inplace=True)

##For ML/Dl Models, we need to convert the Categorical columns into NUmerical columns, as Model wont be able to understand text



# Define categorical columns
categorical_cols = df_train_clean.select_dtypes(include=['object']).columns.tolist()

# Combine train and test for consistent encoding
combined = pd.concat([df_train_clean[categorical_cols], df_test_clean[categorical_cols]], axis=0)

# Apply Label Encoding
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    encoders[col] = le

# Now split back
df_train_clean[categorical_cols] = combined.iloc[:len(df_train_clean)][categorical_cols]
df_test_clean[categorical_cols] = combined.iloc[len(df_train_clean):][categorical_cols]

###Now Feature Scaling is to be done, to bring all the features into similar range of values

# Initialize the scaler
scaler = StandardScaler()

# Define features and target
X_train = df_train_clean.drop(['Item_Outlet_Sales'], axis=1)
y_train = df_train_clean['Item_Outlet_Sales']

X_test = df_test_clean.copy() 

# Fit the scaler only on training data
scaler.fit(X_train)

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

##Converting the array into dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


###Now Model Builiding
##Linear Regression
# Train model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predict
y_pred_lr = lr.predict(X_train_scaled)

# Evaluate
rmse_lr = np.sqrt(mean_squared_error(y_train, y_pred_lr))
r2_lr=r2_score(y_train, y_pred_lr)
print(f"Linear Regression RMSE on Train: {rmse_lr:.4f}")

##Ridge and Lasso Model



# Assuming you already have:
# X_train_scaled, X_test_scaled, y_train, y_test

# 1. Train Ridge Regression
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)

# Predict
ridge_preds_train = ridge.predict(X_train_scaled)
ridge_preds_test = ridge.predict(X_test_scaled)

# 2. Train Lasso Regression
lasso = Lasso(alpha=0.001, random_state=42)  # small alpha because Lasso can shrink too much
lasso.fit(X_train_scaled, y_train)

# Predict
lasso_preds_train = lasso.predict(X_train_scaled)
lasso_preds_test = lasso.predict(X_test_scaled)

# 3. Evaluate models
before_tuning_ridge_rmse=np.sqrt(mean_squared_error(y_train, ridge_preds_train))
before_tuning_ridge_r2=r2_score(y_train, ridge_preds_train)

print("Ridge RMSE on Train:",before_tuning_ridge_rmse)

before_tuning_lasso_rmse=np.sqrt(mean_squared_error(y_train, lasso_preds_train))
before_tuning_lasso_r2=r2_score(y_train, ridge_preds_train)
print("Lasso RMSE on Train:", before_tuning_lasso_rmse)

##Checking andd Ploting the Feature Importance


# Get feature names
feature_names = X_train.columns  # X_train_scaled must be a DataFrame for this

# Ridge Coefficients
ridge_coeff = pd.Series(np.abs(ridge.coef_), index=feature_names).sort_values(ascending=False)

# Lasso Coefficients
lasso_coeff = pd.Series(np.abs(lasso.coef_), index=feature_names).sort_values(ascending=False)

# Plot
plt.figure(figsize=(14,6))

# Ridge
plt.subplot(1, 2, 1)
ridge_coeff.plot(kind='bar')
plt.title('Feature Importance - Ridge Regression')
plt.tight_layout()

# Lasso
plt.subplot(1, 2, 2)
lasso_coeff.plot(kind='bar', color='orange')
plt.title('Feature Importance - Lasso Regression')
plt.tight_layout()

plt.show()

##Hperparamter tuning to findout the right value of alpha

# Ridge tuning
ridge_params = {'alpha': np.logspace(-3, 3, 20)}  # from 0.001 to 1000
ridge_grid = RandomizedSearchCV(Ridge(random_state=42), ridge_params, cv=5, scoring='neg_root_mean_squared_error')
ridge_grid.fit(X_train_scaled, y_train)

print("Best alpha for Ridge:", ridge_grid.best_params_['alpha'])

# Lasso tuning
lasso_params = {'alpha': np.logspace(-4, 2, 20)}  # Lasso usually needs smaller alpha
lasso_grid = RandomizedSearchCV(Lasso(random_state=42, max_iter=1000), lasso_params, cv=5, scoring='neg_root_mean_squared_error')
lasso_grid.fit(X_train_scaled, y_train)

print("Best alpha for Lasso:", lasso_grid.best_params_['alpha'])

##From the Plots we got to know the features which arent adding any value
features_drop=['Item_Type','Item_Weight','Outlet_Identifier','Weight/Visibility']

X_train_scaled2=X_train_scaled.copy()

X_train_scaled2.drop(columns=['Item_Weight','Item_Type'],inplace=True)

X_test_scaled2=X_test_scaled.copy()
X_test_scaled2.drop(columns=['Item_Weight','Item_Type'],inplace=True)

###Now retrainig the Lasso and Ridge model for the new alpha value and reduced features
# 1. Train Ridge Regression
ridge = Ridge(alpha=ridge_grid.best_params_['alpha'], random_state=42)
ridge.fit(X_train_scaled2, y_train)

# Predict
ridge_preds_train = ridge.predict(X_train_scaled2)
ridge_preds_test = ridge.predict(X_test_scaled2)

# 2. Train Lasso Regression
lasso = Lasso(alpha=lasso_grid.best_params_['alpha'], random_state=42)  # small alpha because Lasso can shrink too much
lasso.fit(X_train_scaled2, y_train)

# Predict
lasso_preds_train = lasso.predict(X_train_scaled2)
lasso_preds_test = lasso.predict(X_test_scaled2)

# 3. Evaluate models
after_tuning_ridge_rmse=np.sqrt(mean_squared_error(y_train, ridge_preds_train))
after_tuning_ridge_r2=r2_score(y_train, ridge_preds_train)
print("Ridge RMSE on Train:",after_tuning_ridge_rmse)

after_tuning_lasso_rmse=np.sqrt(mean_squared_error(y_train, lasso_preds_train))
after_tuning_lasso_r2=r2_score(y_train, lasso_preds_train)
print("Lasso RMSE on Train:", after_tuning_lasso_rmse)

##After reducing the parameter and changing the alpha, there isnt much change,hence Trying Decision Tree 
# 1. Initialize Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)

# 2. Train the model
dt_model.fit(X_train, y_train)  ###We will be using unscaled versions of Train and Test Data for Tree based model

# 3. Predict on Train Set
y_train_pred_dt = dt_model.predict(X_train)

# 4. Evaluate
train_rmse_dt = np.sqrt(mean_squared_error(y_train, y_train_pred_dt))
train_r2_dt = r2_score(y_train, y_train_pred_dt)

print("Decision Tree Performance on Train Set:")
print(f"Train RMSE: {train_rmse_dt:.4f}")
print(f"Train R2 Score: {train_r2_dt:.4f}")

##This is performing very well when compared to LR,Lassso and Ridge models
#Now Tuning the Decision Tree model, to check if its performing better or not
# 1. Initialize base model
dt2 = DecisionTreeRegressor(random_state=42)

# 2. Define a reasonable Grid (to not make it too heavy)
param_grid_dt = {
    'max_depth': [3, 5, 7, 9, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 3. Set up GridSearchCV
rand_search_dt = RandomizedSearchCV(estimator=dt2,
                               param_distributions=param_grid_dt,
                               cv=5,
                               scoring='neg_root_mean_squared_error',
                               verbose=1,
                               n_jobs=-1)

# 4. Fit GridSearch
rand_search_dt.fit(X_train, y_train)

# 5. Best hyperparameters
print("Best Hyperparameters Found:")
print(rand_search_dt.best_params_)

# 6. Train final model with best parameters
best_dt_model = rand_search_dt.best_estimator_

# 7. Predict on train set
y_train_pred_best_dt = best_dt_model.predict(X_train)

# 8. Evaluate
train_rmse_best_dt = np.sqrt(mean_squared_error(y_train, y_train_pred_best_dt))
train_r2_best_dt = r2_score(y_train, y_train_pred_best_dt)

print("Final Decision Tree Performance on Train Set:")
print(f"Train RMSE: {train_rmse_best_dt:.4f}")
print(f"Train R2 Score: {train_r2_best_dt:.4f}")


## When compared the DT model before and after Tuning, it was found that RMSE is increased after tuning, which suggestes that before tunnig its working

##Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

#Get Feature Importances
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

#Model Evaluation

y_train_pred_rf = rf_model.predict(X_train)
rf_bef_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
rf_bef_r2 = r2_score(y_train, y_train_pred_rf)

print(f"Validation RMSE before tuning Selection: {rf_bef_rmse:.4f}")
print(f"Validation R2 before tuning Selection: {rf_bef_r2:.4f}")



##Hyperparameter tuning for Random Forest
# 1. Initialize Random Forest
rf2 = RandomForestRegressor(random_state=42)

# 2. Define parameter grid (small and smart)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# 3. GridSearchCV setup
grid_search_rf = RandomizedSearchCV(estimator=rf2,
                              param_distributions=param_grid_rf,
                              n_iter=10,
                              cv=5,
                              n_jobs=-1,
                              verbose=2,
                              scoring='neg_root_mean_squared_error')

# 4. Fit Random Forest
grid_search_rf.fit(X_train, y_train)

# 5. Best parameters
print(" Best Hyperparameters for Random Forest:")
print(grid_search_rf.best_params_)

# 6. Predict on Train Set
best_rf_model = grid_search_rf.best_estimator_
best_y_train_pred_rf = best_rf_model.predict(X_train)

# 7. Evaluate on Train
after_train_rmse_rf = np.sqrt(mean_squared_error(y_train, best_y_train_pred_rf))
after_train_r2_rf = r2_score(y_train, best_y_train_pred_rf)

print("\nRandom Forest Performance on Train Set:")
print(f"Train RMSE: {after_train_rmse_rf:.4f}")
print(f"Train R2 Score: {after_train_r2_rf:.4f}")


###XG Boost Modeling
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)


#Model Evaluation

y_train_pred_xbg = xgb_model.predict(X_train)
xgb_bef_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_xbg))
xgb_bef_r2 = r2_score(y_train, y_train_pred_xbg)

print(f"Validation RMSE before tuning Selection: {xgb_bef_rmse:.4f}")
print(f"Validation R2 before tuning Selection: {xgb_bef_r2:.4f}")

##R2 score and RMSE changes drastically, might be the case of Overfitting
##XGBoost Hyperparameter
# Define XGBoost Regressor
xgb2 = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define Hyperparameter Grid
param_grid_xgb = {
    'n_estimators': [100, 300, 500, 800],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}

# Split X_train_scaled into internal train/val
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)


#  Setup Grid Search
ran_search_xgb = RandomizedSearchCV(estimator=xgb2,
                               param_distributions=param_grid_xgb,
                               cv=5,
                               n_jobs=-1,
                               verbose=2,
                               n_iter=10,
                               scoring='neg_root_mean_squared_error')

#  Fit the model
ran_search_xgb.fit(X_train_split, y_train_split,
                  early_stopping_rounds=10,
                  eval_set=[(X_val_split, y_val_split)],
                  verbose=False)

# Best model
best_xgb_model = ran_search_xgb.best_estimator_

print("Best Hyperparameters:")
print(ran_search_xgb.best_params_)

y_train_bestpred_xbg = ran_search_xgb.predict(X_val_split)
xgb_after_rmse = np.sqrt(mean_squared_error(y_val_split, y_train_bestpred_xbg))
xgb_after_r2 = r2_score(y_val_split, y_train_bestpred_xbg)

print(f"Validation RMSE before tuning Selection: {xgb_after_rmse:.4f}")
print(f"Validation R2 before tuning Selection: {xgb_after_r2:.4f}")

print("Best Parameters (XGBoost):", ran_search_xgb.best_params_)
print("Best CV RMSE (XGBoost):", -ran_search_xgb.best_score_)



##Creating the all model perforamances dataframe
# Create a list of dictionaries
model_performance = [
    {'Model': 'Ridge Regression', 'RMSE': before_tuning_ridge_rmse, 'R2 Score': before_tuning_ridge_r2},
    {'Model': 'Ridge Regression Hyperparameter','RMSE': after_tuning_ridge_rmse, 'R2 Score': after_tuning_ridge_r2},
    {'Model': 'Lasso Regression', 'RMSE': before_tuning_lasso_rmse, 'R2 Score': before_tuning_lasso_r2},
    {'Model': 'Lasso Regression Hyperparameter', 'RMSE': after_tuning_lasso_rmse, 'R2 Score': after_tuning_lasso_r2},
    {'Model': 'Decision Tree', 'RMSE': train_rmse_dt, 'R2 Score': train_r2_dt},
    {'Model': 'Decision Tree Hyperparameter', 'RMSE': train_rmse_best_dt, 'R2 Score': train_r2_best_dt},
    {'Model': 'Random Forest', 'RMSE': rf_bef_rmse, 'R2 Score': rf_bef_r2},
    {'Model': 'Random Forest Hyperparameter', 'RMSE': after_train_rmse_rf, 'R2 Score': after_train_r2_rf},
    {'Model': 'XBG', 'RMSE': xgb_bef_rmse, 'R2 Score': xgb_bef_r2},
    {'Model': 'XBG Hyperparamter', 'RMSE': xgb_after_rmse, 'R2 Score': xgb_after_r2}
    # {'Model': 'Random Forest', 'RMSE': rmse_rf, 'R2 Score': r2_rf},
    # {'Model': 'XGBoost', 'RMSE': rmse_xgb, 'R2 Score': r2_xgb},
    # Add more models here
]

# Convert into a DataFrame
performance_df = pd.DataFrame(model_performance)

# View it nicely
performance_df.sort_values(by='RMSE')

