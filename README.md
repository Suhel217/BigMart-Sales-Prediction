# BigMart-Sales-Prediction
"Predict sales at Big Mart outlets using machine learning models."

## 📋 Problem Statement
Big Mart, a retail company, wants to understand the factors that drive product sales and predict future sales for their products across multiple outlets.

## 📚 Project Steps

### 1. Data Cleaning
- Handled missing values (`Item_Weight`, `Outlet_Size`, `Item_Visibility`) using interpolation and logical imputation.
- Checked and verified no duplicate records.

### 2. Feature Engineering
- Created new features such as:
  - `Outlet_Type_Size_Location`
  - `Outlet_Age`
  - `Weight/Visibility`
  - `Category-wise MRP Mean`
  - `Item_Outlet_Visibility_Deviation`
- Extracted item category from `Item_Identifier`.

### 3. Feature Encoding and Scaling
- Label Encoding performed on categorical features.
- StandardScaler applied on numerical features after splitting Train/Test.

### 4. Model Building and Tuning
- Trained models: 
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree
  - Random Forest
  - XGBoost Regressor
- Hyperparameter tuning via RandomizedSearchCV with 5-fold Cross-Validation.

### 5. Evaluation Metrics
- RMSE (Root Mean Squared Error)
- R² Score

Random Forest achieved the best performance with the lowest RMSE and highest R² score.

### 6. Final Prediction and Submission
- Best model used for test prediction.
- Final submission file created with `Item_Identifier`, `Outlet_Identifier`, and predicted `Item_Outlet_Sales`.

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn
- XGBoost

## 📦 Repository Structure
