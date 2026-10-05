import pickle

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

df = pd.read_csv("monthly_feature.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])

feature_cols = ["Sales_lag1","Sales_lag2","Sales_lag3","Profit_lag1","Profit_lag2","Profit_lag3","Month","Cat_Furniture","Cat_Office Supplies","Cat_Technology"]

#train test split by date?(not random because its time data)
df= df.sort_values("Order Date")
split_data = df["Order Date"].iloc[int(len(df)*0.8)]
train = df[df["Order Date"] <  split_data]
test = df[df["Order Date"] >= split_data]
print("train rows:",len(train))
print("test rows:",len(test))

X_train = train[feature_cols]
y_train = train["Profit"]
X_test = test[feature_cols]
y_test = test["Profit"]

#model 1 - linear regression(baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

#model 2 - randomforest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

#model 3 - xgboost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

#checking results
def show(name,pred):
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(name, "RMSE:", round(rmse, 2), "MAE:", round(mae, 2), "R2:", round(r2, 3))

show("Linear Regression", lr_pred)
show("Random Forest", rf_pred)
show("XGBoost", xgb_pred)

#saving random forest model(it gave best result)
pickle.dump(rf,open("model.pkl","wb"))
pickle.dump(feature_cols,open("feature_cols.pkl","wb"))