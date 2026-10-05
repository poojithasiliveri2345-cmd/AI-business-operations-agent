import pandas as pd 
df = pd.read_csv("superstore.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])

#making monthly data for  each category
monthly = df.groupby(["Category",pd.Grouper(key="Order Date", freq="MS")])[["Sales","Profit"]].sum()
monthly = monthly.reset_index()
monthly = monthly.sort_values(by=["Category","Order Date"])
print(monthly.head())

#lag features (previous 3 months values)
for i in [1,2,3]:
    monthly["Sales_lag"+str(i)] = monthly.groupby("Category")["Sales"].shift(i)
    monthly["Profit_lag"+str(i)] = monthly.groupby("Category")["Profit"].shift(i)

monthly["Month"] = monthly["Order Date"].dt.month

#first 3 months will have null values because no previous data
monthly = monthly.dropna()

#one hot encoding for category
dummies = pd.get_dummies(monthly["Category"], prefix="Cat", dtype=int)
final = pd.concat([monthly,dummies], axis=1)

print(final.shape)
print(final.head())

final.to_csv("monthly_feature.csv", index=False)