#importing libraries
import pandas as pd 

#load data 

df = pd.read_csv("superstore.csv")
print(df.head())
print(df.shape)
print(df.info())

#checking for missing values
print(df.isnull().sum())
#csv saves dates as text so convert to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

print(df["Order Date"].min())
print(df["Order Date"].max())

print(df["Category"].value_counts())
print(df["Segment"].value_counts())
print(df["Market"].value_counts())
print(df[["Sales","Profit","Discount","Shipping Cost"]].describe()) 