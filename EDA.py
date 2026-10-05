import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("superstore.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"])
#chart 1 - monthly sales and profit
monthly = df.groupby(pd.Grouper(key="Order Date", freq="MS"))[["Sales","Profit"]].sum()
plt.figure(figsize=(10,4))
plt.plot(monthly.index,monthly["Sales"],label="Sales")
plt.plot(monthly.index,monthly["Profit"],label="Profit")
plt.title("Monthly Sales and Profits")
plt.legend()
plt.savefig("chart1_monthly.png")
plt.show()

#chart 2 - sales by category
plt.figure(figsize=(6, 4))
cat_profit=df.groupby("Category")["Profit"].sum()
cat_profit.plot(kind="bar")
plt.title("Profit by Category")
plt.ylabel("Profit")
plt.savefig("chart2_category.png")
plt.show()

#chart3 - profit by sub category
plt.figure(figsize=(8,6))
sub_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values()
sub_profit.plot(kind="barh")
plt.title("Profit by Sub-Category")
plt.savefig("chart3_subcategory.png")
plt.show()

#chart4 - discount vs profit
plt.figure(figsize=(6, 4))
sample = df.sample(5000, random_state=1)
sns.scatterplot(x="Discount", y="Profit", data=sample)
plt.title("Discount vs Profit")
plt.savefig("chart4_discount_profit.png")   
plt.show()

#chart5 - ship mode vs shippingdays
plt.figure(figsize=(6, 4))
sns.boxplot(x="Ship Mode", y="Shipping_Days", data=df)
plt.title("Ship Mode vs Shipping Days")
plt.savefig("chart5_shipmode_shippingdays.png")
plt.show()

#profit by market and segment
print(df.groupby("Market")["Profit"].sum())
print(df.groupby("Segment")["Profit"].sum())