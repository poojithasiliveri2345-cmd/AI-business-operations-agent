import pandas as pd
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# loading everything we saved before
model = pickle.load(open("model.pkl", "rb"))
feature_cols = pickle.load(open("feature_cols.pkl", "rb"))
data = pd.read_csv("monthly_feature.csv")
data["Order Date"] = pd.to_datetime(data["Order Date"])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# function to search policy
def ask_policy(question):
    results = db.similarity_search(question, k=2)
    answer = ""
    for r in results:
        answer = answer + r.page_content + "\n\n"
    return answer


# function to forecast next month profit
def forecast(category):
    d = data[data["Category"] == category].sort_values("Order Date")
    last3 = d.tail(3)
    last_month = d["Order Date"].max().month

    new_row = {}
    new_row["Sales_lag1"] = last3["Sales"].iloc[2]
    new_row["Sales_lag2"] = last3["Sales"].iloc[1]
    new_row["Sales_lag3"] = last3["Sales"].iloc[0]
    new_row["Profit_lag1"] = last3["Profit"].iloc[2]
    new_row["Profit_lag2"] = last3["Profit"].iloc[1]
    new_row["Profit_lag3"] = last3["Profit"].iloc[0]
    new_row["Month"] = last_month % 12 + 1
    new_row["Cat_Furniture"] = 1 if category == "Furniture" else 0
    new_row["Cat_Office Supplies"] = 1 if category == "Office Supplies" else 0
    new_row["Cat_Technology"] = 1 if category == "Technology" else 0

    X = pd.DataFrame([new_row])[feature_cols]
    predicted = model.predict(X)[0]
    old_avg = last3["Profit"].mean()
    change = (predicted - old_avg) / abs(old_avg) * 100
    return predicted, old_avg, change


# agent - combines forecast and policy
def run_agent(category):
    predicted, old_avg, change = forecast(category)

    result = category + " - predicted profit next month: " + str(round(predicted)) + "\n"
    result = result + "Last 3 months average profit: " + str(round(old_avg)) + "\n"
    result = result + "Change: " + str(round(change, 1)) + "%\n\n"

    if change < -5:
        result = result + "Profit is going down. Maybe discounts or shipping cost is high.\n"
        result = result + "Recommendation: check the discount policy below\n\n"
        result = result + ask_policy("discount approval rules and negative profit review for " + category)
    else:
        result = result + "Profit looks fine, no action needed.\n"
    return result


#for c in ["Furniture", "Technology", "Office Supplies"]:
   # print(run_agent(c))
   # print("=" * 50)