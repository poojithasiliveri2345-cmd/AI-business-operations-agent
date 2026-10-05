import streamlit as st
import matplotlib.pyplot as plt
from agent import data, run_agent, ask_policy

st.title("AI Business Operations Agent")

#forecasting section
st.header("Forecast")
category = st.selectbox("Select Category", ["Furniture","Office Supplies", "Technology"])

d = data[data["Category"] == category]
fig, ax = plt.subplots()
ax.plot(d["Order Date"], d["Profit"])
ax.set_title("Monthly Profit - " + category)
st.pyplot(fig)

if st.button("Run Agent"):
    st.text(run_agent(category))

#chatbot section
st.header("Policy Chatbot")
question = st.text_input("Ask a quetion about shipping, returns or discounts")
if question:
    st.text(ask_policy(question))    
