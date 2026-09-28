import io

import pandas as pd
import streamlit as st

from loader import load_data
from Categorizer import hybrid_categorize, refine_transfer_category
from analytics import monthly_spend
from anomaly import detect_anomalies
from agent import plan_query, execute_plan, generate_answer

VALID_CATEGORIES = ["Groceries", "Food and dining", "Transport", "Subscriptions",
                    "Rent", "Shopping", "Fees", "Income", "Transfer In",
                    "Transfer Out", "Education"]

NON_SPEND = ["Transfer In", "Transfer Out", "Income"]

st.set_page_config(page_title="Finance Agent", layout="wide")
st.title("Personal Finance Agent")
st.caption("Runs fully locally!!!")


@st.cache_data(show_spinner="Categorizing transactions (LLM fallback can take a minute)...")
def load_and_categorize(file_bytes: bytes) -> pd.DataFrame:
    # Caching means the slow LLM categorization only runs once per uploaded file,
    # not every time you click something in the UI.
    df = load_data(io.BytesIO(file_bytes))
    df["Category"] = df["Description"].apply(hybrid_categorize)
    df["Category"] = df.apply(refine_transfer_category, axis=1)
    return df


uploaded = st.sidebar.file_uploader("Upload bank statement CSV", type="csv")
if uploaded is None:
    st.info("Upload your statement CSV in the sidebar to get started.")
    st.stop()

df = load_and_categorize(uploaded.getvalue())

tab_overview, tab_anomalies, tab_ask, tab_data = st.tabs(
    ["Overview", "Anomalies", "Ask the agent", "Transactions"]
)

with tab_overview:
    exclude_big = st.checkbox(
        "Hide very large one-off payments (e.g. tuition) so everyday spending is readable",
        value=True,
    )
    spend = df[(df["Amount"] < 0) & (~df["Category"].isin(NON_SPEND))]
    if exclude_big:
        spend = spend[spend["Amount"].abs() < 2000]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total spending", f"${-spend['Amount'].sum():,.2f}")
    c2.metric("Transactions", len(spend))
    c3.metric("Uncategorized", int((df["Category"] == "Uncategorized").sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("Spending by category")
        by_cat = (-spend.groupby("Category")["Amount"].sum()).sort_values(ascending=False)
        st.bar_chart(by_cat)
    with right:
        st.subheader("Spending by month")
        by_month = -spend.assign(Month=spend["Date"].dt.to_period("M").astype(str)) \
                         .groupby("Month")["Amount"].sum()
        st.bar_chart(by_month)

with tab_anomalies:
    st.subheader("Unusual transactions (z-score)")
    st.dataframe(detect_anomalies(df.copy()), use_container_width=True)

with tab_ask:
    st.subheader("Ask a question about your spending")
    question = st.text_input("e.g. How much did I spend on food each month?")
    if st.button("Ask") and question.strip():
        with st.spinner("Planning..."):
            plan = plan_query(question, df)
        with st.expander("Agent plan (which tools it chose)"):
            st.json(plan)
        if not plan:
            st.warning("The model returned an unusable plan. Try rephrasing the question.")
        else:
            results = execute_plan(plan, df, VALID_CATEGORIES)
            with st.spinner("Writing answer..."):
                answer = generate_answer(question, results)
            st.success(answer)

with tab_data:
    st.dataframe(df, use_container_width=True)