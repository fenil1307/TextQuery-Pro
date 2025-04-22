import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
import os
import pandas as pd
import re
import sqlite3
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Text-to-SQL Pro", layout="wide")
st.title("🧠 Text-to-SQL Final Year Project – Chinook DB")

query_input = st.text_input("💬 Enter your natural language query:")
show_schema = st.checkbox("📂 Show DB Schema")
run_btn = st.button("⚡ Generate SQL and Execute")

# DB Setup
DB_PATH = r"C:\Users\DELL\py\chinook1.db"
conn = sqlite3.connect(DB_PATH)

# Show schema if checked
if show_schema:
    st.markdown("### 📋 Available Tables and Schema:")
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    for table in tables['name']:
        st.markdown(f"**{table}**")
        schema = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
        st.dataframe(schema)

# Define LangGraph state
class GraphState(TypedDict):
    question: str
    query: str
    result: str

# Extract clean SQL
def extract_sql(response_text):
    match = re.search(r"(?i)SQLQuery:\s*(SELECT.+)", response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()

# Generate SQL from NL
def generate_sql(state):
    llm_response = query_chain.invoke({"question": state["question"]})
    sql = extract_sql(llm_response)
    return {"query": sql}

# Execute SQL on DB
def query_db(state):
    sql = state["query"]
    try:
        df = pd.read_sql_query(sql, conn)
        return {"result": df}
    except Exception as e:
        raise ValueError(f"SQL Execution Error: {e}")

# Visualization Function
def plot_charts(df):
    if df.shape[1] < 2:
        st.warning("Not enough data for visualization.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Bar Chart")
        st.bar_chart(df.set_index(df.columns[0]))

    with col2:
        st.markdown("#### 📈 Line Chart")
        st.line_chart(df.set_index(df.columns[0]))

    if df.shape[1] == 2:
        st.markdown("#### 🥧 Pie Chart")
        plt.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%')
        st.pyplot(plt)

# Only run if query is submitted
if run_btn and query_input:
    try:
        # Load LLM
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Load DB into LangChain
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        query_chain = create_sql_query_chain(llm, db)

        # Build LangGraph
        workflow = StateGraph(GraphState)
        workflow.add_node("Generate SQL", generate_sql)
        workflow.add_node("Query DB", query_db)

        workflow.set_entry_point("Generate SQL")
        workflow.add_edge("Generate SQL", "Query DB")
        workflow.set_finish_point("Query DB")

        app = workflow.compile()

        # Run the workflow
        result = app.invoke({"question": query_input})
        query = result["query"]
        df = result["result"]

        st.success("✅ Query executed successfully!")
        st.markdown("### 🧾 SQL Query")
        st.code(query, language="sql")

        st.markdown("### 📈 Result Data")
        st.dataframe(df)

        # Result Summary
        st.markdown("### 🧠 Summary")
        st.write(f"The query returned **{len(df)} rows** and **{df.shape[1]} columns**.")
        st.write("Here’s a preview of your data and chart visualization.")

        # Charts
        plot_charts(df)

        # Download options
        csv = df.to_csv(index=False).encode('utf-8')
        excel = df.to_excel("result.xlsx", index=False)

        st.download_button("📥 Download CSV", csv, "result.csv", "text/csv")
        st.download_button("📥 Download Excel", data=open("result.xlsx", "rb").read(), file_name="result.xlsx")

    except Exception as e:
        st.error(f"❌ {e}")

elif run_btn:
    st.warning("Please enter a valid question to continue.")
