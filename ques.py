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
import random

# Page setup
st.set_page_config(page_title="Text-to-SQL Pro", layout="wide")
st.title("🧠 Text-to-SQL Executor With LangChain – Chinook DB")

# DB Setup
DB_PATH = r"C:\Users\DELL\py\chinook1.db"
conn = sqlite3.connect(DB_PATH)

# Define sample questions
SAMPLE_QUESTIONS = [
    "Show me all albums in the database",
    "How many tracks are in the database?",
    "List all customers from Germany",
    "What's the total sales revenue?",
    "Which country generates the most revenue?",
    "Who are the top 5 spending customers?",
    "Which artist has the most albums?",
    "What genre has the most tracks?",
    "Show me the longest songs in the database",
    "Which albums have the most tracks?",
    "What's the average track length by genre?",
    "Which sales agent has generated the most revenue?",
    "Show me monthly sales for 2009",
    "What's the revenue breakdown by genre?",
    "Show me customers who purchased jazz tracks",
    "Which employee supports the most customers?",
    "What's the most popular playlist?",
    "Show me tracks that appear on the most playlists",
    "Which artist appears most frequently in playlists?",
    "Compare revenue from different billing countries",
    "Show me the average invoice amount by country",
    "List the top 10 selling tracks of all time",
    "Which customer has spent the most money on classical music?",
    "Show me sales trends by quarter for each year",
    "What's the distribution of track lengths across different genres?"
]

# Create tabs for different query input methods
tab1, tab2, tab3 = st.tabs(["📋 Suggested Questions", "💬 Custom Question", "📚 Database Schema"])

# Tab 1: Suggested Questions
with tab1:
    st.markdown("### Select from Sample Questions")
    selected_question = st.selectbox(
        "Choose a question from the list:",
        SAMPLE_QUESTIONS
    )
    use_selected = st.button("⚡ Run Selected Question")

# Tab 2: Custom Question
with tab2:
    st.markdown("### Enter Your Own Question")
    custom_question = st.text_input("Type your natural language query:")
    run_custom = st.button("⚡ Run Custom Question")

# Tab 3: Database Schema
with tab3:
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

# Enhanced Visualization Function
def plot_charts(df):
    # Check if we have enough data to visualize
    if len(df) == 0:
        st.warning("No data returned from query to visualize.")
        return
    
    if df.shape[1] < 2:
        st.warning("At least two columns are needed for full visualization. Try a query that returns multiple columns.")
        
        # Still show simple visualization for single column if it has numeric data
        if df.shape[1] == 1 and pd.api.types.is_numeric_dtype(df[df.columns[0]]):
            st.markdown("#### 📊 Single Column Bar Chart")
            st.bar_chart(df)
        return

    # If we have multiple columns, proceed with visualizations
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Bar Chart")
            st.bar_chart(df.set_index(df.columns[0]))

        with col2:
            st.markdown("#### 📈 Line Chart")
            st.line_chart(df.set_index(df.columns[0]))

        # Only show pie chart for appropriate data (2 columns, numeric second column)
        if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
            st.markdown("#### 🥧 Pie Chart")
            fig, ax = plt.subplots()
            ax.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%')
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not create some visualizations: {e}")

# Determine which question to process
run_query = False
query_input = ""

if use_selected:
    run_query = True
    query_input = selected_question
elif run_custom and custom_question:
    run_query = True
    query_input = custom_question

# Main query processing logic
if run_query:
    try:
        # Check for API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("❌ GROQ_API_KEY not found in environment variables. Please set your API key.")
            st.stop()
            
        # Show the query being processed
        st.markdown("---")
        st.markdown(f"### Processing query: \"{query_input}\"")
        
        # Load LLM
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=api_key
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
        
        # Charts
        st.markdown("### 📊 Visualizations")
        plot_charts(df)

        # Download options
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "result.csv", "text/csv")
        
        # Try Excel download if openpyxl is available
        try:
            excel = df.to_excel("result.xlsx", index=False)
            st.download_button("📥 Download Excel", data=open("result.xlsx", "rb").read(), file_name="result.xlsx")
        except ImportError:
            st.warning("Excel download requires the 'openpyxl' package. Install it with 'pip install openpyxl'")

    except Exception as e:
        st.error(f"❌ {e}")