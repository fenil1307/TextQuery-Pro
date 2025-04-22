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


st.set_page_config(page_title="Text-to-SQL Pro", layout="wide")
st.title("🧠 Text-to-SQL Executor With LangChain – Chinook DB")

query_input = st.text_input("💬 Enter your natural language query:")
run_btn = st.button("⚡ Generate SQL and Execute")

# DB Setup
DB_PATH = r"/Users/fenilkheni/Desktop/final_report _sem8/wetransfer_sql-py_2025-04-19_0300/chinook1.db"
conn = sqlite3.connect(DB_PATH)




class GraphState(TypedDict):
    question: str
    query: str
    result: str

# Extract clean SQL
def extract_sql(response_text):
    match = re.search(r"(?i)SQLQuery:\s*(SELECT.+)", response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()


def generate_sql(state):
    llm_response = query_chain.invoke({"question": state["question"]})
    sql = extract_sql(llm_response)
    return {"query": sql}


def query_db(state):
    sql = state["query"]
    try:
        df = pd.read_sql_query(sql, conn)
        return {"result": df}
    except Exception as e:
        raise ValueError(f"SQL Execution Error: {e}")


def plot_charts(df):
   
    if len(df) == 0:
        st.warning("No data returned from query to visualize.")
        return
    
    if df.shape[1] < 2:
        st.warning("At least two columns are needed for visualization. Try a query that returns multiple columns.")
        

        if df.shape[1] == 1 and pd.api.types.is_numeric_dtype(df[df.columns[0]]):
            st.markdown("#### 📊 Single Column Bar Chart")
            st.bar_chart(df)
        return

    
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Bar Chart")
            st.bar_chart(df.set_index(df.columns[0]))

        with col2:
            st.markdown("#### 📈 Line Chart")
            st.line_chart(df.set_index(df.columns[0]))

        
        if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
            st.markdown("#### 🥧 Pie Chart")
            fig, ax = plt.subplots()
            ax.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%')
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not create some visualizations: {e}")


if run_btn and query_input:
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("❌ GROQ_API_KEY not found in environment variables. Please set your API key.")
            st.stop()
            
   
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=api_key
        )

       
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        query_chain = create_sql_query_chain(llm, db)

       
        workflow = StateGraph(GraphState)
        workflow.add_node("Generate SQL", generate_sql)
        workflow.add_node("Query DB", query_db)

        workflow.set_entry_point("Generate SQL")
        workflow.add_edge("Generate SQL", "Query DB")
        workflow.set_finish_point("Query DB")

        app = workflow.compile()

        
        result = app.invoke({"question": query_input})
        query = result["query"]
        df = result["result"]

        st.success("✅ Query executed successfully!")
        st.markdown("### 🧾 SQL Query")
        st.code(query, language="sql")

        st.markdown("### 📈 Result Data")
        st.dataframe(df)

       
        st.markdown("### 🧠 Summary")
        st.write(f"The query returned **{len(df)} rows** and **{df.shape[1]} columns**.")
        
       
        st.markdown("### 📊 Visualizations")
        plot_charts(df)

       
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "result.csv", "text/csv")
        
       
        try:
            excel = df.to_excel("result.xlsx", index=False)
            st.download_button("📥 Download Excel", data=open("result.xlsx", "rb").read(), file_name="result.xlsx")
        except ImportError:
            st.warning("Excel download requires the 'openpyxl' package. Install it with 'pip install openpyxl'")

    except Exception as e:
        st.error(f"❌ {e}")

elif run_btn:
    st.warning("Please enter a valid question to continue.")