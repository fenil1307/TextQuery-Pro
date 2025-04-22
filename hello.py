import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
import os
import re

# Streamlit UI
st.set_page_config(page_title="Text-to-SQL Generator", layout="wide")
st.title("🧠 Natural Language to SQL using LangGraph + Groq")
st.markdown("Generate SQL queries from natural language questions using the Chinook database.")

query_input = st.text_input("💬 Enter your natural language query:")
run_btn = st.button("⚡ Generate SQL and Execute")

# Define app state for LangGraph
class GraphState(TypedDict):
    question: str
    query: str
    result: str

# Function to extract clean SQL query from model response
def extract_sql(response_text):
    match = re.search(r"(?i)SQLQuery:\s*(SELECT.+)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

# Generate SQL query from NL question
def generate_sql(state):
    llm_response = query_chain.invoke({"question": state["question"]})
    sql = extract_sql(llm_response)
    return {"query": sql}

# Run SQL on Chinook DB
def query_db(state):
    sql = state["query"]
    result = db.run(sql)
    return {"result": result}

# Only proceed if button clicked and query is provided
if run_btn and query_input:
    try:
        # Setup LLM from Groq (make sure your env var is set!)
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Connect to Chinook SQLite DB
        db = SQLDatabase.from_uri(r"sqlite:///C:\Users\DELL\py\chinook1.db")  # adjust path as needed

        # Create the LangChain SQL query chain
        query_chain = create_sql_query_chain(llm, db)

        # Build LangGraph
        workflow = StateGraph(GraphState)
        workflow.add_node("Generate SQL", generate_sql)
        workflow.add_node("Query DB", query_db)

        workflow.set_entry_point("Generate SQL")
        workflow.add_edge("Generate SQL", "Query DB")
        workflow.set_finish_point("Query DB")

        app = workflow.compile()

        # Invoke graph
        inputs = {"question": query_input}
        result = app.invoke(inputs)

        st.success("✅ Query executed successfully!")
        st.markdown("### SQL Query Generated:")
        st.code(result["query"], language="sql")
        st.markdown("### Output:")
        st.write(result["result"])

    except Exception as e:
        st.error(f"❌ Error: {e}")

elif run_btn:
    st.warning("Please enter a valid question to continue.")
