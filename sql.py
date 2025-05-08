import streamlit as st
from langchain_community.utilities import SQLDatabase
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
import time
from datetime import datetime

if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'successful_queries' not in st.session_state:
    st.session_state.successful_queries = 0
if 'rerun_query' not in st.session_state:
    st.session_state.rerun_query = None
if 'rerun_question' not in st.session_state:
    st.session_state.rerun_question = None

st.set_page_config(
    page_title="Text-to-SQL Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Text-to-SQL Query Executor")
st.markdown("Convert natural language to SQL queries and visualize results from the Chinook database.")


DB_PATH = r"/Users/fenilkheni/Desktop/final_report _sem8/wetransfer_sql-py_2025-04-19_0300/chinook1.db"
conn = sqlite3.connect(DB_PATH)

SAMPLE_QUESTIONS = [
    "Which artist has the most albums?",
    "What genre has the most tracks?",
    "Show me the longest songs in the database",
    "Which sales agent has generated the most revenue?",
    "Show me monthly sales for 2009",
    "What's the revenue breakdown by genre?",
    "Show me customers who purchased jazz tracks",
    "What's the most popular playlist?",
    "Show me tracks that appear on the most playlists",
    "Which artist appears most frequently in playlists?",
    "Compare revenue from different billing countries",
    "Show me the average invoice amount by country",
    "List the top 10 selling tracks of all time",
]


tab1, tab2, tab3, tab4 = st.tabs([" Suggested Questions", " Custom Question", " Database Schema", " Query History"])

with tab1:
    st.markdown("### Select from Sample Questions")
    selected_question = st.selectbox(
        "Choose a question from the list:",
        SAMPLE_QUESTIONS
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        use_selected = st.button("Run Selected Question", use_container_width=True)


with tab2:
    st.markdown("### Enter Your Own Question")
    custom_question = st.text_input("Type your natural language query:")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_custom = st.button(" Run Custom Question", use_container_width=True)


with tab3:
    st.markdown("###  Chinook Database Schema")
    st.markdown("The Chinook database represents a digital media store, including tables for artists, albums, media tracks, invoices and customers.")
    
  
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
    table_names = tables['name'].tolist()
    
    
    
    st.markdown("#### Database Overview")
    st.write(f"The database contains **{len(table_names)}** tables.")
    
    
    table_rows = {}
    for table in table_names:
        count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
        table_rows[table] = count
    
    
    table_info = pd.DataFrame({
        'Table Name': table_rows.keys(),
        'Row Count': table_rows.values()
    })
    st.dataframe(table_info, width=800)
    
    
    st.markdown("#### Detailed Table Schemas")
    
    
    show_all = st.checkbox("Expand all tables", value=False)
    
    if not show_all:
        selected_tables = st.multiselect(
            "Select tables to view schema:",
            options=table_names,
            default=[]
        )
        tables_to_show = selected_tables
    else:
        tables_to_show = table_names
    
    
    for table in tables_to_show:
        with st.expander(f"Table: {table} ({table_rows[table]} rows)", expanded=show_all):
            
            
           
            schema = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
            st.dataframe(schema)
            
            
            sample_data = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
            st.markdown("##### Sample Data")
            st.dataframe(sample_data)
              

with tab4:
    st.markdown("###  Query History")
    
   
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries Run", st.session_state.total_queries)
    with col2:
        success_rate = 100 * (st.session_state.successful_queries / st.session_state.total_queries) if st.session_state.total_queries > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    if not st.session_state.query_history:
        st.info("Your query history will appear here once you run some queries.")
    else:
        
        for i, history_item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Query {len(st.session_state.query_history) - i}: {history_item['question'][:50] + '...' if len(history_item['question']) > 50 else history_item['question']}"):
                st.markdown(f"**Question:** {history_item['question']}")
                st.markdown(f"**Execution Time:** {history_item['execution_time']:.2f} seconds")
                st.markdown(f"**Timestamp:** {history_item['timestamp']}")
                st.markdown("**SQL Query:**")
                st.code(history_item['query'], language="sql")
                  
                st.markdown(f"**Results:** {history_item['row_count']} rows returned")
                   
                if st.button(f" Rerun Query", key=f"rerun_{i}"):
                    st.session_state.rerun_query = history_item['query']
                    st.session_state.rerun_question = history_item['question']
                    st.rerun()
                    
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(" Clear History", use_container_width=True):
                st.session_state.query_history = []
                st.session_state.total_queries = 0
                st.session_state.successful_queries = 0
                st.success("History cleared successfully!")
                st.rerun()


class GraphState(TypedDict):
    question: str
    query: str
    result: str


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
        st.warning("At least two columns are needed for full visualization. Try a query that returns multiple columns.")
        
        
        if df.shape[1] == 1 and pd.api.types.is_numeric_dtype(df[df.columns[0]]):
            st.markdown("####  Single Column Chart")
            st.bar_chart(df, height=400, use_container_width=True)
        return

    
    try:
        
        chart_height = 400
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("####  Bar Chart")
            st.bar_chart(df.set_index(df.columns[0]), height=chart_height, use_container_width=True)
            

        with col2:
            st.markdown("####  Line Chart")
            st.line_chart(df.set_index(df.columns[0]), height=chart_height, use_container_width=True)

        
        if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
            st.markdown("####  Pie Chart")
    
    
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    
    
            wedges, texts, autotexts = ax.pie(
            df[df.columns[1]], 
            labels=None,  
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'width': 0.6}
          )
    
            ax.legend(
            wedges, 
            df[df.columns[0]], 
            title=df.columns[0],
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
           )
    
    
        for autotext in autotexts:
            autotext.set_color('#f9f9f9')  
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
    
            ax.axis('equal')  
    
    
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    
        st.pyplot(fig)
            
    except Exception as e:
        st.warning(f"Could not create Charts: {e}")


run_query = False
query_input = ""

if use_selected:
    run_query = True
    query_input = selected_question
elif run_custom and custom_question:
    run_query = True
    query_input = custom_question


if run_query or st.session_state.rerun_query:
    try:
        
        start_time = time.time()
        
        
        if st.session_state.rerun_query:
            query_input = st.session_state.rerun_question
            predefined_query = st.session_state.rerun_query
            st.session_state.rerun_query = None
            st.session_state.rerun_question = None
            is_rerun = True
        else:
            is_rerun = False
        
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error(" Missing GROQ_API_KEY.")
            st.stop()
            
        
        st.markdown("---")
        st.markdown(f"### Processing query: \"{query_input}\"")
        
        with st.spinner("Generating SQL and executing query"):
            if is_rerun:
                
                query = predefined_query
                df = pd.read_sql_query(query, conn)
            else:
               
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
            execution_time = time.time() - start_time
    
            st.session_state.total_queries += 1
            st.session_state.successful_queries += 1
            

            history_item = {
                'question': query_input,
                'query': query,
                'execution_time': execution_time,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'row_count': len(df),
                'column_count': df.shape[1]
            }
            st.session_state.query_history.append(history_item)
            
            
            if len(st.session_state.query_history) > 20:
                st.session_state.query_history = st.session_state.query_history[-20:]

        st.success(" Query executed successfully!")
        
        st.markdown("###  SQL Query")
        st.code(query, language="sql")
        st.markdown("###  Result Data")
        st.dataframe(df, use_container_width=True)
        
        st.markdown("###  Summary")
        st.write(f"The query returned **{len(df)} rows** and **{df.shape[1]} columns**.")
        
        st.markdown("###  Visualizations")
        plot_charts(df) 
        st.markdown("###  Download Results")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "result.csv", "text/csv", use_container_width=True)
        
        with col2:
            
            try:
                excel = df.to_excel("result.xlsx", index=False)
                st.download_button("Download Excel", data=open("result.xlsx", "rb").read(), 
                                  file_name="result.xlsx", use_container_width=True)
            except ImportError:
                st.warning("Excel download requires the 'openpyxl' package.'")
        

    except Exception as e:
        
        st.session_state.total_queries += 1
        st.error(f" {e}")
