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

# Page and layout configuration
st.set_page_config(
    page_title="Text-to-SQL Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 4px 4px 0px 0px;
    }
    .chart-container {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
    }
    .result-container {
        background-color: #f5f7ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
    }
    .schema-container {
        background-color: #f0f8ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0px;
    }
    .table-relation {
        background-color: #fff8e1;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0px;
    }
    h1, h2, h3 {
        margin-bottom: 20px;
    }
    .stButton button {
        width: 100%;
    }
    .stExpander {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App header with improved styling
st.title("🧠 Text-to-SQL Final Year Project – Chinook DB")
st.markdown("Convert natural language to SQL queries and visualize results from the Chinook database.")

# DB Setup
DB_PATH = r"C:\Users\DELL\py\chinook1.db"
conn = sqlite3.connect(DB_PATH)

# Function to get table relationships (foreign keys)
def get_foreign_keys(conn, table_name):
    query = f"PRAGMA foreign_key_list({table_name});"
    try:
        foreign_keys = pd.read_sql_query(query, conn)
        if not foreign_keys.empty:
            return foreign_keys
        return None
    except:
        return None

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

# Create tabs with improved styling
tab1, tab2, tab3 = st.tabs(["📋 Suggested Questions", "💬 Custom Question", "📚 Database Schema"])

# Tab 1: Suggested Questions
with tab1:
    st.markdown("### Select from Sample Questions")
    selected_question = st.selectbox(
        "Choose a question from the list:",
        SAMPLE_QUESTIONS
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        use_selected = st.button("⚡ Run Selected Question", use_container_width=True)

# Tab 2: Custom Question
with tab2:
    st.markdown("### Enter Your Own Question")
    custom_question = st.text_input("Type your natural language query:")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_custom = st.button("⚡ Run Custom Question", use_container_width=True)

# Tab 3: Enhanced Database Schema
with tab3:
    st.markdown("### 📋 Chinook Database Schema")
    st.markdown("The Chinook database represents a digital media store, including tables for artists, albums, media tracks, invoices and customers.")
    
    # Get all tables
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;", conn)
    table_names = tables['name'].tolist()
    
    # General overview of database
    st.markdown('<div class="schema-container">', unsafe_allow_html=True)
    st.markdown("#### Database Overview")
    st.write(f"The database contains **{len(table_names)}** tables with relationships between them. Expand each table below to see its schema details.")
    
    # Table counts
    table_rows = {}
    for table in table_names:
        count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
        table_rows[table] = count
    
    # Create a DataFrame to show table counts
    table_info = pd.DataFrame({
        'Table Name': table_rows.keys(),
        'Row Count': table_rows.values()
    })
    st.dataframe(table_info, width=800)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display schema for each table with relationship info
    st.markdown("#### Detailed Table Schemas")
    
    # Option to expand all or select specific tables
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
    
    # Display each table schema
    for table in tables_to_show:
        with st.expander(f"Table: {table} ({table_rows[table]} rows)", expanded=show_all):
            st.markdown('<div class="schema-container">', unsafe_allow_html=True)
            
            # Get schema
            schema = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
            
            # Highlight primary keys
            pk_columns = schema[schema['pk'] > 0]['name'].tolist()
            if pk_columns:
                st.markdown(f"**Primary Key(s):** {', '.join(pk_columns)}")
            
            # Show table schema
            st.dataframe(schema, use_container_width=True)
            
            # Get sample data (first 5 rows)
            sample_data = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
            st.markdown("##### Sample Data")
            st.dataframe(sample_data, use_container_width=True)
            
            # Show foreign keys if they exist
            foreign_keys = get_foreign_keys(conn, table)
            if foreign_keys is not None and not foreign_keys.empty:
                st.markdown("##### Foreign Key Relationships")
                st.markdown('<div class="table-relation">', unsafe_allow_html=True)
                for _, fk in foreign_keys.iterrows():
                    st.markdown(f"• Column **{fk['from']}** references **{fk['table']}.{fk['to']}**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Database Relationship Diagram hint
    st.markdown('<div class="schema-container">', unsafe_allow_html=True)
    st.markdown("#### Database Entity Relationships")
    st.markdown("""
    The Chinook database has the following key relationships:
    
    - **Artist** → **Album** (One-to-Many): Each artist can have multiple albums
    - **Album** → **Track** (One-to-Many): Each album contains many tracks
    - **Genre** → **Track** (One-to-Many): Each genre can have multiple tracks
    - **MediaType** → **Track** (One-to-Many): Each media type can have multiple tracks
    - **Customer** → **Invoice** (One-to-Many): Each customer can have multiple invoices
    - **Invoice** → **InvoiceLine** (One-to-Many): Each invoice contains multiple line items
    - **InvoiceLine** → **Track** (Many-to-One): Each invoice line refers to a specific track
    - **Track** → **PlaylistTrack** (One-to-Many): Each track can be in multiple playlists
    - **Playlist** → **PlaylistTrack** (One-to-Many): Each playlist contains multiple tracks
    - **Employee** → **Customer** (One-to-Many): Each employee (sales agent) supports multiple customers
    """)
    st.markdown('</div>', unsafe_allow_html=True)

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

# Enhanced Visualization Function with consistent sizing
def plot_charts(df):
    # Check if we have enough data to visualize
    if len(df) == 0:
        st.warning("No data returned from query to visualize.")
        return
    
    # For single column data
    if df.shape[1] < 2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.warning("At least two columns are needed for full visualization. Try a query that returns multiple columns.")
        
        # Still show simple visualization for single column if it has numeric data
        if df.shape[1] == 1 and pd.api.types.is_numeric_dtype(df[df.columns[0]]):
            st.markdown("#### 📊 Single Column Chart")
            st.bar_chart(df, height=400, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # For multi-column data
    try:
        # Bar and Line charts with consistent sizing
        chart_height = 400
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📊 Bar Chart")
            st.bar_chart(df.set_index(df.columns[0]), height=chart_height, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📈 Line Chart")
            st.line_chart(df.set_index(df.columns[0]), height=chart_height, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Only show pie chart for appropriate data (2 columns, numeric second column)
        if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 🥧 Pie Chart")
            
            # Create pie chart with consistent size
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%')
            ax.axis('equal')  # Equal aspect ratio ensures circular pie
            
            # Display the chart with proper sizing
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
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
        
        with st.spinner("Generating SQL and executing query..."):
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
        
        # SQL Query with improved styling
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### 🧾 SQL Query")
        st.code(query, language="sql")
        st.markdown('</div>', unsafe_allow_html=True)

        # Result Data with improved styling
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### 📈 Result Data")
        st.dataframe(df, use_container_width=True)
        
        # Result Summary
        st.markdown("### 🧠 Summary")
        st.write(f"The query returned **{len(df)} rows** and **{df.shape[1]} columns**.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts with improved styling
        st.markdown("### 📊 Visualizations")
        plot_charts(df)

        # Download options with improved styling
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown("### 📥 Download Results")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "result.csv", "text/csv", use_container_width=True)
        
        with col2:
            # Try Excel download if openpyxl is available
            try:
                excel = df.to_excel("result.xlsx", index=False)
                st.download_button("Download Excel", data=open("result.xlsx", "rb").read(), 
                                  file_name="result.xlsx", use_container_width=True)
            except ImportError:
                st.warning("Excel download requires the 'openpyxl' package. Install it with 'pip install openpyxl'")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ {e}")