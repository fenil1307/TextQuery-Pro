# TextQuery-Pro


An interactive Text-to-SQL web application that allows users to ask natural language questions and receive SQL-based answers with visualizations and summaries. Built using **LangChain**, **Groq API**, **Streamlit**, and the **Chinook** sample database.

##  Features

-  Text-to-SQL using LLM (Groq)
-  Visualization of results using Bar, Pie, and Line charts
-  Download query results as CSV or Excel
-  View database schema
-  Auto-generated query explanation and summary

##  Tech Stack

- **Backend**: Python, LangChain, SQLite (Chinook DB)
- **LLM Integration**: Groq API (llama3-70b-8192)
- **Frontend/UI**: Streamlit
- **Visualization**: Pandas, Matplotlib, Plotly

## Installation

1. **Clone the repo**
```bash
git clone https://github.com/fenil1307/TextQuery-Pro.git
cd TextQuery-Pro
