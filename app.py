import os
import sys
import re
import requests
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sec_edgar_api import EdgarClient
from prettytable import PrettyTable
import json
import traceback

# For LangChain components
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_community.llms import HuggingFaceHub

# Set the environment variable to allow deserialization
os.environ["STREAMLIT_ALLOW_DANGEROUS_DESERIALIZATION"] = "true"

# Silence BS4's HTML parser warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ——— THIS must come FIRST ———
st.set_page_config(page_title="Financial Chatbot", page_icon="💰", layout="wide")

# Configuration
TICKER_CIKS = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
    "MSFT": {"cik": "0000789019", "name": "Microsoft Corporation"},
    "GOOGL": {"cik": "0001652044", "name": "Alphabet Inc."},
    "AMZN": {"cik": "0001018724", "name": "Amazon.com, Inc."},
    "META": {"cik": "0001326801", "name": "Meta Platforms, Inc."}
}

# Define paths for locally saved financial data
DATA_DIR = "financial_data"

# For GitHub deployment, we need to store the FAISS indexes
INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'messages_chart' not in st.session_state:
    st.session_state.messages_chart = []

if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

if 'ticker_info' not in st.session_state:
    st.session_state.ticker_info = {}

if 'xbrl_data' not in st.session_state:
    st.session_state.xbrl_data = None

if 'financial_data' not in st.session_state:
    st.session_state.financial_data = {}

if 'selected_statement' not in st.session_state:
    st.session_state.selected_statement = "Balance Sheet"

if 'start_year' not in st.session_state:
    st.session_state.start_year = 2020

if 'end_year' not in st.session_state:
    st.session_state.end_year = 2024

if 'indexes_loaded' not in st.session_state:
    st.session_state.indexes_loaded = {}

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# Initialize embeddings model
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# Initialize LLM
@st.cache_resource
def get_llm():
    return HuggingFaceHub(
        huggingfacehub_api_token="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",
        repo_id="meta-llama/Llama-3.1-8B-Instruct",   # HF-hosted Llama-3 B chat model
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": 2048
        }
    )

# LLM Prompt - Using your test4.py format
PROMPT = PromptTemplate(
    input_variables=["narrative_context", "xbrl_context", "table_context", "question"],
    template=("""
        You are a financial analyst. Use ONLY the narrative excerpts, the financial facts and the relevant tables provided below to answer the question. Try your absolute best to write at least 300 words unless specified in the question. \n\n Try your best to include any numerical values found in the financial facts or Relevant tables. The Relevant tables have poor structure and need to be analyzed properly to make sure they are being read right.
        Use a professional, analytical tone and focus on providing insights that would be valuable for financial decision-making.

        When analyzing the financial data:
        - Highlight key metrics and ratios relevant to the question
        - Consider both short and long-term implications
        - Analyze trends and year-over-year comparisons when appropriate
        - Assess potential impacts on profitability, liquidity, and solvency
        - Acknowledge market conditions and economic factors
        - Present balanced views on investment opportunities
        - When appropriate, include quantitative analysis with specific numbers from the context
        - Relate the analysis to industry benchmarks or competitors when possible
        - When analyzing tables be sure to analyze the context before the tables to understand what kind of table and what the units are for the values.
        - Cite specific data points from the context when appropriate
        - format all the markdowns correctly and make sure they are readable
              
        Narrative Excerpts:
        {narrative_context}

        Financial Facts:
        {xbrl_context}

        Relevant tables:
        {table_context}

        Question: {question}
        
        Answer:
        """
    )
)

# Statement analysis prompt
STATEMENT_PROMPT = PromptTemplate(
    input_variables=["statement_type", "statement_data", "question"],
    template=("""
        You are an expert financial analyst focused specifically on {statement_type} analysis.

        Here is the {statement_type} data:
        {statement_data}

        Based on this information, please answer the following question:
        {question}

        Focus your analysis specifically on the data points in the {statement_type}. 
        If calculations are needed, show your work.
        If the question can't be answered using the available data, explain why and what additional information might be needed.
        Keep your response focused and concise.
        Make sure to format the response properly without any bold or big font sizes
        Always end with 'END OF ANALYSIS'
        
        """
    )
)

def load_cached_financial_data(ticker):
    """Load financial data from cached files"""
    ticker_data = {}
    company_dir = os.path.join(DATA_DIR, ticker)
    
    # If data directory doesn't exist, return empty dict
    if not os.path.exists(company_dir):
        return ticker_data
    
    # Try to load stock history
    stock_history_path = os.path.join(company_dir, "stock_history.csv")
    if os.path.exists(stock_history_path):
        try:
            stock_history = pd.read_csv(stock_history_path)
            stock_history['date'] = pd.to_datetime(stock_history['date'])
            ticker_data['stock_history'] = stock_history
        except Exception as e:
            st.error(f"Error loading stock history for {ticker}: {e}")
    
    # Try to load income statement
    income_path = os.path.join(company_dir, "income_statement.csv")
    if os.path.exists(income_path):
        try:
            income = pd.read_csv(income_path)
            # Convert date column to datetime
            income['fiscalDateEnding'] = pd.to_datetime(income['fiscalDateEnding'])
            ticker_data['income_statement'] = income
        except Exception as e:
            st.error(f"Error loading income statement for {ticker}: {e}")
    
    # Try to load balance sheet
    balance_path = os.path.join(company_dir, "balance_sheet.csv")
    if os.path.exists(balance_path):
        try:
            balance = pd.read_csv(balance_path)
            # Convert date column to datetime
            balance['fiscalDateEnding'] = pd.to_datetime(balance['fiscalDateEnding'])
            ticker_data['balance_sheet'] = balance
        except Exception as e:
            st.error(f"Error loading balance sheet for {ticker}: {e}")
    
    # Try to load cash flow statement
    cash_flow_path = os.path.join(company_dir, "cash_flow.csv")
    if os.path.exists(cash_flow_path):
        try:
            cash_flow = pd.read_csv(cash_flow_path)
            # Convert date column to datetime
            cash_flow['fiscalDateEnding'] = pd.to_datetime(cash_flow['fiscalDateEnding'])
            ticker_data['cash_flow'] = cash_flow
        except Exception as e:
            st.error(f"Error loading cash flow statement for {ticker}: {e}")
    
    # Try to load company overview
    overview_path = os.path.join(company_dir, "company_overview.json")
    if os.path.exists(overview_path):
        try:
            with open(overview_path, 'r') as f:
                overview = json.load(f)
            ticker_data['company_overview'] = overview
        except Exception as e:
            st.error(f"Error loading company overview for {ticker}: {e}")
    
    return ticker_data

def print_pretty_table_ignore_empty(table):
    """
    Given a BeautifulSoup <table> tag, prints it as an ASCII table:
    - expands colspan
    - drops blank rows
    - drops columns that are blank throughout
    - uses the first non-blank row as header if no <th> present
    - ensures header names are unique (and non-empty)
    """
    # 1) Build a raw grid, expanding colspan
    grid = []
    max_cols = 0

    for tr in table.find_all("tr"):
        row = []
        for cell in tr.find_all(["th", "td"]):
            text = cell.get_text(strip=True)
            span = int(cell.get("colspan", 1))
            row.extend([text] * span)
        if any(cell for cell in row):  # skip entirely blank rows
            grid.append(row)
            max_cols = max(max_cols, len(row))

    if not grid:
        return None

    # 2) Pad rows to same width
    for row in grid:
        row += [""] * (max_cols - len(row))

    # 3) Figure out which columns have any non-blank cell
    keep = [
        any(row[c].strip() for row in grid)
        for c in range(max_cols)
    ]

    # 4) Filter out blank columns
    filtered = [
        [row[c] for c in range(max_cols) if keep[c]]
        for row in grid
    ]

    # 5) Split header vs data
    th_cells = table.find_all("th")
    if th_cells:
        header = [th.get_text(strip=True) for th in th_cells]
        data_rows = filtered
    else:
        header = filtered.pop(0)
        data_rows = filtered

    # 6) Normalize header: fill empties and make all names unique
    seen = {}
    unique_header = []
    for idx, name in enumerate(header):
        # if blank, give it a generic name
        name = name or f"Column{idx+1}"
        # if we've seen it before, append a suffix
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        unique_header.append(name)

    # 7) Build & print
    pt = PrettyTable()
    pt.field_names = unique_header
    for row in data_rows:
        pt.add_row(row)

    return pt

def fetch_recent_filing_meta(cik: str) -> Dict:
    """Fetch metadata for the most recent 10-K filing"""
    client = EdgarClient(user_agent="FinancialChatbot (your.email@example.com)")
    subs = client.get_submissions(cik=cik)
    recent = subs["filings"]["recent"]
    
    filing_info = None
    for form, acc, doc, date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["filingDate"]
    ):
        if form == "10-K":
            filing_info = {
                "accession": acc, 
                "primary_doc": doc, 
                "filing_date": date,
                "form": form
            }
            break
    
    if not filing_info:
        raise ValueError(f"No 10-K found for CIK {cik}")
    
    return filing_info

def download_filing_html(cik: str, accession: str, primary_doc: str) -> (str, str):
    """Download a filing's HTML and return (html, base_url)"""
    cik_int = str(int(cik))
    path = accession.replace("-", "")
    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{path}/{primary_doc}"
    
    # Add delay to respect SEC's rate limiting
    time.sleep(0.1)
    
    resp = requests.get(base_url, headers={"User-Agent": "FinancialChatbot (your.email@example.com)"})
    resp.raise_for_status()
    return resp.text, base_url

def narrative_to_docs(
    html: str,
    cik: str,
    filing_date: str,
    base_url: str,
    company_name: str
) -> tuple[List[Document], List[Document]]:
    """
    Parses the HTML into:
    - n_docs: narrative chunks from <div> (5-at-a-time)
    - t_docs: table chunks rendered via print_pretty_table_ignore_empty,
             plus 8 divs before and 8 divs after each table for context
    Returns (n_docs, t_docs).
    """
    soup = BeautifulSoup(html, "html.parser")
    paras = soup.find_all("div")
    tables = soup.find_all("table")
    cik_str = cik.zfill(10)
    n_docs: List[Document] = []
    t_docs: List[Document] = []
    
    # Narrative chunks
    for i in range(0, len(paras), 5):
        group = paras[i : i + 5]
        text = "\n\n".join(p.get_text().strip() for p in group if p.get_text().strip())
        if not text:
            continue
        fragment = next(
            (tag[attr] for tag in group for attr in ("id", "name") if tag.has_attr(attr)),
            None
        )
        source_url = base_url + (f"#{fragment}" if fragment else "")
        n_docs.append(Document(
            page_content=text,
            metadata={
                "entity": cik_str,
                "company_name": company_name,
                "filing_date": filing_date,
                "chunk_type": "narrative",
                "chunk_id": i // 5,
                "source_url": source_url
            }
        ))
    
    # Map divs to their positions in the document
    all_elements = soup.find_all(['div', 'table'])
    element_positions = {}
    for idx, element in enumerate(all_elements):
        element_positions[element] = idx
    
    # Table chunks with context (8 divs before and 8 divs after)
    for idx, table in enumerate(tables):
        # Only tables with >1 row
        rows = table.find_all("tr")
        if len(rows) <= 1:
            continue
        
        # Render table via pretty-table function
        table_str = print_pretty_table_ignore_empty(table)
        if not table_str:
            continue
            
        # Get table position in the document
        table_pos = element_positions.get(table)
        if table_pos is None:
            continue
            
        # Find context divs before and after
        context_divs_before = []
        context_divs_after = []
        
        # Get divs before table
        count_before = 0
        for i in range(table_pos - 1, -1, -1):
            if i < 0 or count_before >= 4:
                break
            if all_elements[i].name == 'div' and all_elements[i].get_text().strip():
                context_divs_before.append(all_elements[i])
                count_before += 1
        
        # Get divs after table
        count_after = 0
        for i in range(table_pos + 1, len(all_elements)):
            if i >= len(all_elements) or count_after >= 8:
                break
            if all_elements[i].name == 'div' and all_elements[i].get_text().strip():
                context_divs_after.append(all_elements[i])
                count_after += 1
        
        # Combine context and table content
        context_before_text = "\n\n".join(div.get_text().strip() for div in reversed(context_divs_before))
        context_after_text = "\n\n".join(div.get_text().strip() for div in context_divs_after)
        
        combined_content = ""
        if context_before_text:
            combined_content += "CONTEXT BEFORE TABLE:\n" + context_before_text + "\n\n"
            
        combined_content += "TABLE CONTENT:\n" + str(table_str) + "\n\n"
        
        if context_after_text:
            combined_content += "CONTEXT AFTER TABLE:\n" + context_after_text
        
        # Get fragment for source URL
        fragment = table.get("id") or table.get("name")
        source_url = base_url + (f"#{fragment}" if fragment else "")
        
        t_docs.append(Document(
            page_content=combined_content,
            metadata={
                "entity": cik_str,
                "company_name": company_name,
                "filing_date": filing_date,
                "chunk_type": "table_with_context",
                "chunk_id": idx,
                "source_url": source_url
            }
        ))
    
    return n_docs, t_docs

def xbrl_facts_to_docs(cik: str, company_name: str) -> List[Document]:
    """Extract XBRL facts into Document objects"""
    client = EdgarClient(user_agent="FinancialChatbot (your.email@example.com)")
    raw = client.get_company_facts(cik=cik)
    docs, seen = [], set()
    cik_str = str(raw["cik"]).zfill(10)
    
    # Extract company name from API if not provided
    if not company_name:
        company_name = raw.get("entityName", "Unknown")

    for _, concepts in raw["facts"].items():
        for concept, entry in concepts.items():
            # grab the description once per concept
            description = entry.get("description", "").strip() if entry.get("description", "") else None
            for unit, vals in entry["units"].items():
                for fact in vals:
                    key = (concept, fact["end"], unit)
                    if key in seen:
                        continue
                    seen.add(key)

                    # build page_content with description
                    desc_part = f" — {description}" if description else ""
                    content = (
                        f"{concept}{desc_part} as of {fact['end']}: "
                        f"{fact['val']} {unit}"
                    )

                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "tag": concept,
                            "period": fact["end"],
                            "unit": unit,
                            "entity": cik_str,
                            "company_name": company_name,
                            "chunk_type": "xbrl_fact",
                            "val": str(fact["val"])  # Store value as string
                        }
                    ))
    return docs

def build_indexes(ticker, cik):
    """Build three separate FAISS indexes for a company"""
    # Show progress
    progress_text = f"Building vector indexes for {ticker}. This may take a few minutes..."
    progress_bar = st.progress(0, text=progress_text)
    
    st.info(progress_text)
    embeddings = get_embeddings_model()
    
    try:
        # Step 1: Process XBRL data (15%)
        progress_bar.progress(0.05, text="Fetching XBRL facts...")
        company_info = fetch_company_info(cik)
        company_name = company_info["name"]
        
        x_docs = xbrl_facts_to_docs(cik, company_name)
        progress_bar.progress(0.15, text="Processing XBRL facts...")
        
        # Limit to 3000 docs to prevent memory issues in cloud deployment
        if len(x_docs) > 3000:
            x_docs = x_docs[:3000]
        
        x_idx = FAISS.from_documents(x_docs, embeddings)
        
        # Step 2: Process narrative and tables (40%)
        progress_bar.progress(0.25, text="Fetching 10-K filing...")
        meta = fetch_recent_filing_meta(cik)
        html, base_url = download_filing_html(cik, meta["accession"], meta["primary_doc"])
        
        progress_bar.progress(0.35, text="Processing narrative and tables...")
        n_docs, t_docs = narrative_to_docs(html, cik, meta["filing_date"], base_url, company_name)
        
        progress_bar.progress(0.5, text="Building narrative index...")
        n_idx = FAISS.from_documents(n_docs, embeddings)
        
        progress_bar.progress(0.65, text="Building table index...")
        t_idx = FAISS.from_documents(t_docs, embeddings)
        
        # Step 3: Save indexes in memory (20%)
        progress_bar.progress(0.75, text="Saving indexes...")
        
        # Create directory for this ticker
        ticker_dir = os.path.join(INDEX_DIR, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # Store in session state (no pickle files)
        st.session_state[f'{ticker}_n_idx'] = n_idx
        st.session_state[f'{ticker}_x_idx'] = x_idx
        st.session_state[f'{ticker}_t_idx'] = t_idx
        
        # Save company metadata
        ticker_info = {
            "ticker": ticker,
            "cik": cik,
            "company_name": company_name,
            "recent_filing_date": meta["filing_date"],
            "recent_filing_url": base_url,
            "last_updated": time.strftime("%Y-%m-%d")
        }
        
        # Save ticker info 
        with open(os.path.join(ticker_dir, "info.json"), "w") as f:
            json.dump(ticker_info, f)
        
        progress_bar.progress(1.0, text="Processing complete!")
        time.sleep(1)  # Show the completion for a moment
        
        # Store ticker info in session state
        st.session_state.ticker_info[ticker] = ticker_info
        
        # Mark as loaded
        st.session_state.indexes_loaded[ticker] = True
        
        return True
    
    except Exception as e:
        progress_bar.progress(1.0, text=f"Error: {str(e)}")
        st.error(f"Error building indexes for {ticker}: {str(e)}")
        return False

def load_indexes(ticker):
    """Load indexes for a ticker if they exist"""
    if ticker in st.session_state.indexes_loaded and st.session_state.indexes_loaded[ticker]:
        # Already loaded in this session
        return True
    
    ticker_dir = os.path.join(INDEX_DIR, ticker)
    
    if not os.path.exists(ticker_dir):
        # Need to build indexes first
        return False
    
    try:
        # Load ticker info
        info_path = os.path.join(ticker_dir, "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                import json
                st.session_state.ticker_info[ticker] = json.load(f)
        
        # Mark as loaded
        st.session_state.indexes_loaded[ticker] = True
        return True
    
    except Exception as e:
        st.error(f"Error loading indexes for {ticker}: {str(e)}")
        return False

def fetch_company_info(cik: str) -> Dict:
    """Fetch basic company information using the submissions endpoint"""
    client = EdgarClient(user_agent="FinancialChatbot (your.email@example.com)")
    try:
        # Get submissions data which includes company name
        subs = client.get_submissions(cik=cik)
        company_name = subs.get("name", "Unknown Company")
        
        return {
            "cik": cik,
            "name": company_name
        }
    except Exception as e:
        st.error(f"Error fetching company info for {cik}: {str(e)}")
        return {
            "cik": cik,
            "name": "Unknown Company"
        }

def set_current_ticker(ticker):
    """Set the current ticker and load data"""
    if ticker == st.session_state.current_ticker:
        return
    
    # Check if indexes exist or need to be built
    indexes_exist = load_indexes(ticker)
    
    # Check if we have the indexes in memory
    indexes_in_memory = (
        f'{ticker}_n_idx' in st.session_state and 
        f'{ticker}_x_idx' in st.session_state and 
        f'{ticker}_t_idx' in st.session_state
    )
    
    if not indexes_exist or not indexes_in_memory:
        # Need to build indexes
        cik = TICKER_CIKS[ticker]["cik"]
        success = build_indexes(ticker, cik)
        
        if not success:
            st.error(f"Failed to process {ticker}. Please try again.")
            return
    
    # Update current ticker
    st.session_state.current_ticker = ticker
    
    # Reset selected statement
    st.session_state.selected_statement = "Balance Sheet"
    
    # Clear chat history
    st.session_state.messages = []
    st.session_state.messages_chart = []
    
    # Load XBRL data (from SEC filings)
    load_financial_data(ticker)
    
    # Load cached financial data (Alpha Vantage)
    st.session_state.financial_data[ticker] = load_cached_financial_data(ticker)

def load_financial_data(ticker):
    """Load financial data for charts from XBRL facts"""
    # Check if we have the index in memory
    x_idx_key = f'{ticker}_x_idx'
    if x_idx_key not in st.session_state:
        st.warning(f"XBRL index not found for {ticker}. Charts may not be available.")
        st.session_state.xbrl_data = None
        return
    
    x_idx = st.session_state[x_idx_key]
    
    try:
        # Important financial metrics to look for
        important_metrics = [
            "us-gaap:Revenue", 
            "us-gaap:NetIncomeLoss",
            "us-gaap:Assets",
            "us-gaap:Liabilities",
            "us-gaap:StockholdersEquity",
            "us-gaap:OperatingIncomeLoss",
            "us-gaap:EarningsPerShareBasic",
            "us-gaap:CashAndCashEquivalentsAtCarryingValue"
        ]
        
        # Get all XBRL docs
        result = x_idx.similarity_search("financial metrics", k=1000)
        all_docs = result
        
        # Filter for important metrics
        filtered_docs = [doc for doc in all_docs if any(metric in doc.page_content for metric in important_metrics)]
        
        # Process into dataframe
        rows = []
        for doc in filtered_docs:
            meta = doc.metadata
            content = doc.page_content
            
            # Extract metric from content
            metric_match = re.match(r"([\w\-:]+)", content)
            if not metric_match:
                continue
                
            metric = metric_match.group(1)
            
            # Only process important metrics
            if not any(m in metric for m in important_metrics):
                continue
            
            # Extract value using regex
            val_match = re.search(r"as of [\d\-]+: ([^\s]+)", content)
            if not val_match:
                continue
                
            val_str = val_match.group(1)
            try:
                val = float(val_str)
            except:
                continue
            
            # Parse period
            period = meta.get("period")
            try:
                period_date = datetime.strptime(period, "%Y-%m-%d")
                # Filter by selected year range
                year = period_date.year
                if year < st.session_state.start_year or year > st.session_state.end_year:
                    continue
            except:
                continue
            
            rows.append({
                "metric": metric,
                "period": period,
                "period_date": period_date,
                "value": val,
                "unit": meta.get("unit", "")
            })
        
        # Create dataframe
        if rows:
            df = pd.DataFrame(rows)
            # Sort by date
            df = df.sort_values("period_date")
            st.session_state.xbrl_data = df
        else:
            st.session_state.xbrl_data = None
            
    except Exception as e:
        st.error(f"Error loading financial data: {str(e)}")
        st.session_state.xbrl_data = None

def get_xbrl_items_by_statement_type(ticker, statement_type):
    """Get XBRL facts for a specific statement type"""
    if st.session_state.xbrl_data is None:
        return []
    
    df = st.session_state.xbrl_data
    
    # Filter by statement type
    if statement_type == "Balance Sheet":
        keywords = ["Assets", "Liabilities", "Equity", "Cash", "Inventory", "Receivable", "Debt"]
    elif statement_type == "Income Statement":
        keywords = ["Revenue", "Income", "Earnings", "Profit", "Loss", "Sales", "Cost", "Expense", "Tax", "EPS"]
    elif statement_type == "Cash Flow Statement":
        keywords = ["Cash", "Flow", "Operating", "Investing", "Financing", "Dividend", "Expenditure", "CAPEX"]
    else:
        return []
    
    # Filter data by keywords
    filtered_items = []
    for _, row in df.iterrows():
        if any(keyword.lower() in row["metric"].lower() for keyword in keywords):
            filtered_items.append({
                "metric": row["metric"],
                "period": row["period"],
                "value": row["value"],
                "unit": row["unit"]
            })
    
    return filtered_items

def query_10k(ticker, question):
    """Query the 10-K with a question and get an answer"""
    n_idx = st.session_state.get(f'{ticker}_n_idx')
    x_idx = st.session_state.get(f'{ticker}_x_idx')
    t_idx = st.session_state.get(f'{ticker}_t_idx')
    
    if n_idx is None or x_idx is None or t_idx is None:
        return {
            "answer": "Index data not available. Please rebuild the index for this company.",
            "narrative_sources": [],
            "xbrl_sources": [],
            "table_sources": []
        }
    
    try:
        # Create retrievers
        n_ret = n_idx.as_retriever(search_kwargs={"k": 8})
        x_ret = x_idx.as_retriever(search_kwargs={"k": 20})
        t_ret = t_idx.as_retriever(search_kwargs={"k": 2})
        
        # Get relevant documents
        n_docs = n_ret.get_relevant_documents(question)
        x_docs = x_ret.get_relevant_documents(question)
        t_docs = t_ret.get_relevant_documents(question)
        
        # Combine contexts
        n_ctx = "\n\n".join(d.page_content for d in n_docs)
        x_ctx = "\n\n".join(d.page_content for d in x_docs)
        t_ctx = "\n\n".join(d.page_content for d in t_docs)
        
        # Get the LLM
        llm = get_llm()
        
        # Create the chain
        chain = LLMChain(llm=llm, prompt=PROMPT)
        
        # Run the chain
        result = chain.invoke({
            "narrative_context": n_ctx,
            "xbrl_context": x_ctx,
            "table_context": t_ctx,
            "question": question
        })
        print(result["text"])
        # Return the answer and sources
        return {
            "answer": result["text"].split("Answer:")[1],
            "narrative_sources": [{"metadata": d.metadata, "content": d.page_content} for d in n_docs],
            "xbrl_sources": [{"metadata": d.metadata, "content": d.page_content} for d in x_docs],
            "table_sources": [{"metadata": d.metadata, "content": d.page_content} for d in t_docs]
        }
        
    except Exception as e:
        st.error(f"Error querying 10-K: {str(e)}")
        return {
            "answer": f"I encountered an error: {str(e)}",
            "narrative_sources": [],
            "xbrl_sources": [],
            "table_sources": []
        }

def query_statement_data(ticker, statement_type, question):
    """Generate insights about a specific financial statement"""
    try:
        # Get the LLM
        llm = get_llm()
        
        # Get statement data
        ticker_data = st.session_state.financial_data.get(ticker, {})
        
        # Get the appropriate statement data
        statement_data = None
        if statement_type == "Income Statement" and 'income_statement' in ticker_data:
            statement_data = ticker_data['income_statement']
        elif statement_type == "Balance Sheet" and 'balance_sheet' in ticker_data:
            statement_data = ticker_data['balance_sheet']
        elif statement_type == "Cash Flow Statement" and 'cash_flow' in ticker_data:
            statement_data = ticker_data['cash_flow']
            
        # If Alpha Vantage data not available, try XBRL data
        if statement_data is None or statement_data.empty:
            # Try XBRL data as fallback
            xbrl_items = get_xbrl_items_by_statement_type(ticker, statement_type)
            
            if xbrl_items:
                # Format items as text
                statement_text = "\n".join([
                    f"{item['metric']} as of {item['period']}: {item['value']} {item['unit']}"
                    for item in xbrl_items
                ])
            else:
                return f"I don't have enough {statement_type.lower()} data to answer this question."
        else:
            # Filter by year range
            mask = (
                (statement_data['fiscalDateEnding'].dt.year >= st.session_state.start_year) &
                (statement_data['fiscalDateEnding'].dt.year <= st.session_state.end_year)
            )
            filtered_data = statement_data[mask]
            
            if filtered_data.empty:
                return f"I don't have {statement_type.lower()} data for the selected year range ({st.session_state.start_year}-{st.session_state.end_year})."
                
            # Format as text
            statement_text = filtered_data.to_string()
        
        # Create the chain
        chain = LLMChain(llm=llm, prompt=STATEMENT_PROMPT)
        
        # Generate the answer
        result = chain.invoke({
            "statement_type": statement_type,
            "statement_data": statement_text,
            "question": question
        })
        print(result["text"])
        return result["text"].split("END OF ANALYSIS")[-1]
        
    except Exception as e:
        st.error(f"Error analyzing {statement_type}: {str(e)}")
        return f"I encountered an error while analyzing the {statement_type}: {str(e)}"

def create_financial_charts():
    """Create financial charts from XBRL data"""
    if st.session_state.xbrl_data is None or not isinstance(st.session_state.xbrl_data, pd.DataFrame) or st.session_state.xbrl_data.empty:
        st.info("No financial data available from SEC filings for this company.")
        return
    
    df = st.session_state.xbrl_data
    
    try:
        # Chart 1: Revenue and Net Income over time
        st.subheader("Revenue and Net Income")
        
        # Filter data for revenue and net income
        revenue_df = df[df["metric"].str.contains("Revenue")].copy()
        income_df = df[df["metric"].str.contains("NetIncomeLoss")].copy()
        
        if not revenue_df.empty and not income_df.empty:
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add revenue bars
            fig.add_trace(go.Bar(
                x=revenue_df["period"],
                y=revenue_df["value"],
                name="Revenue",
                marker_color='rgb(55, 83, 109)'
            ))
            
            # Add net income line
            fig.add_trace(go.Scatter(
                x=income_df["period"],
                y=income_df["value"],
                name="Net Income",
                marker_color='rgb(26, 118, 255)',
                mode='lines+markers'
            ))
            
            # Update layout
            fig.update_layout(
                title='Revenue and Net Income Over Time',
                xaxis_tickfont_size=12,
                yaxis=dict(
                    title='USD',
                    titlefont_size=14,
                    tickfont_size=12,
                ),
                legend=dict(
                    x=0,
                    y=1.0,
                    bgcolor='rgba(255, 255, 255, 0)',
                    bordercolor='rgba(255, 255, 255, 0)'
                ),
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue or Net Income data not available.")
        
        # Chart 2: Assets, Liabilities, and Equity
        st.subheader("Balance Sheet Overview")
        
        # Filter data
        assets_df = df[df["metric"].str.contains("Assets")].copy()
        liabilities_df = df[df["metric"].str.contains("Liabilities")].copy()
        equity_df = df[df["metric"].str.contains("StockholdersEquity")].copy()
        
        if not assets_df.empty and not liabilities_df.empty and not equity_df.empty:
            # Create a new dataframe for the stacked bar chart
            balance_df = pd.DataFrame()
            
            # Ensure we have matching dates
            common_dates = set(assets_df["period"]) & set(liabilities_df["period"]) & set(equity_df["period"])
            
            # Prepare data
            balance_data = []
            for date in common_dates:
                assets = assets_df[assets_df["period"] == date]["value"].values[0]
                liabilities = liabilities_df[liabilities_df["period"] == date]["value"].values[0]
                equity = equity_df[equity_df["period"] == date]["value"].values[0]
                
                balance_data.append({"period": date, "category": "Assets", "value": assets})
                balance_data.append({"period": date, "category": "Liabilities", "value": liabilities})
                balance_data.append({"period": date, "category": "Equity", "value": equity})
            
            balance_df = pd.DataFrame(balance_data)
            
            if not balance_df.empty:
                # Create grouped bar chart
                fig = px.bar(
                    balance_df,
                    x="period",
                    y="value",
                    color="category",
                    barmode="group",
                    title="Assets, Liabilities, and Equity",
                    labels={"value": "USD", "period": "Period", "category": "Category"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Balance sheet data not available for comparison.")
        else:
            st.info("Complete balance sheet data not available.")
        
        # Chart 3: EPS over time
        st.subheader("Earnings Per Share")
        
        eps_df = df[df["metric"].str.contains("EarningsPerShare")].copy()
        
        if not eps_df.empty:
            fig = px.line(
                eps_df,
                x="period",
                y="value",
                markers=True,
                title="Earnings Per Share (Basic)",
                labels={"value": "USD per Share", "period": "Period"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("EPS data not available.")
            
        # Chart 4: Operating Income
        st.subheader("Operating Income")
        
        op_income_df = df[df["metric"].str.contains("OperatingIncomeLoss")].copy()
        
        if not op_income_df.empty:
            fig = px.bar(
                op_income_df,
                x="period",
                y="value",
                title="Operating Income",
                labels={"value": "USD", "period": "Period"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Operating income data not available.")
            
    except Exception as e:
        st.error(f"Error creating financial charts: {str(e)}")
        st.info("Could not generate charts due to an error with the financial data.")

def create_stock_chart(ticker):
    """Create stock price chart from cached data"""
    # Get stock price data
    ticker_data = st.session_state.financial_data.get(ticker, {})
    stock_history = ticker_data.get('stock_history')
    
    if stock_history is None or not isinstance(stock_history, pd.DataFrame) or stock_history.empty:
        st.info("No stock price history available.")
        return
    
    try:
        # Filter by year range
        mask = (
            (stock_history['date'].dt.year >= st.session_state.start_year) &
            (stock_history['date'].dt.year <= st.session_state.end_year)
        )
        filtered_data = stock_history[mask]
        
        if filtered_data.empty:
            st.info(f"No stock price data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
            return
        
        # Create stock price chart
        st.subheader("Stock Price History")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['adjusted_close'],
            mode='lines',
            name='Adjusted Close',
            line=dict(color='royalblue')
        ))
        
        fig.update_layout(
            title=f"{ticker} Stock Price ({st.session_state.start_year}-{st.session_state.end_year})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create volume chart
        st.subheader("Trading Volume")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=filtered_data['date'],
            y=filtered_data['volume'],
            marker_color='lightblue',
            name='Volume'
        ))
        
        fig.update_layout(
            title=f"{ticker} Trading Volume ({st.session_state.start_year}-{st.session_state.end_year})",
            xaxis_title="Date",
            yaxis_title="Volume",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating stock chart: {e}")
        st.info("Could not generate stock price chart due to an error.")

def filter_statement_by_year(df):
    """Filter a financial statement dataframe by the selected year range"""
    if df is None or df.empty:
        return df
    
    if 'fiscalDateEnding' not in df.columns:
        return df
    
    mask = (
        (df['fiscalDateEnding'].dt.year >= st.session_state.start_year) & 
        (df['fiscalDateEnding'].dt.year <= st.session_state.end_year)
    )
    
    return df[mask]

def display_financial_statements(statement_type=None):
    """Display financial statements from cached Alpha Vantage data"""
    ticker = st.session_state.current_ticker
    
    if not ticker:
        st.info("No company selected.")
        return
    
    ticker_data = st.session_state.financial_data.get(ticker, {})
    
    try:
        # If a specific statement is requested, show only that one
        if statement_type:
            if statement_type == "Income Statement":
                income_statement = ticker_data.get('income_statement')
                
                if income_statement is not None and not income_statement.empty:
                    # Filter by year range
                    filtered_data = filter_statement_by_year(income_statement)
                    
                    if not filtered_data.empty:
                        # Transpose for better display
                        st.dataframe(filtered_data, use_container_width=True)
                    else:
                        st.info(f"No income statement data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                else:
                    st.info("Income statement data not available.")
            
            elif statement_type == "Balance Sheet":
                balance_sheet = ticker_data.get('balance_sheet')
                
                if balance_sheet is not None and not balance_sheet.empty:
                    # Filter by year range
                    filtered_data = filter_statement_by_year(balance_sheet)
                    
                    if not filtered_data.empty:
                        # Transpose for better display
                        st.dataframe(filtered_data, use_container_width=True)
                    else:
                        st.info(f"No balance sheet data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                else:
                    st.info("Balance sheet data not available.")
            
            elif statement_type == "Cash Flow Statement":
                cash_flow = ticker_data.get('cash_flow')
                
                if cash_flow is not None and not cash_flow.empty:
                    # Filter by year range
                    filtered_data = filter_statement_by_year(cash_flow)
                    
                    if not filtered_data.empty:
                        # Transpose for better display
                        st.dataframe(filtered_data, use_container_width=True)
                    else:
                        st.info(f"No cash flow data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                else:
                    st.info("Cash flow data not available.")
            
            else:
                st.warning(f"Unknown statement type: {statement_type}")
        
        # Otherwise, use tabs to show all statements
        else:
            # Create tabs for different statements
            tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            
            # Income Statement
            with tabs[0]:
                income_statement = ticker_data.get('income_statement')
                
                if income_statement is not None and not income_statement.empty:
                    # Filter by year range
                    filtered_data = filter_statement_by_year(income_statement)
                    
                    if not filtered_data.empty:
                        st.dataframe(filtered_data, use_container_width=True)
                    else:
                        st.info(f"No income statement data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                else:
                    st.info("Income statement data not available.")
            
            # Balance Sheet
            with tabs[1]:
                balance_sheet = ticker_data.get('balance_sheet')
                
                if balance_sheet is not None and not balance_sheet.empty:
                    # Filter by year range
                    filtered_data = filter_statement_by_year(balance_sheet)
                    
                    if not filtered_data.empty:
                        st.dataframe(filtered_data, use_container_width=True)
                    else:
                        st.info(f"No balance sheet data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                else:
                    st.info("Balance sheet data not available.")
            
            # Cash Flow
            with tabs[2]:
                cash_flow = ticker_data.get('cash_flow')
                
                if cash_flow is not None and not cash_flow.empty:
                    # Filter by year range
                    filtered_data = filter_statement_by_year(cash_flow)
                    
                    if not filtered_data.empty:
                        st.dataframe(filtered_data, use_container_width=True)
                    else:
                        st.info(f"No cash flow data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                else:
                    st.info("Cash flow data not available.")
    
    except Exception as e:
        st.error(f"Error displaying financial statements: {str(e)}")
        st.info("Could not display financial statements due to an error.")

def create_statement_charts(statement_type):
    """Create charts based on the selected financial statement from Alpha Vantage data"""
    ticker = st.session_state.current_ticker
    
    if not ticker:
        st.info("No company selected.")
        return
    
    ticker_data = st.session_state.financial_data.get(ticker, {})
    
    try:
        if statement_type == "Income Statement":
            income_statement = ticker_data.get('income_statement')
            
            if income_statement is None or income_statement.empty:
                st.info("No income statement data available.")
                return
                
            # Filter by year range
            filtered_data = filter_statement_by_year(income_statement)
            
            if filtered_data.empty:
                st.info(f"No income statement data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                return
                
            # Extract key metrics
            key_metrics = {
                'Revenue': 'totalRevenue',
                'Gross Profit': 'grossProfit',
                'Operating Income': 'operatingIncome',
                'Net Income': 'netIncome',
                'EPS': 'dilutedEPS'
            }
            
            # Create chart data
            chart_data = []
            
            # Get the column names that exist in the dataframe
            existing_columns = [col for col in key_metrics.values() if col in filtered_data.columns]
            
            if not existing_columns:
                st.info("No key metrics found in income statement data.")
                return
                
            # Prepare data for chart
            for col in existing_columns:
                metric_name = next((k for k, v in key_metrics.items() if v == col), col)
                
                for _, row in filtered_data.iterrows():
                    fiscal_year = row['fiscalDateEnding'].year
                    value = row[col]
                    
                    chart_data.append({
                        "Year": str(fiscal_year),
                        "Metric": metric_name,
                        "Value": value
                    })
            
            if chart_data:
                # Convert to DataFrame
                chart_df = pd.DataFrame(chart_data)
                
                # Create Revenue and Net Income chart
                st.subheader("Revenue and Net Income")
                
                rev_income_metrics = ['Revenue', 'Net Income']
                rev_income_data = chart_df[chart_df['Metric'].isin(rev_income_metrics)]
                
                if not rev_income_data.empty:
                    fig = px.bar(
                        rev_income_data,
                        x="Year",
                        y="Value",
                        color="Metric",
                        barmode="group",
                        title="Revenue and Net Income Over Time",
                        labels={"Value": "USD", "Year": "Fiscal Year"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Revenue and Net Income data not available.")
                
                # Create Profitability Metrics chart
                st.subheader("Profitability Metrics")
                
                profit_metrics = ['Gross Profit', 'Operating Income', 'Net Income']
                profit_data = chart_df[chart_df['Metric'].isin(profit_metrics)]
                
                if not profit_data.empty:
                    fig = px.line(
                        profit_data,
                        x="Year",
                        y="Value",
                        color="Metric",
                        markers=True,
                        title="Profitability Metrics Over Time",
                        labels={"Value": "USD", "Year": "Fiscal Year"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Profitability metrics not available.")
                
                # Calculate and chart profit margins
                if 'totalRevenue' in filtered_data.columns:
                    st.subheader("Profit Margins")
                    
                    margin_data = []
                    
                    # Calculate margins
                    for _, row in filtered_data.iterrows():
                        fiscal_year = row['fiscalDateEnding'].year
                        total_revenue = row['totalRevenue']
                        
                        if total_revenue != 0:
                            if 'grossProfit' in row.index and pd.notna(row['grossProfit']):
                                gross_margin = (row['grossProfit'] / total_revenue) * 100
                                margin_data.append({
                                    "Year": str(fiscal_year),
                                    "Metric": "Gross Margin",
                                    "Value": gross_margin
                                })
                            
                            if 'operatingIncome' in row.index and pd.notna(row['operatingIncome']):
                                operating_margin = (row['operatingIncome'] / total_revenue) * 100
                                margin_data.append({
                                    "Year": str(fiscal_year),
                                    "Metric": "Operating Margin",
                                    "Value": operating_margin
                                })
                            
                            if 'netIncome' in row.index and pd.notna(row['netIncome']):
                                net_margin = (row['netIncome'] / total_revenue) * 100
                                margin_data.append({
                                    "Year": str(fiscal_year),
                                    "Metric": "Net Margin",
                                    "Value": net_margin
                                })
                    
                    if margin_data:
                        margin_df = pd.DataFrame(margin_data)
                        fig = px.line(
                            margin_df,
                            x="Year",
                            y="Value",
                            color="Metric",
                            markers=True,
                            title="Profit Margins Over Time",
                            labels={"Value": "Percentage (%)", "Year": "Fiscal Year"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Could not calculate profit margins from available data.")
            else:
                st.info("Insufficient data for income statement analysis.")
                
        elif statement_type == "Balance Sheet":
            balance_sheet = ticker_data.get('balance_sheet')
            
            if balance_sheet is None or balance_sheet.empty:
                st.info("No balance sheet data available.")
                return
                
            # Filter by year range
            filtered_data = filter_statement_by_year(balance_sheet)
            
            if filtered_data.empty:
                st.info(f"No balance sheet data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                return
                
            # Extract key metrics
            key_metrics = {
                'Total Assets': 'totalAssets',
                'Current Assets': 'totalCurrentAssets',
                'Total Liabilities': 'totalLiabilities',
                'Current Liabilities': 'totalCurrentLiabilities',
                'Total Shareholder Equity': 'totalShareholderEquity'
            }
            
            # Create chart data
            chart_data = []
            
            # Get the column names that exist in the dataframe
            existing_columns = [col for col in key_metrics.values() if col in filtered_data.columns]
            
            if not existing_columns:
                st.info("No key metrics found in balance sheet data.")
                return
                
            # Prepare data for chart
            for col in existing_columns:
                metric_name = next((k for k, v in key_metrics.items() if v == col), col)
                
                for _, row in filtered_data.iterrows():
                    fiscal_year = row['fiscalDateEnding'].year
                    value = row[col]
                    
                    chart_data.append({
                        "Year": str(fiscal_year),
                        "Metric": metric_name,
                        "Value": value
                    })
            
            if chart_data:
                # Convert to DataFrame
                chart_df = pd.DataFrame(chart_data)
                
                # Create Assets and Liabilities chart
                st.subheader("Assets and Liabilities")
                
                asset_liability_metrics = ['Total Assets', 'Total Liabilities', 'Total Shareholder Equity']
                al_data = chart_df[chart_df['Metric'].isin(asset_liability_metrics)]
                
                if not al_data.empty:
                    fig = px.bar(
                        al_data,
                        x="Year",
                        y="Value",
                        color="Metric",
                        barmode="group",
                        title="Assets, Liabilities, and Equity Over Time",
                        labels={"Value": "USD", "Year": "Fiscal Year"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Assets and Liabilities data not available.")
                
                # Create Current vs. Long-term components chart
                if ('totalCurrentAssets' in filtered_data.columns and 
                    'totalCurrentLiabilities' in filtered_data.columns and
                    'totalAssets' in filtered_data.columns and 
                    'totalLiabilities' in filtered_data.columns):
                    
                    st.subheader("Current vs. Long-term Components")
                    
                    current_data = []
                    for _, row in filtered_data.iterrows():
                        fiscal_year = row['fiscalDateEnding'].year
                        
                        # Calculate non-current components
                        non_current_assets = row['totalAssets'] - row['totalCurrentAssets']
                        non_current_liabilities = row['totalLiabilities'] - row['totalCurrentLiabilities']
                        
                        # Add current assets
                        current_data.append({
                            "Year": str(fiscal_year),
                            "Component": "Current Assets",
                            "Value": row['totalCurrentAssets']
                        })
                        
                        # Add non-current assets
                        current_data.append({
                            "Year": str(fiscal_year),
                            "Component": "Non-current Assets",
                            "Value": non_current_assets
                        })
                        
                        # Add current liabilities
                        current_data.append({
                            "Year": str(fiscal_year),
                            "Component": "Current Liabilities",
                            "Value": row['totalCurrentLiabilities']
                        })
                        
                        # Add non-current liabilities
                        current_data.append({
                            "Year": str(fiscal_year),
                            "Component": "Non-current Liabilities",
                            "Value": non_current_liabilities
                        })
                    
                    if current_data:
                        current_df = pd.DataFrame(current_data)
                        fig = px.bar(
                            current_df,
                            x="Year",
                            y="Value",
                            color="Component",
                            barmode="group",
                            title="Current vs. Non-current Components",
                            labels={"Value": "USD", "Year": "Fiscal Year"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Current vs. Non-current component data not available.")
                
                # Calculate and chart key ratios
                if ('totalCurrentAssets' in filtered_data.columns and 
                    'totalCurrentLiabilities' in filtered_data.columns and
                    'totalAssets' in filtered_data.columns and 
                    'totalLiabilities' in filtered_data.columns and
                    'totalShareholderEquity' in filtered_data.columns):
                    
                    st.subheader("Financial Ratios")
                    
                    ratio_data = []
                    for _, row in filtered_data.iterrows():
                        fiscal_year = row['fiscalDateEnding'].year
                        
                        # Current ratio
                        if row['totalCurrentLiabilities'] != 0:
                            current_ratio = row['totalCurrentAssets'] / row['totalCurrentLiabilities']
                            ratio_data.append({
                                "Year": str(fiscal_year),
                                "Ratio": "Current Ratio",
                                "Value": current_ratio
                            })
                        
                        # Debt-to-equity ratio
                        if row['totalShareholderEquity'] != 0:
                            debt_equity_ratio = row['totalLiabilities'] / row['totalShareholderEquity']
                            ratio_data.append({
                                "Year": str(fiscal_year),
                                "Ratio": "Debt-to-Equity",
                                "Value": debt_equity_ratio
                            })
                        
                        # Debt-to-assets ratio
                        if row['totalAssets'] != 0:
                            debt_assets_ratio = row['totalLiabilities'] / row['totalAssets']
                            ratio_data.append({
                                "Year": str(fiscal_year),
                                "Ratio": "Debt-to-Assets",
                                "Value": debt_assets_ratio
                            })
                    
                    if ratio_data:
                        ratio_df = pd.DataFrame(ratio_data)
                        fig = px.line(
                            ratio_df,
                            x="Year",
                            y="Value",
                            color="Ratio",
                            markers=True,
                            title="Key Financial Ratios Over Time",
                            labels={"Value": "Ratio", "Year": "Fiscal Year"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Could not calculate financial ratios from available data.")
            else:
                st.info("Insufficient data for balance sheet analysis.")
                
        elif statement_type == "Cash Flow Statement":
            cash_flow = ticker_data.get('cash_flow')
            
            if cash_flow is None or cash_flow.empty:
                st.info("No cash flow statement data available.")
                return
                
            # Filter by year range
            filtered_data = filter_statement_by_year(cash_flow)
            
            if filtered_data.empty:
                st.info(f"No cash flow data available for the selected year range ({st.session_state.start_year}-{st.session_state.end_year}).")
                return
                
            # Extract key metrics
            key_metrics = {
                'Operating Cash Flow': 'operatingCashflow',
                'Capital Expenditure': 'capitalExpenditures',
                'Cash Flow from Investment': 'cashflowFromInvestment',
                'Cash Flow from Financing': 'cashflowFromFinancing',
                'Free Cash Flow': 'operatingCashflow'  # We'll calculate this
            }
            
            # Check which columns are available
            existing_columns = [col for col in key_metrics.values() if col in filtered_data.columns]
            
            if not existing_columns:
                st.info("No key metrics found in cash flow data.")
                return
                
            # Create chart data
            chart_data = []
            
            # Prepare data for CF components chart
            cf_components = []
            
            for _, row in filtered_data.iterrows():
                fiscal_year = row['fiscalDateEnding'].year
                
                # Add operating cash flow
                if 'operatingCashflow' in row.index and pd.notna(row['operatingCashflow']):
                    cf_components.append({
                        "Year": str(fiscal_year),
                        "Component": "Operating Cash Flow",
                        "Value": row['operatingCashflow']
                    })
                
                # Add cash flow from investment
                if 'cashflowFromInvestment' in row.index and pd.notna(row['cashflowFromInvestment']):
                    cf_components.append({
                        "Year": str(fiscal_year),
                        "Component": "Cash Flow from Investment",
                        "Value": row['cashflowFromInvestment']
                    })
                
                # Add cash flow from financing
                if 'cashflowFromFinancing' in row.index and pd.notna(row['cashflowFromFinancing']):
                    cf_components.append({
                        "Year": str(fiscal_year),
                        "Component": "Cash Flow from Financing",
                        "Value": row['cashflowFromFinancing']
                    })
                
                # Calculate free cash flow (FCF = OCF - CapEx)
                if ('operatingCashflow' in row.index and 
                    'capitalExpenditures' in row.index and 
                    pd.notna(row['operatingCashflow']) and 
                    pd.notna(row['capitalExpenditures'])):
                    
                    free_cash_flow = row['operatingCashflow'] - abs(row['capitalExpenditures'])
                    chart_data.append({
                        "Year": str(fiscal_year),
                        "Metric": "Free Cash Flow",
                        "Value": free_cash_flow
                    })
                
                # Add operating cash flow to chart_data
                if 'operatingCashflow' in row.index and pd.notna(row['operatingCashflow']):
                    chart_data.append({
                        "Year": str(fiscal_year),
                        "Metric": "Operating Cash Flow",
                        "Value": row['operatingCashflow']
                    })
                
                # Add capital expenditure to chart_data (as positive value for chart)
                if 'capitalExpenditures' in row.index and pd.notna(row['capitalExpenditures']):
                    chart_data.append({
                        "Year": str(fiscal_year),
                        "Metric": "Capital Expenditure",
                        "Value": abs(row['capitalExpenditures'])
                    })
            
            # Create cash flow components chart
            if cf_components:
                st.subheader("Cash Flow Components")
                
                cf_df = pd.DataFrame(cf_components)
                fig = px.bar(
                    cf_df,
                    x="Year",
                    y="Value",
                    color="Component",
                    barmode="group",
                    title="Cash Flow Components Over Time",
                    labels={"Value": "USD", "Year": "Fiscal Year"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cash flow component data not available.")
            
            # Create operating cash flow and free cash flow chart
            if chart_data:
                st.subheader("Operating & Free Cash Flow")
                
                # Convert to DataFrame
                cf_metrics = ['Operating Cash Flow', 'Free Cash Flow']
                ocf_fcf_data = pd.DataFrame(chart_data)
                ocf_fcf_data = ocf_fcf_data[ocf_fcf_data['Metric'].isin(cf_metrics)]
                
                if not ocf_fcf_data.empty:
                    fig = px.line(
                        ocf_fcf_data,
                        x="Year",
                        y="Value",
                        color="Metric",
                        markers=True,
                        title="Operating & Free Cash Flow Over Time",
                        labels={"Value": "USD", "Year": "Fiscal Year"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Operating and Free Cash Flow data not available.")
                
                # Create capital expenditure chart
                capex_data = pd.DataFrame(chart_data)
                capex_data = capex_data[capex_data['Metric'] == 'Capital Expenditure']
                
                if not capex_data.empty:
                    st.subheader("Capital Expenditure")
                    fig = px.bar(
                        capex_data,
                        x="Year",
                        y="Value",
                        title="Capital Expenditure Over Time",
                        labels={"Value": "USD", "Year": "Fiscal Year"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Capital expenditure data not available.")
            else:
                st.info("Insufficient data for cash flow analysis.")
        else:
            st.warning(f"Unknown statement type: {statement_type}")
            
    except Exception as e:
        st.error(f"Error creating statement charts: {str(e)}")
        st.info(f"Could not generate charts for {statement_type}.")
        traceback.print_exc()

def extract_xbrl_facts_for_display(ticker, statement_type):
    """Extract XBRL facts from SEC filings for display as a table"""
    if st.session_state.xbrl_data is None:
        return pd.DataFrame()
    
    df = st.session_state.xbrl_data
    
    # Filter by statement type
    if statement_type == "Balance Sheet":
        keywords = ["Assets", "Liabilities", "Equity", "Cash", "Inventory", "Receivable", "Debt"]
    elif statement_type == "Income Statement":
        keywords = ["Revenue", "Income", "Earnings", "Profit", "Loss", "Sales", "Cost", "Expense", "Tax", "EPS"]
    elif statement_type == "Cash Flow Statement":
        keywords = ["Cash", "Flow", "Operating", "Investing", "Financing", "Dividend", "Expenditure"]
    else:
        return pd.DataFrame()
    
    # Filter data by keywords
    filtered_df = df[df["metric"].apply(lambda x: any(keyword.lower() in x.lower() for keyword in keywords))]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Group by metric and pivot to create a statement-like table
    try:
        # Extract year from period for better display
        filtered_df['year'] = filtered_df['period'].apply(lambda x: x.split('-')[0])
        
        # Get sorted list of unique years
        years = sorted(filtered_df['year'].unique())
        
        # Get unique metrics
        metrics = filtered_df['metric'].unique()
        
        # Create a clean statement table
        table_data = []
        for metric in metrics:
            row_data = {'Metric': metric.split(':')[-1]}  # Remove us-gaap: prefix
            
            for year in years:
                year_data = filtered_df[(filtered_df['metric'] == metric) & (filtered_df['year'] == year)]
                if not year_data.empty:
                    # Use the first value if multiple exist for same year/metric
                    row_data[year] = year_data.iloc[0]['value']
                else:
                    row_data[year] = np.nan
                    
            table_data.append(row_data)
            
        # Convert to DataFrame
        if table_data:
            result_df = pd.DataFrame(table_data)
            # Sort years in descending order
            year_cols = [col for col in result_df.columns if col != 'Metric']
            result_df = result_df[['Metric'] + sorted(year_cols, reverse=True)]
            return result_df
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error formatting {statement_type} data: {e}")
        return pd.DataFrame()

def main():
    # st.set_page_config(page_title="Financial Chatbot", page_icon="💰", layout="wide")
    
    # Sidebar with ticker selection
    with st.sidebar:
        st.title("Financial Chatbot")
        st.subheader("Available Companies")
        
        # Display available tickers
        for ticker, info in TICKER_CIKS.items():
            if st.button(f"{ticker} - {info['name']}", key=f"btn_{ticker}"):
                set_current_ticker(ticker)
        
        st.divider()
        
        # Year range selector
        st.subheader("Analysis Range")
        start_year, end_year = st.slider(
            "Select Year Range",
            min_value=2020,
            max_value=2024,
            value=(st.session_state.start_year, st.session_state.end_year)
        )
        
        # Update session state if changed
        if start_year != st.session_state.start_year or end_year != st.session_state.end_year:
            st.session_state.start_year = start_year
            st.session_state.end_year = end_year
            
            # Reload data if a ticker is selected
            if st.session_state.current_ticker:
                # Reload filtered data without re-fetching from API
                load_financial_data(st.session_state.current_ticker)
        
        st.divider()
        
        # Display current ticker info
        st.subheader("Current Company")
        if st.session_state.current_ticker:
            ticker = st.session_state.current_ticker
            ticker_info = st.session_state.ticker_info.get(ticker, {})
            
            st.write(f"**Ticker:** {ticker}")
            st.write(f"**Company:** {ticker_info.get('company_name', TICKER_CIKS[ticker]['name'])}")
            st.write(f"**Last Filing:** {ticker_info.get('recent_filing_date', 'Unknown')}")
            st.write(f"**Last Updated:** {ticker_info.get('last_updated', 'Unknown')}")
        else:
            st.info("No company selected. Please select a company from the list above.")
    
    # Main content area with tabs
    if st.session_state.current_ticker:
        ticker = st.session_state.current_ticker
        ticker_info = st.session_state.ticker_info.get(ticker, {})
        company_name = ticker_info.get('company_name', TICKER_CIKS[ticker]['name'])
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Chat", "Company Overview", "Financial Charts"])
        
        # Tab 1: Chat Interface
        with tab1:
            st.header(f"{company_name} Financial Chatbot")
            st.write("Ask questions about the company's financial data from their SEC filings.")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    # Display message content
                    st.markdown(message["content"])
                    
                    # For assistant messages, add an expander with sources
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("View Sources"):
                            if message["sources"]["narrative_sources"]:
                                st.subheader("Narrative Sources")
                                for i, src in enumerate(message["sources"]["narrative_sources"][:3]):
                                    source_url = src['metadata'].get('source_url', 'N/A')
                                    if source_url and source_url != 'N/A':
                                        st.markdown(f"**Source {i+1}:** [Link]({source_url})")
                                    else:
                                        st.markdown(f"**Source {i+1}:**")
                                    st.code(src['content'][:300] + "...", language="")
                            
                            if message["sources"]["xbrl_sources"]:
                                st.subheader("Financial Facts")
                                for i, src in enumerate(message["sources"]["xbrl_sources"][:5]):
                                    st.markdown(f"**Fact {i+1}:** {src['content']}")
                                    
                            if "table_sources" in message["sources"] and message["sources"]["table_sources"]:
                                st.subheader("Table Sources")
                                for i, src in enumerate(message["sources"]["table_sources"][:2]):
                                    st.markdown(f"**Table {i+1}:**")
                                    st.code(src['content'][:500] + "...", language="")
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the company's SEC filings...", key="sec_input"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Researching and generating response..."):
                        # Query 10-K
                        result = query_10k(ticker, prompt)
                        
                        # Display response in formatted way
                        st.markdown(result["answer"])
                        print(result.keys())
                        # Add assistant message to chat history with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result["answer"],
                            "sources": {
                                "narrative_sources": result["narrative_sources"],
                                "xbrl_sources": result["xbrl_sources"],
                                "table_sources": result["table_sources"]
                            }
                        })
        
        # Tab 2: Company Overview
        with tab2:
            st.header(f"{company_name} Overview")
            
            # Display stock price history
            create_stock_chart(ticker)
            
            # Display financial statements
            st.subheader("Financial Statements")
            display_financial_statements()
        
        # Tab 3: Financial Charts
        with tab3:
            st.header(f"{company_name} Financial Analysis")
            
            # Create two columns - one for statement selector and charts, one for chatbot
            col1, col2 = st.columns([7, 3])
            
            with col1:
                # Statement selector
                statement_types = ["Balance Sheet", "Income Statement", "Cash Flow Statement"]
                selected_statement = st.selectbox("Select Financial Statement", statement_types, index=statement_types.index(st.session_state.selected_statement))
                
                # Update session state
                st.session_state.selected_statement = selected_statement
                
                # Display statement data
                st.subheader(f"{selected_statement} Data")
                display_financial_statements(selected_statement)
                
                # Generate charts for selected statement
                st.subheader(f"{selected_statement} Analysis")
                create_statement_charts(selected_statement)
            
            with col2:
                # Add a specialized chatbot for the selected statement
                st.subheader(f"{selected_statement} Chatbot")
                st.write(f"Ask questions about this {selected_statement.lower()}")
                
                # Display chart chat messages
                for message in st.session_state.messages_chart:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chart chat input
                if chart_prompt := st.chat_input("Ask about this statement...", key="chart_input"):
                    # Add user message to chart chat history
                    st.session_state.messages_chart.append({"role": "user", "content": chart_prompt})
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.markdown(chart_prompt)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            # Query statement data
                            result = query_statement_data(ticker, selected_statement, chart_prompt)
                            
                            # Display response
                            st.markdown(result)
                            
                            # Add assistant message to chart chat history
                            st.session_state.messages_chart.append({
                                "role": "assistant", 
                                "content": result
                            })
    else:
        # No company selected
        st.title("Financial Chatbot")
        st.write("👈 Please select a company from the sidebar to get started.")
        st.info("This application allows you to analyze SEC filings and chat with an AI about financial data.")
        
        # Check if cached financial data exists
        if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
            st.warning("""
            No cached financial data found. Please run the data fetcher script first:
            ```
            python data_fetcher.py
            ```
            This will download and cache financial data from Alpha Vantage for all companies.
            """)
        
        # Explanation of the app
        st.subheader("How it works")
        st.write("""
        1. **Select a company** from the sidebar to analyze its SEC filings and financial data
        2. **Adjust the year range** using the slider for focused financial analysis
        3. **Wait for processing** - the first time you select a company, it will process the SEC filings
        4. **Explore the tabs:**
           - **Chat:** Ask questions about SEC filings and regulatory information
           - **Company Overview:** View stock price history and financial statements
           - **Financial Charts:** Select specific statements to visualize and analyze with a dedicated chatbot
        """)
        
        st.subheader("Example questions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**SEC Filing Questions:**")
            st.write("- What was Apple's revenue in the most recent fiscal year?")
            st.write("- What are the main risk factors mentioned in the latest 10-K?") 
            st.write("- How does the company discuss competition in its filings?")
        with col2:
            st.markdown("**Financial Statement Questions:**")
            st.write("- What is the trend in gross profit margin?")
            st.write("- How has the debt-to-equity ratio changed?")
            st.write("- What's driving the change in operating income?")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())
        st.info("The application encountered an error. Please try again or select a different company.")
