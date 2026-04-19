from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import json
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
import traceback
import gc 
import psutil 
from sqlalchemy.pool import QueuePool
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import re
import ast
import html

# Pandas AI Imports for Chat framework
from pandasai import SmartDataframe
from langchain_groq import ChatGroq

def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    else:
        return obj

def apply_pro_layout(fig):
    """Utility to make Plotly charts look highly professional (BI standard)"""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, r=20, b=100, l=60),
        xaxis=dict(showgrid=False, tickangle=-45, automargin=True),
        yaxis=dict(gridcolor='#eee', zerolinecolor='#ccc', automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    # Add opacity to avoid overplotting on scatters/bars
    fig.update_traces(marker=dict(opacity=0.6, line=dict(width=0.5, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.update_traces(marker=dict(opacity=0.8), selector=dict(type='bar'))
    return fig

from ai_analyzer import EnhancedAIAnalyzer, DataScienceWorkflowGenerator

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Shared Groq Configuration for existing Chatbots
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY_HERE")
def get_groq_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=GROQ_API_KEY)

class EnhancedDataIntelligenceEngine:
    def __init__(self):
        self.data = None
        self.multi_datasets = {}
        self.analysis_results = {}
        self.chunk_size = 50000 
        self.max_memory_usage = 0.8 
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.current_model = None
        self.current_features = None
        
        # Added trackers for the Export Report feature
        self.latest_eda_report = None
        self.latest_advanced_report = None
        self.latest_ml_report = None

    def check_memory_usage(self):
        memory_percent = psutil.virtual_memory().percent
        return memory_percent / 100.0

    def optimize_dataframe(self, df):
        try:
            initial_memory = df.memory_usage(deep=True).sum()
            for col in df.select_dtypes(include=['int64']).columns:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')

            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')

            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5: 
                    df[col] = df[col].astype('category')

            final_memory = df.memory_usage(deep=True).sum()
            memory_reduction = (initial_memory - final_memory) / initial_memory * 100
            print(f"Memory optimization: {memory_reduction:.2f}% reduction")
            return df
        except Exception as e:
            print(f"Memory optimization failed: {e}")
            return df

    def load_csv_chunked(self, file_path, chunksize=None):
        try:
            if chunksize is None:
                chunksize = self.chunk_size

            file_size = os.path.getsize(file_path)
            available_memory = psutil.virtual_memory().available
            if file_size > available_memory * 0.1:
                chunksize = min(chunksize, max(1000, int(chunksize * 0.5)))
                print(f"Large file detected ({file_size/1024/1024:.1f}MB). Using chunk size: {chunksize}")

            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            chunks = []
            
            for encoding in encodings:
                try:
                    chunk_iterator = pd.read_csv(file_path, encoding=encoding, chunksize=chunksize)
                    for i, chunk in enumerate(chunk_iterator):
                        if i == 0:
                            print(f"First chunk loaded with {encoding} encoding")
                        chunk = self.optimize_dataframe(chunk)
                        chunks.append(chunk)
                        if self.check_memory_usage() > self.max_memory_usage:
                            print(f"High memory usage detected. Processing {len(chunks)} chunks so far...")
                            break
                        if i > 0 and i % 10 == 0:
                            print(f"Processed {i+1} chunks...")

                    print(f"Combining {len(chunks)} chunks...")
                    self.data = pd.concat(chunks, ignore_index=True)
                    self.data = self.optimize_dataframe(self.data)
                    chunks.clear()
                    gc.collect()
                    break

                except UnicodeDecodeError:
                    chunks.clear()
                    continue
                except Exception as e:
                    print(f"Error with {encoding}: {e}")
                    chunks.clear()
                    continue

            if self.data is None:
                raise ValueError("Could not read CSV file with any encoding")

            print(f"Successfully loaded dataset: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
            self.data = self._clean_data(self.data)
            return self.get_data_summary()
            
        except Exception as e:
            print(f"CSV loading error: {e}")
            traceback.print_exc()
            return {"error": f"Error loading CSV: {str(e)}"}

    def load_csv(self, file_path):
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:
                return self.load_csv_chunked(file_path)
            else:
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        self.data = pd.read_csv(file_path, encoding=encoding)
                        self.data = self.optimize_dataframe(self.data)
                        print(f"Successfully loaded CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error with {encoding}: {e}")
                        continue

                if self.data is None:
                    raise ValueError("Could not read CSV file with any encoding")

                self.data = self._clean_data(self.data)
                return self.get_data_summary()
                
        except Exception as e:
            print(f"CSV loading error: {e}")
            return {"error": f"Error loading CSV: {str(e)}"}

    def load_excel(self, file_path):
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024: 
                excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                if len(excel_file.sheet_names) == 1:
                    self.data = pd.read_excel(file_path, engine='openpyxl')
                else:
                    self.data = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
            else:
                excel_file = pd.ExcelFile(file_path)
                if len(excel_file.sheet_names) == 1:
                    self.data = pd.read_excel(file_path)
                else:
                    self.data = pd.read_excel(file_path, sheet_name=0)

            self.data = self.optimize_dataframe(self.data)
            self.data = self._clean_data(self.data)
            return self.get_data_summary()
            
        except Exception as e:
            print(f"Excel loading error: {e}")
            return {"error": f"Error loading Excel: {str(e)}"}

    def load_multi_dataset(self, file_path, dataset_id):
        try:
            if file_path.lower().endswith('.csv'):
                file_size = os.path.getsize(file_path)
                if file_size > 100 * 1024 * 1024:
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    chunks = []
                    for encoding in encodings:
                        try:
                            chunk_iterator = pd.read_csv(file_path, encoding=encoding, chunksize=self.chunk_size)
                            for chunk in chunk_iterator:
                                chunk = self.optimize_dataframe(chunk)
                                chunks.append(chunk)
                                if len(chunks) > 20: 
                                    break
                            data = pd.concat(chunks, ignore_index=True)
                            chunks.clear()
                            gc.collect()
                            break
                        except UnicodeDecodeError:
                            chunks.clear()
                            continue
                else:
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            data = pd.read_csv(file_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
            else:
                data = pd.read_excel(file_path)
            
            data = self.optimize_dataframe(data)
            data = self._clean_data(data)
            self.multi_datasets[dataset_id] = data
            
            # Link to self.data so single-dataset tools also work on active dataset
            self.data = data
            
            return self.get_data_summary_for_dataset(data)
            
        except Exception as e:
            print(f"Multi-dataset loading error: {e}")
            return {"error": f"Error loading dataset {dataset_id}: {str(e)}"}

    def connect_sql(self, db_type, host=None, user=None, password=None, database=None, query=None, port=None, sqlite_path=None):
        try:
            if db_type == "sqlite":
                if not sqlite_path:
                    raise ValueError("SQLite path is required for sqlite database type.")
                engine = create_engine(f"sqlite:///{sqlite_path}")

            elif db_type == "mysql":
                port = port or 3306
                host = host or 'localhost'
                user = user or 'root'
                password = password or 'Uzumymw*1'  
                
                if not database:
                    raise ValueError("Database name is required for MySQL connection")
                
                connection_strings = [
                    f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}",
                    f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
                ]
                
                db_engine = None
                for conn_str in connection_strings:
                    try:
                        print(f"Attempting MySQL connection to {host}:{port}/{database}")
                        db_engine = create_engine(
                            conn_str,
                            poolclass=QueuePool,
                            pool_size=5,
                            max_overflow=10,
                            pool_pre_ping=True,
                            pool_recycle=3600,
                            connect_args={
                                "charset": "utf8mb4",
                                "connect_timeout": 60,
                                "read_timeout": 60,
                                "write_timeout": 60
                            }
                        )
                        with db_engine.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        print("MySQL connection successful!")
                        break
                    except Exception as e:
                        print(f"Connection attempt failed: {e}")
                        db_engine = None
                        continue
                
                if db_engine is None:
                    raise Exception("All MySQL connection attempts failed.")

            elif db_type == "postgresql":
                port = port or 5432
                if not all([host, user, password, database]):
                    raise ValueError("Host, user, password, and database are required for PostgreSQL")
                db_engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            try:
                query_upper = query.upper().strip()
                if not any(keyword in query_upper for keyword in ['LIMIT', 'TOP']) and 'SELECT' in query_upper:
                    if not query_upper.startswith('SELECT COUNT'):
                        query += " LIMIT 100000" 

                with db_engine.connect() as conn:
                    try:
                        self.data = pd.read_sql(text(query), conn, chunksize=self.chunk_size)
                        if hasattr(self.data, '__iter__') and not isinstance(self.data, pd.DataFrame):
                            chunks = []
                            for i, chunk in enumerate(self.data):
                                chunks.append(self.optimize_dataframe(chunk))
                                if self.check_memory_usage() > self.max_memory_usage:
                                    break
                            self.data = pd.concat(chunks, ignore_index=True)
                            chunks.clear()
                            gc.collect()
                        else:
                            self.data = self.optimize_dataframe(self.data)
                    except Exception as chunk_error:
                        self.data = pd.read_sql(text(query), conn)
                        self.data = self.optimize_dataframe(self.data)

                if self.data.empty:
                    return {"error": "Query returned no data"}

            except Exception as e:
                return {"error": f"Query execution error: {str(e)}"}
            finally:
                if db_engine:
                    db_engine.dispose()

            self.data = self._clean_data(self.data)
            return self.get_data_summary()

        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error connecting to database: {str(e)}"}

    def test_sql_connection(self, db_type, host=None, user=None, password=None, database=None, port=None, sqlite_path=None):
        try:
            if db_type == "sqlite":
                if not sqlite_path:
                    raise ValueError("SQLite path is required")
                db_engine = create_engine(f"sqlite:///{sqlite_path}")
            elif db_type == "mysql":
                port = port or 3306
                host = host or 'localhost'
                user = user or 'root'
                password = password or 'Uzumymw*1'
                if not database:
                    raise ValueError("Database name is required for MySQL")
                connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
                db_engine = create_engine(connection_string, connect_args={"connect_timeout": 10})
            elif db_type == "postgresql":
                port = port or 5432
                if not all([host, user, password, database]):
                    raise ValueError("Host, user, password, and database are required for PostgreSQL")
                db_engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            db_engine.dispose()
            return {"success": True, "message": "Connection test successful"}
            
        except Exception as e:
            return {"error": f"Connection test failed: {str(e)}"}

    def _clean_data(self, df):
        try:
            df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
            for col in df.select_dtypes(include=['object']).columns:
                sample_size = min(1000, len(df))
                sample = df[col].dropna().head(sample_size)
                try:
                    pd.to_numeric(sample, errors='raise')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass 
            df = df.reset_index(drop=True)
            df = self.optimize_dataframe(df)
            return df
        except Exception as e:
            return df

    def safe_fillna_for_display(self, data_sample):
        try:
            display_data = data_sample.copy()
            for col in display_data.columns:
                if display_data[col].dtype.name == 'category':
                    if display_data[col].isnull().any():
                        current_categories = display_data[col].cat.categories.tolist()
                        if 'N/A' not in current_categories:
                            display_data[col] = display_data[col].cat.add_categories(['N/A'])
                        display_data[col] = display_data[col].fillna('N/A')
                elif display_data[col].dtype == 'object':
                    display_data[col] = display_data[col].fillna('N/A')
                else:
                    display_data[col] = display_data[col].fillna('N/A')
            return display_data
        except Exception as e:
            try:
                return data_sample.astype(str).fillna('N/A')
            except:
                return data_sample

    def get_data_summary_for_dataset(self, data):
        try:
            sample_data = data
            if len(data) > 100000:
                sample_data = data.sample(n=10000, random_state=42)

            summary = {
                "shape": list(data.shape),
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "missing_values": {col: int(data[col].isnull().sum()) for col in data.columns},
                "numeric_columns": list(data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(data.select_dtypes(include=['object', 'category']).columns),
                "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }

            display_sample = self.safe_fillna_for_display(sample_data.head(10))
            summary["sample_data"] = display_sample.to_dict('records')

            if summary["numeric_columns"]:
                numeric_data = sample_data[summary["numeric_columns"]] if len(data) > 100000 else data[summary["numeric_columns"]]
                summary["statistics"] = numeric_data.describe().fillna(0).to_dict()

            return summary
            
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error generating data summary: {str(e)}"}

    def get_data_summary(self):
        if self.data is None:
            return {"error": "No data loaded"}
        missing_count = int(self.data.isna().sum().sum())
        summary = self.get_data_summary_for_dataset(self.data)
        summary["total_missing"] = missing_count
        
        try:
            working_data = self.data if len(self.data) <= 10000 else self.data.sample(10000, random_state=42)
            analyzer = EnhancedAIAnalyzer(working_data)
            quality = analyzer.analyze_data_quality()
            
            summary["quality_score"] = quality.get("summary_score", 0)
            summary["duplicates"] = quality.get("duplicates", {}).get("rows", 0)
            
            missing_series = self.data.isnull().sum()
            top_missing = missing_series[missing_series > 0].sort_values(ascending=False).head(3).to_dict()
            summary["top_missing_features"] = top_missing
        except Exception as e:
            summary["quality_score"] = "N/A"
            summary["duplicates"] = "N/A"
            summary["top_missing_features"] = {}
            
        return summary

    def generate_predefined_visualizations(self):
        if self.data is None:
            return {"error": "No data loaded"}
        try:
            plotly_plots = []
            vega_specs = []
            vega_data = []
            df = self.data.copy()
            if len(df) > 5000:
                df = df.sample(5000, random_state=42)
                
            try:
                from langchain_groq import ChatGroq
                import re
                
                PREDEFINED_VIZ_API_KEY = os.environ.get("PREDEFINED_VIZ_API_KEY", "YOUR_API_KEY_HERE")
                viz_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=PREDEFINED_VIZ_API_KEY)
                
                schema_info = df.dtypes.astype(str).to_dict()
                
                prompt = f"""
                You are a strict Data Visualization Configuration Engine. 
                Below is the schema for a dataset (column names and data types):
                {json.dumps(schema_info)}
                
                Generate a JSON array of 5 valid Vega-Lite v5 JSON specifications to visualize this data.
                RULES:
                1. ONLY output a valid JSON array of objects. No markdown, no explanations, no code.
                2. Each object MUST be a valid Vega-Lite specification.
                3. Do NOT include the "data" field in the specs (we will inject it dynamically on the frontend).
                4. CRITICAL: ONLY generate standard bar charts and stacked bar charts using the "bar" mark. Do NOT use arc, line, or area charts.
                5. TITLES: The chart "title" must be an object: {{"text": "Your Title", "color": "black", "fontWeight": "bold"}}.
                6. CRITICAL: Set mark opacity to 0.6 to prevent overplotting.
                7. CRITICAL: ALWAYS map a categorical/nominal field to the 'color' encoding when making grouped/stacked charts.
                8. CRITICAL FOR LEGENDS: You MUST set the legend text and title to be bold and black. Inside your encoding (like color), add: "legend": {{"titleColor": "black", "labelColor": "black", "titleFontWeight": "bold", "labelFontWeight": "bold"}}
                9. CRITICAL REQUIREMENT: "TOP N" CHARTS ONLY. Every single chart MUST aggregate the data and display a dynamic Top N values (e.g., Top 2, Top 4, Top 5, Top 15, Top 20) for categorical fields to prevent clutter. 
                   CRITICAL: The chart's title MUST explicitly state the chosen N (e.g., "Top 5 Artists by Count", "Top 15 Songs by Revenue").
                   You MUST use Vega-Lite "transform" blocks. Example for metric aggregation (use "sum", "mean", or "count" appropriately):
                   "transform": [
                     {{"aggregate": [{{"op": "sum", "field": "TargetMetric", "as": "metric_val"}}], "groupby": ["CategoryField"]}},
                     {{"window": [{{"op": "rank", "as": "rank"}}], "sort": [{{"field": "metric_val", "order": "descending"}}]}},
                     {{"filter": "datum.rank <= 15"}} // Replace 15 with your chosen N
                   ]
                   Ensure your encodings map to the newly aggregated field (e.g., 'metric_val').
                   CRITICAL: In your X or Y encoding for the categorical field, you MUST explicitly sort it so the bars render in order: "sort": {{"field": "metric_val", "order": "descending"}}.
                10. CRITICAL FOR VALUE LABELS: Every chart MUST be a layered chart ("layer": [...]) where the first layer is the mark (bar, etc.) and the second layer is a "text" mark displaying the numerical value. Set the text mark color to "black" and fontWeight to "bold".
                """
                
                ai_response = viz_llm.invoke(prompt).content
                
                json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                
                if json_match:
                    try:
                        vega_specs = json.loads(json_match.group(0))
                    except:
                        pass
                
                safe_df = df.copy() 
                safe_df = safe_df.where(pd.notnull(safe_df), None)
                vega_data = safe_df.to_dict(orient='records')
                        
            except Exception as ai_err:
                print(f"AI Predefined Viz Config Error: {ai_err}")

            return {
                "plotly_plots": plotly_plots,
                "vega_specs": vega_specs,
                "vega_data": vega_data
            }
        except Exception as e:
            print(f"Viz Error: {e}")
            return {"error": str(e)}

    def perform_eda(self):
        if self.data is None:
            return {"error": "No data loaded"}

        try:
            working_data = self.data
            if len(self.data) > 100000:
                working_data = self.data.sample(n=50000, random_state=42)

            ai_analyzer = EnhancedAIAnalyzer(working_data)
            comprehensive_results = ai_analyzer.comprehensive_analysis()

            results = {
                "correlation_analysis": {},
                "distribution_analysis": {},
                "outlier_analysis": {},
                "insights": [],
                "ai_analysis": comprehensive_results,
                "natural_language_insights": ai_analyzer.generate_natural_language_insights(),
                "advanced_statistics": ai_analyzer.perform_advanced_statistics(),
                "sample_note": f"Analysis based on sample of {len(working_data):,} rows" if len(self.data) > 100000 else None
            }

            numeric_cols = working_data.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 1:
                try:
                    corr_cols = numeric_cols[:50] if len(numeric_cols) > 50 else numeric_cols
                    corr_matrix = working_data[corr_cols].corr()
                    results["correlation_analysis"] = {
                        "matrix": corr_matrix.fillna(0).to_dict(),
                        "high_correlations": self.find_high_correlations(corr_matrix)
                    }
                except Exception as e:
                    results["correlation_analysis"] = {"error": "Correlation analysis failed"}

            for col in numeric_cols[:20]:
                try:
                    col_data = working_data[col].dropna()
                    if len(col_data) > 0:
                        results["distribution_analysis"][col] = {
                            "mean": float(col_data.mean()),
                            "median": float(col_data.median()),
                            "std": float(col_data.std()),
                            "skewness": float(col_data.skew()),
                            "kurtosis": float(col_data.kurtosis()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max())
                        }
                except Exception as e:
                    continue

            traditional_insights = self.generate_insights()
            results["insights"] = traditional_insights

            return results
            
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error performing EDA: {str(e)}"}

    def find_high_correlations(self, corr_matrix, threshold=0.7):
        high_corr = []
        try:
            n_cols = len(corr_matrix.columns)
            max_comparisons = 1000
            comparisons_made = 0
            for i in range(n_cols):
                for j in range(i+1, n_cols):
                    if comparisons_made >= max_comparisons:
                        break
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > threshold:
                        high_corr.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": float(corr_val)
                        })
                    comparisons_made += 1
                if comparisons_made >= max_comparisons:
                    break
        except Exception as e:
            print(f"Correlation analysis error: {e}")
        return high_corr

    def generate_insights(self):
        insights = []
        if self.data is None:
            return insights

        try:
            working_data = self.data
            if len(self.data) > 100000:
                working_data = self.data.sample(n=10000, random_state=42)

            missing_pct = (working_data.isnull().sum() / len(working_data)) * 100
            high_missing = missing_pct[missing_pct > 20]
            if not high_missing.empty:
                insights.append(f"High missing values detected in: {', '.join(high_missing.index[:5])}")

            if len(self.data) > 1000000:
                insights.append(f"Large dataset detected: {len(self.data):,} rows. Consider sampling for faster analysis.")
            
            memory_usage = self.data.memory_usage(deep=True).sum() / 1024**2
            if memory_usage > 1000:
                insights.append(f"High memory usage: {memory_usage:.1f}MB. Data optimization applied.")

            numeric_cols = working_data.select_dtypes(include=[np.number]).columns[:10]
            for col in numeric_cols:
                try:
                    skewness = working_data[col].skew()
                    if abs(skewness) > 2:
                        insights.append(f"{col} shows high skewness ({skewness:.2f}) - consider transformation")
                except:
                    continue

            for col in numeric_cols[:5]:
                try:
                    Q1 = working_data[col].quantile(0.25)
                    Q3 = working_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        outliers = working_data[(working_data[col] < (Q1 - 1.5 * IQR)) | 
                                              (working_data[col] > (Q3 + 1.5 * IQR))]
                        if len(outliers) > 0:
                            outlier_pct = len(outliers) / len(working_data) * 100
                            insights.append(f"{col} has {len(outliers)} potential outliers ({outlier_pct:.1f}%)")
                except:
                    continue

            cat_cols = working_data.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols[:5]:
                try:
                    unique_count = working_data[col].nunique()
                    total_count = len(working_data[col].dropna())
                    if unique_count == total_count and total_count > 100:
                        insights.append(f"{col} appears to be a unique identifier (all values unique)")
                    elif unique_count > total_count * 0.9:
                        insights.append(f"{col} has very high cardinality ({unique_count} unique values)")
                except:
                    continue

        except Exception as e:
            print(f"Insights generation error: {e}")

        return insights[:15]

    def compare_datasets(self):
        if len(self.multi_datasets) < 2:
            return {"error": "Need at least 2 datasets for comparison"}

        try:
            comparison_results = {
                "dataset_summaries": {},
                "statistical_comparison": {},
                "column_comparison": {},
                "quality_comparison": {}
            }

            for dataset_id, data in self.multi_datasets.items():
                working_data = data
                if len(data) > 100000:
                    working_data = data.sample(n=10000, random_state=42)
                
                analyzer = EnhancedAIAnalyzer(working_data)
                quality_results = analyzer.analyze_data_quality()
                
                comparison_results["dataset_summaries"][dataset_id] = {
                    "shape": data.shape,
                    "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
                    "categorical_columns": len(data.select_dtypes(include=['object', 'category']).columns),
                    "missing_values": int(data.isnull().sum().sum()),
                    "data_quality_score": quality_results.get('summary_score', 0),
                    "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                }

            if len(self.multi_datasets) == 2:
                datasets = list(self.multi_datasets.items())
                data_a, data_b = datasets[0][1], datasets[1][1]
                
                if len(data_a) > 50000:
                    data_a = data_a.sample(n=10000, random_state=42)
                if len(data_b) > 50000:
                    data_b = data_b.sample(n=10000, random_state=42)
                
                numeric_a = set(data_a.select_dtypes(include=[np.number]).columns)
                numeric_b = set(data_b.select_dtypes(include=[np.number]).columns)
                common_numeric = numeric_a.intersection(numeric_b)
                
                if common_numeric:
                    for col in list(common_numeric)[:5]:
                        try:
                            mean_a = data_a[col].mean()
                            mean_b = data_b[col].mean()
                            
                            comparison_results["statistical_comparison"][f"{col}_mean"] = {
                                "A": float(mean_a),
                                "B": float(mean_b),
                                "difference": float(mean_a - mean_b)
                            }
                        except:
                            continue

            return comparison_results

        except Exception as e:
            return {"error": f"Error comparing datasets: {str(e)}"}

    def generate_timeseries_analysis(self):
        if self.data is None:
            return {"error": "No data loaded"}

        try:
            working_data = self.data
            if len(self.data) > 100000:
                working_data = self.data.sample(n=50000, random_state=42).sort_index()

            datetime_cols = []
            for col in working_data.columns[:20]:
                if working_data[col].dtype == 'object':
                    try:
                        sample = working_data[col].dropna().head(100)
                        pd.to_datetime(sample)
                        datetime_cols.append(col)
                    except:
                        continue

            numeric_cols = working_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found for time series analysis"}

            plots = {}

            if not datetime_cols:
                try:
                    value_col = numeric_cols[0]
                    idx_series = working_data.index.to_series().reset_index(drop=True).rename("index_time")
                    df_ts = pd.DataFrame({
                        "index_time": idx_series,
                        value_col: working_data[value_col].reset_index(drop=True)
                    }).dropna()

                    if len(df_ts) < 2:
                        return {"error": "Not enough data points for time series analysis"}

                    fig = px.line(
                        df_ts,
                        x="index_time",
                        y=value_col,
                        title=f'Time Series (Index as Time): {value_col}',
                        markers=True if len(df_ts) < 1000 else False
                    )
                    fig = apply_pro_layout(fig)
                    plots['timeseries_line_0'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                    if len(df_ts) > 10:
                        window_size = max(3, min(30, len(df_ts) // 10))
                        df_ts['moving_avg'] = df_ts[value_col].rolling(window=window_size).mean()

                        fig_trend = go.Figure()
                        df_ts_sampled = df_ts
                        if len(df_ts) > 5000:
                            step = len(df_ts) // 2000
                            df_ts_sampled = df_ts.iloc[::step]

                        fig_trend.add_trace(go.Scatter(x=df_ts_sampled["index_time"], y=df_ts_sampled[value_col], mode='lines', name='Original'))
                        fig_trend.add_trace(go.Scatter(x=df_ts_sampled["index_time"], y=df_ts_sampled['moving_avg'], mode='lines', name=f'Moving Average ({window_size})'))
                        fig_trend.update_layout(title=f'Trend Analysis (Index as Time): {value_col}', hovermode='x unified')
                        fig_trend = apply_pro_layout(fig_trend)
                        plots['trend_analysis_0'] = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)

                    return {"plots": plots}
                except Exception as e:
                    return {"error": f"Error in time series analysis: {str(e)}"}

            date_col = datetime_cols[0]
            for i, value_col in enumerate(numeric_cols[:3]):
                try:
                    df_ts = working_data[[date_col, value_col]].copy()
                    df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                    df_ts = df_ts.sort_values(date_col)
                    df_ts = df_ts.dropna()
                    
                    if len(df_ts) < 2:
                        continue
                    
                    fig = px.line(
                        df_ts, 
                        x=date_col, 
                        y=value_col,
                        title=f'Time Series: {value_col} over {date_col}',
                        markers=True if len(df_ts) < 1000 else False
                    )
                    fig = apply_pro_layout(fig)
                    plots[f'timeseries_line_{i}'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    
                    if len(df_ts) > 10:
                        window_size = max(3, min(30, len(df_ts) // 10))
                        df_ts['moving_avg'] = df_ts[value_col].rolling(window=window_size).mean()
                        
                        fig_trend = go.Figure()
                        df_ts_sampled = df_ts if len(df_ts) <= 5000 else df_ts.iloc[::len(df_ts) // 2000]
                        
                        fig_trend.add_trace(go.Scatter(x=df_ts_sampled[date_col], y=df_ts_sampled[value_col], mode='lines', name='Original', line=dict(width=1)))
                        fig_trend.add_trace(go.Scatter(x=df_ts_sampled[date_col], y=df_ts_sampled['moving_avg'], mode='lines', name=f'Moving Average ({window_size})', line=dict(width=2)))
                        
                        try:
                            from statsmodels.tsa.arima.model import ARIMA
                            series = df_ts.set_index(date_col)[value_col]
                            series = series.resample('D').mean().fillna(method='ffill')
                            if len(series) > 30 and pd.api.types.is_numeric_dtype(series):
                                model = ARIMA(series, order=(1,1,1))
                                model_fit = model.fit()
                                forecast = model_fit.forecast(steps=30)
                                forecast_dates = pd.date_range(start=series.index[-1], periods=31, freq='D')[1:]
                                fig_trend.add_trace(go.Scatter(x=forecast_dates, y=forecast.values, mode='lines', name='30-Day Forecast (ARIMA)', line=dict(dash='dash', color='red', width=2)))
                        except Exception as forecast_error:
                            print(f"Forecasting skip: {forecast_error}")

                        fig_trend.update_layout(title=f'Trend & Forecast: {value_col}', hovermode='x unified')
                        fig_trend = apply_pro_layout(fig_trend)
                        
                        plots[f'trend_analysis_{i}'] = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)

                except Exception as e:
                    continue

            if not plots:
                return {"error": "Could not generate time series visualizations"}

            return {"plots": plots}

        except Exception as e:
            return {"error": f"Error in time series analysis: {str(e)}"}

    def build_ml_models(self, target_column, model_type='auto'):
        if self.data is None:
            return {"error": "No data loaded"}

        try:
            working_data = self.data
            if len(self.data) > 200000:
                working_data = self.data.sample(n=100000, random_state=42)

            if target_column not in working_data.columns:
                return {"error": f"Target column '{target_column}' not found"}

            if model_type == 'auto':
                unique_values = working_data[target_column].nunique()
                if working_data[target_column].dtype == 'object' or unique_values < min(10, len(working_data) * 0.01):
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            else:
                problem_type = model_type

            df = working_data.copy()
            feature_cols = []
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != target_column:
                    feature_cols.append(col)

            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols[:10]:
                if col != target_column and df[col].nunique() < 50:
                    try:
                        encoded = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                        if encoded.shape[1] <= 20:
                            df = pd.concat([df, encoded], axis=1)
                            feature_cols.extend(encoded.columns.tolist())
                    except Exception as e:
                        continue

            if len(feature_cols) == 0:
                return {"error": "No suitable features found for modeling"}

            X = df[feature_cols].copy()
            y = df[target_column].copy()

            numeric_features = X.select_dtypes(include=[np.number]).columns
            X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
            
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_features:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown')

            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]

            if len(X) < 10:
                return {"error": "Not enough data points for modeling"}

            label_encoder = None
            if problem_type == 'classification' and y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            test_size = min(0.3, max(0.1, 1000 / len(X)))
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if problem_type == 'classification' else None)
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models_results = {}
            self.current_model = None
            self.current_features = feature_cols

            if problem_type == 'classification':
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
                }

                for name, model in models.items():
                    try:
                        start_time = time.time()
                        if name in ['Logistic Regression']:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        if name == 'Random Forest':
                            self.current_model = model

                        accuracy = accuracy_score(y_test, y_pred)
                        training_time = time.time() - start_time
                        
                        feature_importance = {}
                        if hasattr(model, 'feature_importances_'):
                            importance_dict = dict(zip(feature_cols, model.feature_importances_))
                            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20])
                        elif hasattr(model, 'coef_'):
                            coef_values = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                            importance_dict = dict(zip(feature_cols, np.abs(coef_values)))
                            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20])

                        models_results[name] = {'accuracy': float(accuracy), 'training_time': float(training_time), 'feature_importance': feature_importance}
                        
                        if name == 'Random Forest':
                            try:
                                import shap
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(X_test[:100])
                                if isinstance(shap_values, list):
                                    mean_shap = np.abs(shap_values[1]).mean(axis=0)
                                else:
                                    mean_shap = np.abs(shap_values).mean(axis=0)
                                shap_imp = dict(zip(feature_cols, mean_shap))
                                models_results[name]['shap_importance'] = dict(sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)[:10])
                            except Exception as shap_err:
                                models_results[name]['shap_error'] = str(shap_err)
                                
                    except Exception as e:
                        models_results[name] = {'error': str(e)}

            else: 
                models = {
                    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                    'Linear Regression': LinearRegression(),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
                }

                for name, model in models.items():
                    try:
                        start_time = time.time()
                        if name == 'Linear Regression':
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                        if name == 'Random Forest':
                            self.current_model = model

                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        training_time = time.time() - start_time

                        feature_importance = {}
                        if hasattr(model, 'feature_importances_'):
                            importance_dict = dict(zip(feature_cols, model.feature_importances_))
                            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20])
                        elif hasattr(model, 'coef_'):
                            importance_dict = dict(zip(feature_cols, np.abs(model.coef_)))
                            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20])

                        models_results[name] = {'r2_score': float(r2), 'mse': float(mse), 'training_time': float(training_time), 'feature_importance': feature_importance}
                        
                        if name == 'Random Forest':
                            try:
                                import shap
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(X_test[:100])
                                mean_shap = np.abs(shap_values).mean(axis=0)
                                shap_imp = dict(zip(feature_cols, mean_shap))
                                models_results[name]['shap_importance'] = dict(sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)[:10])
                            except Exception as shap_err:
                                models_results[name]['shap_error'] = str(shap_err)
                                
                    except Exception as e:
                        models_results[name] = {'error': str(e)}

            del X_train, X_test, y_train, y_test
            if 'X_train_scaled' in locals():
                del X_train_scaled, X_test_scaled
            gc.collect()

            return {
                'problem_type': problem_type,
                'models': models_results,
                'feature_names': feature_cols,
                'feature_count': len(feature_cols),
                'training_samples': len(X) * (1 - test_size),
                'test_samples': len(X) * test_size,
                'sample_note': f"Models trained on sample of {len(working_data):,} rows" if len(self.data) > 200000 else None
            }

        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error building ML models: {str(e)}"}

engine = EnhancedDataIntelligenceEngine()

@app.route('/')
def login_page():
    return render_template('loginsignup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        session['uploaded_filename'] = filename
        session['data_path'] = filepath

        if filename.lower().endswith('.csv'):
            result = engine.load_csv(filepath)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            result = engine.load_excel(filepath)
        else:
            result = {'error': 'Unsupported file type. Please upload CSV or Excel files.'}

        if 'error' not in result and engine.data is not None:
            memory_usage = engine.data.memory_usage(deep=True).sum() / 1024**2
            result['memory_usage_mb'] = round(memory_usage, 2)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"Upload failed: {str(e)}"}), 500

@app.route('/upload-multi', methods=['POST'])
def upload_multi_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        dataset_id = request.form.get('dataset_id', 'A')
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = f"{dataset_id}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = engine.load_multi_dataset(filepath, dataset_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-connection', methods=['POST'])
def test_connection():
    try:
        data = request.get_json()
        result = engine.test_sql_connection(
            db_type=data.get('db_type', 'mysql'),
            host=data.get('host', 'localhost'),
            user=data.get('user', 'root'),
            password=data.get('password', 'Uzumymw*1'),
            database=data.get('database'),
            port=data.get('port'),
            sqlite_path=data.get('sqlite_path')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sql-connect', methods=['POST'])
def sql_connect():
    try:
        data = request.get_json()
        db_type = data.get('db_type', 'mysql').lower()
        host = data.get('host', 'localhost')
        port = data.get('port')
        user = data.get('user', 'root')
        password = data.get('password', 'Uzumymw*1')
        database = data.get('database')
        query = data.get('query')
        sqlite_path = data.get('sqlite_path')
        
        if not query: return jsonify({'error': 'Query is required'}), 400
        if db_type == 'sqlite' and not sqlite_path: return jsonify({'error': 'SQLite path is required'}), 400
        if db_type in ['mysql', 'postgresql'] and not database: return jsonify({'error': 'Database name is required'}), 400
        
        result = engine.connect_sql(db_type=db_type, host=host, user=user, password=password, database=database, query=query, port=port, sqlite_path=sqlite_path)
        
        if 'error' not in result:
            session['sql_connection'] = {'db_type': db_type, 'host': host, 'user': user, 'database': database, 'port': port}
            session['sql_query'] = query
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clean-data', methods=['POST'])
def clean_data():
    try:
        if engine.data is None: return jsonify({"error": "No data available."}), 400
        df = engine.data.copy()
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = np.clip(df[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            
        engine.data = df
        updated_summary = engine.get_data_summary()
        
        return jsonify({
            "success": True, 
            "message": "Dataset cleaned! Missing values imputed and outliers capped via IQR.",
            "summary": updated_summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clustering', methods=['POST'])
def clustering():
    try:
        if engine.data is None: return jsonify({"error": "No data available."}), 400
        df = engine.data.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) < 2: 
            return jsonify({"error": "Clustering requires at least 2 numeric features."}), 400
            
        X = df[num_cols].fillna(df[num_cols].median())
        if len(X) > 10000:
            X = X.sample(10000, random_state=42)
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=num_cols)
        
        X_plot = pd.DataFrame(X, columns=num_cols)
        X_plot['Cluster'] = [f"Segment {c+1}" for c in clusters]
        fig = px.scatter(X_plot, x=num_cols[0], y=num_cols[1], color="Cluster", title="Unsupervised Segments (K-Means)")
        fig = apply_pro_layout(fig)

        try:
            llm = get_groq_llm()
            prompt = f"""
            We performed Unsupervised K-Means clustering (k=3) on the dataset.
            Here are the numeric centroids for the top 5 features:
            {centroids[num_cols[:5]].to_dict()}
            
            Based strictly on these averages, write a 1-2 sentence human-readable persona for each of the 3 segments. Format as bullet points.
            """
            personas = llm.invoke(prompt).content
        except Exception as groq_err:
            personas = f"Clustering succeeded, but AI Persona generation failed: {groq_err}"

        return jsonify({
            "plot": json.loads(fig.to_json()),
            "personas": personas
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export-jupyter', methods=['GET'])
def export_jupyter():
    try:
        file_name = session.get('uploaded_filename', 'your_dataset.csv')
        notebook = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["# Data Intelligence Pro\n", "### Auto-Generated Reproducible Workflow\n"]},
                {"cell_type": "code", "metadata": {}, "execution_count": 1, "outputs": [], "source": [
                    "import pandas as pd\n", 
                    "import numpy as np\n", 
                    "import plotly.express as px\n", 
                    "from sklearn.model_selection import train_test_split\n", 
                    "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n",
                    "import shap\n"
                ]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Load Data\n"]},
                {"cell_type": "code", "metadata": {}, "execution_count": 2, "outputs": [], "source": [
                    f"df = pd.read_csv('{file_name}')\n", 
                    "df.head()\n"
                ]}
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        }
        return send_file(
            io.BytesIO(json.dumps(notebook).encode()), 
            as_attachment=True, 
            download_name="data_intelligence_workflow.ipynb", 
            mimetype="application/x-ipynb+json"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
        
        results = engine.perform_eda()
        
        try:
            EXPLORATORY_GROQ_API_KEY = os.environ.get("EXPLORATORY_GROQ_API_KEY", "YOUR_API_KEY_HERE")
            report_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=EXPLORATORY_GROQ_API_KEY)
            
            stats_summary = {
                "total_rows": engine.data.shape[0],
                "total_columns": engine.data.shape[1],
                "data_quality_score": results.get('ai_analysis', {}).get('data_quality', {}).get('summary_score', 'N/A'),
                "high_correlations": results.get('correlation_analysis', {}).get('high_correlations', [])[:5],
                "top_outliers": {k: v for k, v in list(results.get('ai_analysis', {}).get('outliers', {}).items())[:5]},
                "missing_value_insights": results.get('insights', [])
            }
            
            prompt = f"""
            You are an expert Data Scientist. Review these raw statistics about an uploaded dataset:
            {json.dumps(stats_summary, indent=2)}
            
            Write a comprehensive, readable Exploratory Data Analysis report in Markdown format.
            Include the following sections:
            1. **Executive Summary**: A high-level overview of the dataset's size and quality.
            2. **Business Impact**: What do the outliers or missing data mean for a business using this data?
            3. **Data Cleaning Recommendations**: Actionable bullet points for handling the missing data or outliers.
            4. **Feature Engineering Ideas**: Suggestions for creating new data points.
            
            Do NOT write code. Just write a professional, analytical markdown report based purely on the numbers above.
            """
            ai_report = report_llm.invoke(prompt).content
            results['groq_markdown_report'] = ai_report
            
            engine.latest_eda_report = ai_report
            
        except Exception as e:
            print(f"Groq Report Generation Error: {e}")
            results['groq_markdown_report'] = "AI Report generation failed. Please verify your connection."

        return jsonify(make_json_serializable(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predefined-visualizations', methods=['POST'])
def predefined_visualizations():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
        
        filename = session.get('uploaded_filename', '').lower()
        cols = set(engine.data.columns.str.lower())
        
        # --- NEW: Anomaly Detection (SQL) ---
        sql_conn = session.get('sql_connection', {})
        sql_query = session.get('sql_query', '').lower().strip()
        db_name = sql_conn.get('database', '').lower() if sql_conn.get('database') else ''
        
        if db_name == 'anomaly_detection' and 'select * from transactions' in ' '.join(sql_query.split()):
            return jsonify({
                "is_anomaly_special": True,
                "redirect_url": "/anomaly-dashboard"
            })

        # --- Netflix Dataset Detection ---
        netflix_indicators = {'show_id', 'type', 'title', 'director', 'cast', 'country', 'release_year'}
        if 'netflix' in filename or netflix_indicators.issubset(cols):
            return jsonify({
                "is_netflix_special": True, 
                "redirect_url": "/netflix-dashboard"
            })

        # --- Retail Dataset Detection ---
        retail_indicators = {'invoice', 'stockcode', 'description', 'quantity', 'price', 'country'}
        if 'retail' in filename or retail_indicators.issubset(cols):
            return jsonify({
                "is_retail_special": True, 
                "redirect_url": "/retail-dashboard"
            })

        # --- Spotify Dataset Detection ---
        spotify_indicators = {'position', 'song', 'artist', 'popularity'}
        if 'spotify' in filename or spotify_indicators.issubset(cols):
            return jsonify({
                "is_spotify_special": True, 
                "redirect_url": "/spotify-dashboard"
            })

        # --- NCR Ride Bookings Dataset Detection ---
        ncr_indicators = {'booking id', 'vehicle type', 'pickup location', 'drop location', 'booking value'}
        if 'ncr' in filename or 'ride' in filename or ncr_indicators.issubset(cols):
            return jsonify({
                "is_ncr_special": True, 
                "redirect_url": "/ncr-dashboard"
            })

        # Standard behavior for all other datasets
        results = engine.generate_predefined_visualizations()
        return jsonify(make_json_serializable(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- NEW: Multi-Dataset Advanced Visualizations Route ---
@app.route('/multi-advanced-visualizations', methods=['POST'])
def multi_advanced_visualizations():
    try:
        if len(engine.multi_datasets) < 2:
            return jsonify({'error': 'Need at least 2 datasets uploaded for multi-dataset advanced visualizations.'}), 400
        
        data_A = engine.multi_datasets.get('A')
        data_B = engine.multi_datasets.get('B')
        
        cols_A = set(data_A.columns.str.lower()) if data_A is not None else set()
        cols_B = set(data_B.columns.str.lower()) if data_B is not None else set()
        
        # Specific IPL Batters and Bowlers signature check
        ipl_bat_cols = {'runs', 'hs', 'sr', '100s'}
        ipl_bowl_cols = {'wkt', 'ovr', 'bbi', 'eco'}
        
        is_ipl = False
        if (ipl_bat_cols.issubset(cols_A) and ipl_bowl_cols.issubset(cols_B)) or \
           (ipl_bat_cols.issubset(cols_B) and ipl_bowl_cols.issubset(cols_A)):
            is_ipl = True
        
        if is_ipl:
            return jsonify({
                "is_ipl_special": True,
                "redirect_url": "/ipl-dashboard"
            })
        
        return jsonify({'error': 'Advanced Visualizations are currently only configured for specific cross-dataset combinations (like IPL Batters & Bowlers).'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# --- END NEW ---

@app.route('/anomaly-dashboard')
def anomaly_dashboard():
    return render_template('anomaly_transactions_dashboard.html')

@app.route('/netflix-dashboard')
def netflix_dashboard():
    return render_template('netflix_highcontent_dashboard.html')

@app.route('/retail-dashboard')
def retail_dashboard():
    return render_template('online_retail_II_dashboard.html')

@app.route('/spotify-dashboard')
def spotify_dashboard():
    return render_template('spotify_global_top50_dashboard.html')

@app.route('/ncr-dashboard')
def ncr_dashboard():
    return render_template('ncr_ride_bookings_dashboard.html')

@app.route('/ipl-dashboard')
def ipl_dashboard():
    return render_template('ipl_2025_dashboard.html')

@app.route('/chat-to-data', methods=['POST'])
def chat_to_data():
    import re
    import json
    import traceback
    
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400

        user_query = request.json.get('query')
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400

        working_data = engine.data
        
        schema_info = working_data.dtypes.astype(str).to_dict()
        sample_data = working_data.head(3).to_dict(orient='records') 
        
        llm = get_groq_llm()
        
        prompt = f"""
        You are a strict JSON Data Visualization API. 
        Do NOT include any markdown formatting, markdown code blocks (like ```json), or conversational text in your response. 
        Output ONLY a raw, minified JSON object.

        DATASET SCHEMA: 
        {json.dumps(schema_info)}
        
        SAMPLE DATA (3 rows):
        {json.dumps(sample_data, default=str)}

        USER QUERY: "{user_query}"

        INSTRUCTIONS:
        1. If the user asks for a heatmap or correlation matrix, output EXACTLY this JSON:
           {{"action": "heatmap", "chart_name": "Correlation Heatmap"}}
           
        2. For any other chart (Bar, Pie, Line, Scatter, etc.), output a Vega-Lite v5 specification inside this JSON structure:
           {{"action": "vega", "chart_name": "Descriptive Chart Title", "spec": {{ ... your vega lite spec ... }} }}
           - Do NOT include the "data" array inside the spec (it is injected by the frontend).
           - CRITICAL: ALWAYS specify the exact "type" for fields ('nominal' for categories/text, 'quantitative' for numbers).
           - CRITICAL TO PREVENT SINGLE COLOR CHARTS: ALWAYS map a categorical/nominal field to the 'color' encoding when grouping or stacking data (e.g., Pie charts, Stacked Bar charts).
           - CRITICAL FOR PIE CHARTS: Use the 'arc' mark, map the metric to the 'theta' channel, and map the category to the 'color' channel.
           - CRITICAL FOR LEGENDS: You MUST set the legend text and title to be bold and black. Inside your encoding (like color), add: "legend": {{"titleColor": "black", "labelColor": "black", "titleFontWeight": "bold", "labelFontWeight": "bold"}}
           - CRITICAL FOR "TOP N" QUERIES (e.g. "top 10 countries"): Do NOT just plot everything. You MUST use a "transform" block to calculate the top items. 
             Example of Top 10 by Count:
             "transform": [
               {{"aggregate": [{{"op": "count", "as": "agg_count"}}], "groupby": ["YourCategoryField"]}},
               {{"window": [{{"op": "rank", "as": "rank"}}], "sort": [{{"field": "agg_count", "order": "descending"}}]}},
               {{"filter": "datum.rank <= 10"}}
             ]
             (If summing, replace "count" with "sum" and specify "field"). Make sure to map the new aggregated field (e.g. 'agg_count') in your 'x' or 'y' encoding!
           - Always explicitly define rich tooltips.
           - Ensure axes have descriptive, human-readable titles.
        """

        try:
            raw_response = llm.invoke(prompt).content
        except Exception as e:
            return jsonify({"type": "text", "answer": "I had trouble connecting to the AI brain. Please try again."})

        response_data = {}
        raw_str = str(raw_response).strip()
        
        if raw_str.startswith("```"):
            raw_str = re.sub(r'^```(json)?|```$', '', raw_str, flags=re.MULTILINE).strip()
            
        try:
            response_data = json.loads(raw_str)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', raw_str, re.DOTALL)
            if json_match:
                try:
                    response_data = json.loads(json_match.group(0))
                except:
                    pass

        if not response_data:
            print(f"Failed to parse LLM output. Raw string was:\n{raw_str}")
            return jsonify({"type": "text", "answer": "I could not generate a valid chart configuration. The query might be too complex or unclear."})

        action = response_data.get('action')

        if action == 'heatmap':
            import plotly.graph_objects as go
            
            num_cols = working_data.select_dtypes(include=['number'])
            if num_cols.empty:
                return jsonify({"type": "text", "answer": "There are no numeric columns to create a correlation heatmap."})
                
            matrix = num_cols.corr()
            fig = go.Figure(data=go.Heatmap(
                z=matrix.values.tolist(),
                x=list(matrix.columns),
                y=list(matrix.index),
                colorscale='RdBu_r'
            ))
            fig.update_layout(title="Correlation Heatmap")
            fig = apply_pro_layout(fig) 
            
            return jsonify({
                "type": "plot",
                "engine": "plotly",
                "answer": f"### {response_data.get('chart_name', 'Correlation Heatmap')}",
                "plot_json": json.loads(fig.to_json())
            })

        elif action == 'vega':
            spec = response_data.get('spec', {})
            if not spec:
                return jsonify({"type": "text", "answer": "I generated the framework but failed to build the chart specification."})
                
            chart_name = response_data.get('chart_name', spec.get('title', 'Data Visualization'))
            
            if 'title' in spec:
                del spec['title']
                
            safe_df = working_data.head(2000).where(pd.notnull(working_data), None)
            
            return jsonify({
                "type": "plot",
                "engine": "vega",
                "answer": f"### {chart_name}",
                "spec": spec,
                "vega_data": safe_df.to_dict(orient='records')
            })
            
        else:
            return jsonify({"type": "text", "answer": "I couldn't determine what type of chart you wanted."})

    except Exception as e:
        print(f"Chat-to-Data error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/data-intel-chat', methods=['POST'])
def data_intel_chat():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400

        user_query = request.json.get('query')
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400

        summary = engine.get_data_summary()
        columns = summary.get('columns', [])
        numeric = summary.get('numeric_columns', [])
        categorical = summary.get('categorical_columns', [])
        shape = summary.get('shape', [0,0])

        prompt = f"""
        You are the "Data Intelligence Chat Bot", an elite AI Data Scientist.
        The user has uploaded a dataset. Here is the context of their data:
        - Rows: {shape[0]:,}
        - Columns: {shape[1]}
        - Numeric Columns: {numeric}
        - Categorical Columns: {categorical}
        
        The user asks: "{user_query}"
        
        Provide a highly intelligent, detailed text response. You can write statistical summaries, suggest Python code snippets, or analyze trends based on the column names. 
        Use beautiful Markdown formatting (bullet points, bold text, code blocks) to make your answer easy to read. Do NOT attempt to output raw JSON or Plotly objects.
        """

        llm = get_groq_llm()
        response = llm.invoke(prompt)
        
        formatted_answer = response.content.replace('\n', '<br>')

        return jsonify({
            "type": "text", 
            "answer": formatted_answer
        })

    except Exception as e:
        print(f"Data Intel Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/timeseries-analysis', methods=['POST'])
def timeseries_analysis():
    try:
        if engine.data is None: return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
        results = engine.generate_timeseries_analysis()
        return jsonify(make_json_serializable(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/advanced-analysis', methods=['POST'])
def advanced_analysis():
    try:
        if engine.data is None: return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
        working_data = engine.data if len(engine.data) <= 100000 else engine.data.sample(n=50000, random_state=42)
        analyzer = EnhancedAIAnalyzer(working_data)
        
        results = analyzer.comprehensive_analysis()
        insights = analyzer.generate_natural_language_insights()
        advanced_stats = analyzer.perform_advanced_statistics()
        
        response_data = {
            'analysis_results': results, 
            'natural_language_insights': insights,
            'advanced_stats': advanced_stats
        }
        
        try:
            ADVANCED_GROQ_API_KEY = os.environ.get("ADVANCED_GROQ_API_KEY", "YOUR_API_KEY_HERE")
            advanced_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=ADVANCED_GROQ_API_KEY)
            
            stats_summary = {
                "total_rows": engine.data.shape[0],
                "total_columns": engine.data.shape[1],
                "columns": list(engine.data.columns),
                "data_types": {col: str(dtype) for col, dtype in engine.data.dtypes.items()},
                "top_outliers": {k: v for k, v in list(results.get('outliers', {}).items())[:5]},
                "advanced_statistics": {k: v for k, v in list(advanced_stats.get('advanced_statistics', {}).items())[:5]}
            }
            
            prompt = f"""
            You are an Elite Data Scientist and Machine Learning Architect.
            Review the following statistical snapshot of an uploaded dataset:
            {json.dumps(stats_summary, indent=2)}
            
            Generate a deep-dive, advanced analytical report in Markdown format. 
            Your report MUST include the following sections exactly:
            
            ## 📖 Data Storytelling Narrative
            (A concise, 2-3 paragraph "Executive Story" synthesizing the dataset's overall health and most surprising findings.)
            
            ## ⚙️ Automated Feature Engineering Suggestions
            (Suggest 3-5 new features that could be created from the existing columns, explaining why they add value.)
            
            ## 🔍 Hypothesis Generation & Root Cause Analysis
            (Generate 2-3 real-world hypotheses explaining potential correlations, skews, or data patterns based on the column names.)
            
            ## 🤖 Predictive Viability Assessment
            (Provide a "Machine Learning Readiness Score" out of 100. Highlight potential issues like cardinality, scaling needs, or missing target variables before model training.)
            
            ## 🚨 Smart Anomaly Interpretation
            (Analyze the nature of the outliers provided. Are they likely data entry errors or natural extreme events? What should be done with them?)
            
            ## 📊 Categorical Deep-Dives (Segmentation)
            (Suggest 2-3 natural segments or clusters within this data that the business should investigate further.)
            
            Do NOT write code. ONLY output the Markdown report. Make it highly professional, visually structured with bullet points, and actionable.
            """
            ai_report = advanced_llm.invoke(prompt).content
            response_data['groq_advanced_report'] = ai_report
            
            engine.latest_advanced_report = ai_report
            
        except Exception as e:
            print(f"Groq Advanced Report Generation Error: {e}")
            response_data['groq_advanced_report'] = f"AI Advanced Analysis failed to generate. Error: {str(e)}"

        return jsonify(make_json_serializable(response_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare-datasets', methods=['POST'])
def compare_datasets():
    try:
        if len(engine.multi_datasets) < 2: return jsonify({'error': 'Need at least 2 datasets for comparison'}), 400
        results = engine.compare_datasets()
        return jsonify(make_json_serializable(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate-workflow', methods=['POST'])
def generate_workflow():
    try:
        if engine.data is None: return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
        working_data = engine.data if len(engine.data) <= 100000 else engine.data.sample(n=10000, random_state=42)
        workflow = DataScienceWorkflowGenerator(working_data).generate_complete_workflow()
        return jsonify({'workflow': workflow})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if engine.data is None: return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
        data = request.get_json()
        target_column = data.get('target_column')
        model_type = data.get('model_type', 'auto')
        if not target_column or target_column not in engine.data.columns: return jsonify({'error': 'Invalid target column.'}), 400
        
        results = engine.build_ml_models(target_column, model_type)
        
        if 'error' not in results:
            try:
                ML_INTERPRETATION_GROQ_API_KEY = os.environ.get("ML_INTERPRETATION_GROQ_API_KEY", "YOUR_API_KEY_HERE")
                ml_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=ML_INTERPRETATION_GROQ_API_KEY)
                
                prompt = f"""
                You are a Senior Machine Learning Engineer.
                Review the following raw machine learning model metrics and feature importances:
                {json.dumps(make_json_serializable(results), indent=2)}
                
                Write a static, comprehensive Model Evaluation Report in Markdown format.
                Include exactly these sections:
                1. Which model performed best and why.
                2. What the top 3 driving features mean in plain English.
                3. Warnings about potential overfitting (e.g., if accuracy/R2 is 100% or 1.0).
                4. Recommendations for next steps (e.g., "Try dropping feature X as it dominates the model").
                
                Do NOT write code. ONLY output the Markdown report. Make it highly professional, visually structured with bullet points, and actionable.
                """
                ai_report = ml_llm.invoke(prompt).content
                results['groq_ml_report'] = ai_report
                
                engine.latest_ml_report = ai_report
                
            except Exception as e:
                print(f"Groq ML Report Generation Error: {e}")
                results['groq_ml_report'] = "AI ML Evaluation Report failed to generate. Please check API key/connection."

        return jsonify(make_json_serializable(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulate-what-if', methods=['POST'])
def simulate_what_if():
    try:
        data = request.json.get('features', {})
        if not engine.current_model or not engine.current_features:
            return jsonify({"error": "You must build and train a Machine Learning model first."}), 400
            
        df_input = pd.DataFrame(columns=engine.current_features)
        df_input.loc[0] = 0.0 
        for k, v in data.items():
            if k in df_input.columns:
                df_input.at[0, k] = float(v)
                
        pred = engine.current_model.predict(df_input)
        return jsonify({"prediction": float(pred[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export-enhanced-report', methods=['POST'])
def export_enhanced_report():
    try:
        if engine.data is None:
            return jsonify({'error': 'No data uploaded yet. Please upload a file first.'}), 400
        
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('EnhancedTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=20, alignment=1, textColor=colors.HexColor('#2563eb'))
        
        story = []
        
        raw_filename = session.get('uploaded_filename', 'Data_Intelligence_Dataset')
        dataset_name = os.path.splitext(raw_filename)[0].replace('_', ' ').title()
        report_title = f"{dataset_name} Report"
        
        story.append(Paragraph(report_title, title_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph("Data Overview", styles['Heading2']))
        data_summary = engine.get_data_summary()
        memory_usage = engine.data.memory_usage(deep=True).sum() / 1024**2
        
        executive_summary = f"""
        This report presents a comprehensive analysis of the uploaded dataset containing {data_summary['shape'][0]:,} records 
        and {data_summary['shape'][1]} features. The dataset occupies approximately {memory_usage:.1f}MB in memory.
        The analysis includes {len(data_summary['numeric_columns'])} numeric variables 
        and {len(data_summary['categorical_columns'])} categorical variables.
        """
        story.append(Paragraph(executive_summary, styles['Normal']))
        story.append(Spacer(1, 20))
        
        def append_md(md_text):
            if not md_text: return
            lines = md_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 8))
                    continue
                
                line = html.escape(line)
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                
                if line.startswith('### '):
                    story.append(Paragraph(line[4:], styles['Heading3']))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['Heading2']))
                elif line.startswith('# '):
                    story.append(Paragraph(line[2:], styles['Heading1']))
                elif line.startswith('- ') or line.startswith('* '):
                    story.append(Paragraph(line[2:], styles['Normal'], bulletText='•'))
                else:
                    story.append(Paragraph(line, styles['Normal']))
        
        if engine.latest_eda_report:
            story.append(PageBreak())
            story.append(Paragraph("Exploratory Data Analysis", styles['Heading1']))
            append_md(engine.latest_eda_report)
            
        if engine.latest_advanced_report:
            story.append(PageBreak())
            story.append(Paragraph("Advanced Deep-Dive Diagnostics", styles['Heading1']))
            append_md(engine.latest_advanced_report)
            
        if engine.latest_ml_report:
            story.append(PageBreak())
            story.append(Paragraph("Machine Learning Evaluation", styles['Heading1']))
            append_md(engine.latest_ml_report)
        
        doc.build(story)
        pdf_buffer.seek(0)
        
        safe_download_name = f'{dataset_name.replace(" ", "_")}_Report_{datetime.now().strftime("%Y%m%d")}.pdf'
        
        return send_file(
            pdf_buffer, as_attachment=True,
            download_name=safe_download_name,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 1GB.'}), 413

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.teardown_appcontext
def cleanup(error):
    gc.collect()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000, threaded=True)