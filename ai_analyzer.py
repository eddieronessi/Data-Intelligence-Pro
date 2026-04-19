import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
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


class EnhancedAIAnalyzer:
    """
    AI analyzer that works for ANY dataset:
    - Provides comprehensive analysis (summary stats, correlations, quality).
    - Generates natural-language insights.
    - Provides advanced statistics summary.
    - Generates dynamic AI-configured visualizations via Vega-Lite.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def generate_predefined_visualizations(self):
        """Generates a dictionary containing AI Vega-Lite specs and the data sample."""
        plots = {}
        configs = []
        vega_data = []
        try:
            df = self.data.copy()
            # Downsample for faster rendering if data is huge
            if len(df) > 5000:
                df = df.sample(5000, random_state=42)

            # 1. Use Groq AI for One-Shot Dynamic Visualization Configuration (Vega-Lite Method)
            try:
                from langchain_groq import ChatGroq
                import re
                
                # Using the exact API key provided for predefined visualizations
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
                
                # Safely extract JSON from the response
                json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
                
                if json_match:
                    try:
                        configs = json.loads(json_match.group(0))
                    except:
                        pass
                
                # Format Data for Frontend Injection
                safe_df = df.copy() 
                safe_df = safe_df.where(pd.notnull(safe_df), None)
                vega_data = safe_df.to_dict(orient='records')
                        
            except Exception as ai_err:
                print(f"AI Predefined Viz Config Error in Analyzer: {ai_err}")

        except Exception as e:
            print(f"Visualization generation error: {e}")

        return {
            "plots": plots,
            "vega_specs": configs,
            "vega_data": vega_data
        }

    def comprehensive_analysis(self):
        """Return a rich, JSON-serializable analysis dictionary."""
        try:
            df = self.data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Basic info
            basic_info = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "numeric_column_count": len(numeric_cols),
                "categorical_column_count": len(categorical_cols),
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }

            # Data quality
            quality = self.analyze_data_quality()

            # Summary statistics (sampled for huge data)
            sample_df = df
            if len(df) > 100_000:
                sample_df = df.sample(10_000, random_state=42)

            numeric_summary = {}
            if len(numeric_cols) > 0:
                desc = sample_df[numeric_cols].describe().to_dict()
                numeric_summary = {k: {m: float(v) for m, v in stats_dict.items()} for k, stats_dict in desc.items()}

            categorical_summary = {}
            for col in categorical_cols[:20]:
                vc = sample_df[col].value_counts(dropna=False).head(20)
                categorical_summary[col] = {str(idx): int(val) for idx, val in vc.items()}

            # Correlations
            correlations = {}
            if len(numeric_cols) > 1:
                corr = sample_df[numeric_cols].corr().fillna(0)
                correlations = corr.to_dict()

            # Outlier overview (IQR-based on first few numeric columns)
            outlier_summary = {}
            for col in numeric_cols[:10]:
                s = sample_df[col].dropna()
                if s.empty:
                    continue
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = s[(s < lower) | (s > upper)]
                outlier_summary[col] = {
                    "outlier_count": int(len(outliers)),
                    "outlier_percentage": float(len(outliers) / len(s) * 100.0),
                }

            return make_json_serializable({
                "basic_info": basic_info,
                "data_quality": quality,
                "numeric_summary": numeric_summary,
                "categorical_summary": categorical_summary,
                "correlations": correlations,
                "outliers": outlier_summary,
            })
        except Exception as e:
            print(f"Comprehensive_analysis error: {e}")
            return {"error": str(e)}

    def analyze_data_quality(self):
        """Return a richer quality dict instead of a static placeholder."""
        try:
            df = self.data
            total_cells = df.shape[0] * df.shape[1] if df.shape[0] and df.shape[1] else 1
            total_missing = int(df.isna().sum().sum())
            completeness = 1.0 - total_missing / total_cells

            # Duplicates
            dup_rows = int(df.duplicated().sum())
            dup_pct = float(dup_rows / max(len(df), 1) * 100.0)

            # Uniqueness per column
            uniq = {col: int(df[col].nunique()) for col in df.columns}

            # Simple consistency proxy: numeric columns with non-finite values
            consistency_issues = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                s = df[col]
                non_finite = (~np.isfinite(s.astype(float))).sum()
                if non_finite > 0:
                    consistency_issues[col] = int(non_finite)

            summary_score = float(
                max(0.0, min(100.0,
                    50.0 * completeness +
                    30.0 * (1.0 - dup_pct / 100.0) +
                    20.0 * (1.0 - (len(consistency_issues) / max(len(df.columns), 1)))
                ))
            )

            return make_json_serializable({
                "summary_score": summary_score,
                "completeness": {
                    "overall_score": float(completeness * 100.0),
                    "total_missing_cells": total_missing,
                },
                "duplicates": {
                    "rows": dup_rows,
                    "percentage": dup_pct,
                },
                "uniqueness": uniq,
                "consistency": {
                    "issue_columns": list(consistency_issues.keys()),
                    "issues": consistency_issues,
                },
            })
        except Exception as e:
            print(f"Analyze_data_quality error: {e}")
            return {"summary_score": 60.0, "error": str(e)}

    def perform_advanced_statistics(self):
        """Basic advanced stats for numeric columns."""
        try:
            df = self.data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            advanced = {}

            for col in numeric_cols[:15]:
                s = df[col].dropna()
                if len(s) < 5:
                    continue
                advanced[col] = {
                    "skewness": float(s.skew()),
                    "kurtosis": float(s.kurtosis()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "median": float(s.median()),
                }

            return make_json_serializable({"advanced_statistics": advanced})
        except Exception as e:
            print(f"Perform_advanced_statistics error: {e}")
            return {"error": str(e)}

    def generate_natural_language_insights(self):
        """Generate text insights that work for any dataset."""
        insights = []
        try:
            df = self.data
            n_rows, n_cols = df.shape

            insights.append(
                f"The dataset contains {n_rows:,} rows and {n_cols} columns."
            )

            # Missing values
            missing_pct = (df.isna().sum() / max(len(df), 1) * 100.0).sort_values(ascending=False)
            high_missing = missing_pct[missing_pct > 20]
            if not high_missing.empty:
                top_missing = ", ".join(
                    f"{col} ({pct:.1f}%)" for col, pct in high_missing.head(5).items()
                )
                insights.append(
                    f"Some columns have a high proportion of missing values, such as: {top_missing}."
                )
            else:
                insights.append("Overall, missing values appear to be relatively low across columns.")

            # Numeric distribution / skew
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:5]:
                s = df[col].dropna()
                if s.empty:
                    continue
                skew = s.skew()
                if abs(skew) > 2:
                    direction = "right" if skew > 0 else "left"
                    insights.append(
                        f"The numeric feature '{col}' is highly {direction}-skewed (skewness = {skew:.2f}); "
                        f"a transformation (e.g., log) may help."
                    )

            # Correlation insight
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr().abs()
                np.fill_diagonal(corr.values, 0)
                stacked = corr.stack().sort_values(ascending=False)
                if len(stacked) > 0:
                    top_pair, top_val = stacked.index[0], stacked.iloc[0]
                    insights.append(
                        f"The strongest linear relationship is between '{top_pair[0]}' and '{top_pair[1]}' "
                        f"with an absolute correlation of {top_val:.2f}."
                    )

            # Categorical high-cardinality
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols[:5]:
                uniq = df[col].nunique(dropna=True)
                if uniq > 0.5 * len(df):
                    insights.append(
                        f"Categorical feature '{col}' has very high cardinality ({uniq} unique values), "
                        "which may require encoding strategies or dimensionality reduction."
                    )

        except Exception as e:
            print(f"Generate_natural_language_insights error: {e}")
            insights.append(f"Automatic insights partially failed: {e}")

        return insights


class DataScienceWorkflowGenerator:
    """
    Workflow generator powered by Groq AI.
    It takes the dataset summary and dynamically outputs a tailored, highly specific workflow
    complete with generated python code, difficulty levels, and risk warnings.
    """
    def __init__(self, data):
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def generate_complete_workflow(self):
        try:
            from langchain_groq import ChatGroq
            import re
            
            df = self.data
            schema_info = df.dtypes.astype(str).to_dict()
            missing_info = df.isnull().sum().to_dict()
            
            # Filter out columns with 0 missing values to keep prompt concise
            missing_info = {k: v for k, v in missing_info.items() if v > 0}
            rows, cols = df.shape

            summary = {
                "rows": rows,
                "columns": cols,
                "schema": schema_info,
                "missing_values_detected": missing_info
            }

            # Utilizing the requested isolated API key for the workflow generator
            WORKFLOW_GROQ_API_KEY = os.environ.get("WORKFLOW_GROQ_API_KEY", "YOUR_API_KEY_HERE")
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=WORKFLOW_GROQ_API_KEY)

            prompt = f"""
            You are an Expert Lead Data Scientist.
            Based on the following dataset summary, generate a highly tailored, step-by-step Data Science Workflow.
            
            Dataset Summary:
            {json.dumps(summary, indent=2)}

            Return ONLY a valid JSON array containing the workflow steps. Do not include markdown ticks (```json) outside the array. Do not include any explanations.
            Each object in the array MUST contain the exact following keys:
            - "phase": string (e.g., "Data Preparation", "Exploratory Analysis", "Modeling")
            - "step_name": string (e.g., "Handle Imbalanced Target Classes")
            - "description": string (Clear instruction on what to do and why)
            - "python_code": string (Exact, context-aware pandas/sklearn python starter code using the actual column names from the summary. Use \n for new lines)
            - "complexity": string (Must be exactly "Beginner", "Intermediate", or "Advanced")
            - "estimated_time": string (e.g., "15-30 mins")
            - "risk_warnings": string (Potential gotchas, data leakage risks, or caveats specific to this step and this data)
            - "dependencies": list of strings (e.g., ["pandas", "scikit-learn"])
            
            Generate between 5 to 7 high-impact steps.
            """

            response = llm.invoke(prompt).content
            
            # Safely extract JSON from the AI response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                workflow_data = json.loads(json_match.group(0))
                return workflow_data
            else:
                return [{"error": "Failed to parse AI workflow data. Please try again."}]
                
        except Exception as e:
            print(f"AI Workflow Generation Error: {e}")
            return [{"error": f"Workflow generation failed: {str(e)}"}]