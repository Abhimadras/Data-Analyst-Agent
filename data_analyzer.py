# data_analyzer.py  (patched to coerce money/year columns -> numeric before comparisons)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns  # kept since you import it elsewhere
import duckdb
import json
import re
import base64
import io
import logging
from typing import Dict, List, Any, Union
import requests
from bs4 import BeautifulSoup
from PIL import Image
import pyarrow.parquet as pq
import time
from urllib.parse import urlparse
from web_scraper import get_website_text_content


class DataAnalyzer:
    def __init__(self):
        self.start_time = None
        self.max_duration = 180  # 3 minutes in seconds

    # ---------------------- Public entrypoint ----------------------

    def analyze(self, questions: str, uploaded_files: Dict[str, str]) -> Union[List, Dict, str]:
        """Main analysis function"""
        self.start_time = time.time()

        try:
            # Parse questions into tasks
            tasks = self._parse_questions(questions)

            # Load data from uploaded files
            data_dict = self._load_data_files(uploaded_files)

            # Process each task
            results = []
            for task in tasks:
                if self._check_timeout():
                    results.append({"error": "Timeout exceeded (3 minutes)"})
                    break

                result = self._process_task(task, data_dict, uploaded_files)
                results.append(result)

            # Return single result if only one task, otherwise return list
            if len(results) == 1:
                return results[0]
            return results

        except Exception as e:
            logging.exception("Analysis error")
            return {"error": str(e)}

    # ---------------------- Internal helpers ----------------------

    def _check_timeout(self) -> bool:
        """Check if we've exceeded the 3-minute time limit"""
        if self.start_time is None:
            return False
        return time.time() - self.start_time > self.max_duration

    def _parse_questions(self, questions: str) -> List[str]:
        """Parse questions text into individual tasks"""
        tasks = []

        numbered_pattern = r'\d+\.\s*'
        if re.search(numbered_pattern, questions):
            parts = re.split(numbered_pattern, questions)
            tasks = [part.strip() for part in parts if part.strip()]
        else:
            parts = re.split(r'[?\n]+', questions)
            tasks = [part.strip() for part in parts if part.strip()]

        if not tasks or len(tasks) == 1:
            tasks = [questions.strip()]

        return tasks

    def _load_data_files(self, uploaded_files: Dict[str, str]) -> Dict[str, Any]:
        """Load data from uploaded files"""
        data_dict = {}

        for file_key, filepath in uploaded_files.items():
            if file_key == 'questions.txt':
                continue

            try:
                if filepath.endswith('.csv'):
                    data_dict[file_key] = pd.read_csv(filepath)
                elif filepath.endswith('.parquet'):
                    data_dict[file_key] = pd.read_parquet(filepath)
                elif filepath.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data_dict[file_key] = json.load(f)
                elif filepath.endswith(('.png', '.jpg', '.jpeg')):
                    data_dict[file_key] = Image.open(filepath)

            except Exception as e:
                logging.warning(f"Could not load file {file_key}: {str(e)}")

        return data_dict

    def _process_task(self, task: str, data_dict: Dict[str, Any], uploaded_files: Dict[str, str]) -> Any:
        """Process a single analysis task"""
        try:
            task_lower = task.lower()

            # Check for Wikipedia or URL scraping first
            urls = re.findall(r'https?://[^\s]+', task)
            if urls:
                for url in urls:
                    if 'wikipedia.org' in url:
                        scraped_data = self._scrape_wikipedia_table(url)
                        if scraped_data is not None:
                            data_dict['scraped_data'] = scraped_data
                    else:
                        scraped_text = get_website_text_content(url)
                        data_dict['scraped_text'] = scraped_text

            # Check if we have any data to work with
            has_data = self._has_usable_data(data_dict)

            if not has_data and not urls:
                return {
                    "error": "No data provided. Please upload data files (CSV, JSON, Parquet) or include URLs in your questions for web scraping."
                }

            # Determine task type and process accordingly
            if any(k in task_lower for k in ['plot', 'chart', 'graph', 'scatterplot']):
                return self._create_visualization(task, data_dict)
            elif 'correlation' in task_lower:
                return self._calculate_correlation(task, data_dict)
            elif 'regression' in task_lower:
                return self._perform_regression(task, data_dict)
            elif 'count' in task_lower or 'how many' in task_lower:
                return self._count_analysis(task, data_dict)
            elif any(k in task_lower for k in ['earliest', 'latest', 'first', 'last']):
                return self._find_extremes(task, data_dict)
            elif any(k in task_lower for k in ['statistics', 'stats', 'summary']):
                return self._calculate_statistics(task, data_dict)
            else:
                # General analysis
                return self._general_analysis(task, data_dict)

        except Exception as e:
            logging.exception("Task processing error")
            return {"error": f"Task failed: {str(e)}"}

    # ---------------------- Scraping ----------------------

    def _scrape_wikipedia_table(self, url: str) -> pd.DataFrame:
        """Scrape tables from Wikipedia URL"""
        try:
            # First try pandas read_html
            tables = pd.read_html(url, header=0)
            if tables and len(tables) > 0:
                largest_table = max(tables, key=len)
                if hasattr(largest_table, 'columns'):
                    largest_table.columns = [str(col).strip() for col in largest_table.columns]
                logging.info(f"Scraped Wikipedia table with {len(largest_table)} rows, {len(largest_table.columns)} cols")
                return largest_table
            return None
        except Exception as e:
            logging.error(f"Wikipedia scraping error: {str(e)}")
            # Fallback: requests + BS4
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                tables = soup.find_all('table', {'class': 'wikitable'})

                if tables:
                    table = tables[0]
                    rows = []
                    headers_row = []

                    header_tr = table.find('tr')
                    if header_tr:
                        headers_row = [th.get_text().strip() for th in header_tr.find_all(['th', 'td'])]

                    for row in table.find_all('tr')[1:]:
                        cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                        if cells:
                            rows.append(cells)

                    if rows and headers_row:
                        df = pd.DataFrame(rows, columns=headers_row)
                        logging.info(f"Scraped Wikipedia table via BS4: {len(df)} rows")
                        return df

            except Exception as e2:
                logging.error(f"Alternative scraping failed: {str(e2)}")

            return None

    # ---------------------- Visualization ----------------------

    def _create_visualization(self, task: str, data_dict: Dict[str, Any]) -> Any:
        """Create visualization and return as base64 encoded image"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found for visualization"}

            # Try to coerce any numeric-like text columns to numeric (best-effort)
            df = self._coerce_numeric_like_columns(df.copy())

            plt.figure(figsize=(10, 6))
            task_lower = task.lower()

            if 'scatterplot' in task_lower or 'scatter' in task_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    plt.scatter(df[x_col], df[y_col], alpha=0.6)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'Scatterplot: {x_col} vs {y_col}')

                    if 'trend' in task_lower or 'regression' in task_lower:
                        x = df[x_col].dropna()
                        y = df[y_col].dropna()
                        n = min(len(x), len(y))
                        x, y = x.iloc[:n], y.iloc[:n]
                        if n >= 2:
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            plt.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8)

            elif 'histogram' in task_lower or 'hist' in task_lower:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    col = numeric_cols[0]
                    plt.hist(df[col].dropna(), bins=30, alpha=0.7)
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.title(f'Histogram of {col}')

            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    if len(df) > 100:
                        plt.plot(df.index, df[numeric_cols[0]])
                        plt.xlabel('Index')
                        plt.ylabel(numeric_cols[0])
                        plt.title(f'Line Plot: {numeric_cols[0]}')
                    else:
                        df[numeric_cols[0]].plot(kind='bar')
                        plt.xlabel('Index')
                        plt.ylabel(numeric_cols[0])
                        plt.title(f'Bar Plot: {numeric_cols[0]}')

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            # If image is big, try a smaller dpi
            if len(image_base64) > 100000:
                buffer = io.BytesIO()
                plt.figure(figsize=(8, 5))
                plt.tight_layout()
                plt.savefig(buffer, format='png', dpi=75, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            logging.exception("Visualization error")
            return {"error": f"Visualization failed: {str(e)}"}

    # ---------------------- Analytics ----------------------

    def _calculate_correlation(self, task: str, data_dict: Dict[str, Any]) -> Any:
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}

            df = self._coerce_numeric_like_columns(df.copy())

            words = task.split()
            potential_cols = [word.strip('.,?!()') for word in words if word.isalnum()]

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            matching_cols = [col for col in numeric_cols if any(pot_col.lower() in col.lower() for pot_col in potential_cols)]

            if len(matching_cols) >= 2:
                corr = df[matching_cols[0]].corr(df[matching_cols[1]])
                return round(float(corr), 6)
            elif len(numeric_cols) >= 2:
                corr = df[numeric_cols[0]].corr(df[numeric_cols[1]])
                return round(float(corr), 6)
            else:
                return {"error": "Need at least 2 numeric columns for correlation"}

        except Exception as e:
            return {"error": f"Correlation calculation failed: {str(e)}"}

    def _perform_regression(self, task: str, data_dict: Dict[str, Any]) -> Any:
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}

            df = self._coerce_numeric_like_columns(df.copy())

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return {"error": "Need at least 2 numeric columns for regression"}

            x = df[numeric_cols[0]].dropna()
            y = df[numeric_cols[1]].dropna()
            n = min(len(x), len(y))
            x = x.iloc[:n]
            y = y.iloc[:n]

            if n < 2:
                return {"error": "Not enough data points for regression"}

            coeffs = np.polyfit(x, y, 1)
            r_squared = np.corrcoef(x, y)[0, 1] ** 2

            return {
                "slope": round(float(coeffs[0]), 6),
                "intercept": round(float(coeffs[1]), 6),
                "r_squared": round(float(r_squared), 6)
            }

        except Exception as e:
            return {"error": f"Regression analysis failed: {str(e)}"}

    def _count_analysis(self, task: str, data_dict: Dict[str, Any]) -> Any:
        """Count items based on criteria in the task"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}

            task_lower = task.lower()

            # Parse values from task
            money_pattern = r'\$?([0-9]+(?:\.[0-9]+)?)\s*(bn|billion|m|million)?'
            money_matches = re.findall(money_pattern, task_lower)

            year_pattern = r'(before|after|in)\s*(\d{4})'
            year_matches = re.findall(year_pattern, task_lower)

            filtered_df = df.copy()

            # ---- COERCE MONEY COLUMNS BEFORE COMPARISON ----
            if money_matches:
                # identify likely money columns
                money_cols = [col for col in filtered_df.columns
                              if any(word in str(col).lower()
                                     for word in ['gross', 'revenue', 'box', 'earning', 'income', 'budget'])]
                if money_cols:
                    for col in money_cols:
                        filtered_df[col] = self._to_money_float_series(filtered_df[col])

                for amount_str, unit in money_matches:
                    amount = float(amount_str)
                    if unit in ['bn', 'billion']:
                        amount *= 1e9
                    elif unit in ['m', 'million']:
                        amount *= 1e6

                    if money_cols:
                        col = money_cols[0]
                        if 'over' in task_lower or 'above' in task_lower or '>' in task_lower:
                            filtered_df = filtered_df[filtered_df[col] > amount]
                        else:
                            filtered_df = filtered_df[filtered_df[col] >= amount]

            # ---- COERCE YEAR/DATE COLUMNS BEFORE COMPARISON ----
            if year_matches:
                year_cols = [col for col in filtered_df.columns
                             if any(word in str(col).lower()
                                    for word in ['year', 'date', 'release'])]
                if year_cols:
                    ycol = year_cols[0]
                    filtered_df[ycol] = self._to_year_series(filtered_df[ycol])

                for relation, year_str in year_matches:
                    year = int(year_str)
                    if year_cols:
                        ycol = year_cols[0]
                        if relation == 'before':
                            filtered_df = filtered_df[filtered_df[ycol] < year]
                        elif relation == 'after':
                            filtered_df = filtered_df[filtered_df[ycol] > year]
                        else:  # 'in'
                            filtered_df = filtered_df[filtered_df[ycol] == year]

            return int(len(filtered_df))

        except Exception as e:
            return {"error": f"Count analysis failed: {str(e)}"}

    def _find_extremes(self, task: str, data_dict: Dict[str, Any]) -> Any:
        """Find earliest, latest, first, last items"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}

            task_lower = task.lower()
            filtered_df = df.copy()

            # Money filter first, with coercion
            money_pattern = r'\$?([0-9]+(?:\.[0-9]+)?)\s*(bn|billion|m|million)?'
            money_matches = re.findall(money_pattern, task_lower)
            if money_matches:
                money_cols = [col for col in filtered_df.columns
                              if any(word in str(col).lower()
                                     for word in ['gross', 'revenue', 'box', 'earning', 'income', 'budget'])]
                if money_cols:
                    for col in money_cols:
                        filtered_df[col] = self._to_money_float_series(filtered_df[col])

                for amount_str, unit in money_matches:
                    amount = float(amount_str)
                    if unit in ['bn', 'billion']:
                        amount *= 1e9
                    elif unit in ['m', 'million']:
                        amount *= 1e6
                    if money_cols:
                        filtered_df = filtered_df[filtered_df[money_cols[0]] > amount]

            if filtered_df.empty:
                return {"error": "No records match the criteria"}

            # Find earliest/latest using year/date after coercion
            year_cols = [col for col in filtered_df.columns
                         if any(word in str(col).lower()
                                for word in ['year', 'date', 'release'])]
            if year_cols:
                ycol = year_cols[0]
                filtered_df[ycol] = self._to_year_series(filtered_df[ycol])

            if 'earliest' in task_lower or 'first' in task_lower:
                if year_cols:
                    idx = filtered_df[ycol].idxmin()
                else:
                    idx = filtered_df.index.min()
            elif 'latest' in task_lower or 'last' in task_lower:
                if year_cols:
                    idx = filtered_df[ycol].idxmax()
                else:
                    idx = filtered_df.index.max()
            else:
                idx = filtered_df.index.min()

            # Return a title/name if available
            name_cols = [col for col in filtered_df.columns
                         if any(word in str(col).lower()
                                for word in ['title', 'name', 'film', 'movie'])]
            if name_cols:
                return filtered_df.loc[idx, name_cols[0]]
            else:
                # fallback: return the row as a dict
                row = filtered_df.loc[idx]
                if isinstance(row, pd.Series):
                    return row.to_dict()
                return row

        except Exception as e:
            return {"error": f"Extreme value search failed: {str(e)}"}

    def _calculate_statistics(self, task: str, data_dict: Dict[str, Any]) -> Any:
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}

            df = self._coerce_numeric_like_columns(df.copy())

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return {"error": "No numeric columns found for statistics"}

            stats = {}
            for col in numeric_cols[:3]:
                series = df[col].dropna()
                if series.empty:
                    continue
                col_stats = {
                    "count": int(series.count()),
                    "mean": round(float(series.mean()), 6),
                    "std": round(float(series.std()), 6),
                    "min": round(float(series.min()), 6),
                    "max": round(float(series.max()), 6),
                    "median": round(float(series.median()), 6)
                }
                stats[col] = col_stats

            if not stats:
                return {"error": "Unable to compute statistics on available columns"}

            return stats

        except Exception as e:
            return {"error": f"Statistics calculation failed: {str(e)}"}

    def _general_analysis(self, task: str, data_dict: Dict[str, Any]) -> Any:
        try:
            df = self._get_main_dataframe(data_dict)
            if df is not None:
                info = {
                    "shape": tuple(df.shape),
                    "columns": [str(c) for c in df.columns.tolist()],
                    "data_types": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
                    "sample_data": df.head().to_dict('records')
                }
                return info

            if 'scraped_text' in data_dict and data_dict['scraped_text']:
                text = data_dict['scraped_text']
                return {
                    "type": "text_content",
                    "length": len(text),
                    "preview": text[:500] + "..." if len(text) > 500 else text,
                    "word_count": len(text.split())
                }

            if data_dict:
                summary = {}
                for key, value in data_dict.items():
                    if isinstance(value, dict):
                        summary[key] = {"type": "dictionary", "keys": len(value)}
                    elif isinstance(value, list):
                        summary[key] = {"type": "list", "length": len(value)}
                    elif hasattr(value, '__class__'):
                        summary[key] = {"type": str(type(value).__name__)}
                return {"data_summary": summary}

            return {"message": "Please provide data files or URLs to analyze"}

        except Exception as e:
            return {"error": f"General analysis failed: {str(e)}"}

    def _has_usable_data(self, data_dict: Dict[str, Any]) -> bool:
        dataframes = [v for v in data_dict.values() if isinstance(v, pd.DataFrame)]
        if dataframes:
            return True

        json_data = [v for v in data_dict.values() if isinstance(v, (dict, list))]
        if json_data:
            return True

        if 'scraped_text' in data_dict and data_dict['scraped_text']:
            return True

        return False

    def _get_main_dataframe(self, data_dict: Dict[str, Any]) -> pd.DataFrame:
        if 'scraped_data' in data_dict and isinstance(data_dict['scraped_data'], pd.DataFrame):
            return data_dict['scraped_data']

        dataframes = [v for v in data_dict.values() if isinstance(v, pd.DataFrame)]
        if dataframes:
            return max(dataframes, key=len)

        # Try to convert JSON data to DataFrame
        for key, value in data_dict.items():
            if isinstance(value, list) and len(value) > 0:
                try:
                    df = pd.DataFrame(value)
                    if not df.empty:
                        return df
                except Exception:
                    pass
            elif isinstance(value, dict):
                try:
                    df = pd.DataFrame([value])
                    if not df.empty:
                        return df
                except Exception:
                    pass

        return None

    # ---------------------- Coercion utilities (the fix) ----------------------

    def _to_money_float_series(self, series: pd.Series) -> pd.Series:
        """Convert strings like '$2.1 billion', '€500m', '1,234,567', '2bn' to float dollars."""
        def parse_money(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float, np.integer, np.floating)):
                return float(x)
            s = str(x).strip().lower()

            # remove currency symbols and commas
            s = s.replace(',', '')
            s = re.sub(r'[^\d\.a-z\s\-]', '', s)  # keep digits, dot, letters (bn/million), minus

            # handle ranges like "2–3 million"
            s = s.replace('–', '-').replace('—', '-')
            if '-' in s:
                try:
                    parts = [p for p in s.split('-') if p]
                    if len(parts) >= 1:
                        s = parts[0].strip()
                except Exception:
                    pass

            # detect unit
            multiplier = 1.0
            if 'billion' in s or 'bn' in s:
                multiplier = 1e9
            elif 'million' in s or s.endswith('m'):
                multiplier = 1e6
            elif 'thousand' in s or s.endswith('k'):
                multiplier = 1e3

            # extract first number
            m = re.search(r'[-+]?\d*\.?\d+', s)
            if not m:
                return np.nan
            num = float(m.group(0))
            return num * multiplier

        try:
            return series.apply(parse_money)
        except Exception:
            # best effort
            return pd.to_numeric(series, errors='coerce')

    def _to_year_series(self, series: pd.Series) -> pd.Series:
        """Extract a 4-digit year from strings or convert existing numeric/date values to year."""
        def parse_year(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, np.integer)) and 1000 <= int(x) <= 3000:
                return int(x)
            if isinstance(x, (float, np.floating)) and 1000 <= int(x) <= 3000:
                return int(x)
            s = str(x)
            # try a date parse first
            try:
                dt = pd.to_datetime(s, errors='raise', utc=True)
                return int(dt.year)
            except Exception:
                pass
            # fall back to regex for 4-digit year
            m = re.search(r'(\d{4})', s)
            if m:
                return int(m.group(1))
            return np.nan

        return series.apply(parse_year)

    def _coerce_numeric_like_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Best-effort conversion of object columns that look numeric
        (e.g., '1,234', '45.6%', '$2.1b', '3 million') into floats.
        """
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            if df[col].dtype == object:
                sample = df[col].astype(str).head(50).str.lower()
                looks_money = sample.str.contains(r'\$|€|£|bn|billion|m|million|k|thousand', regex=True, na=False).any()
                looks_percent = sample.str.contains(r'%', regex=True, na=False).any()
                looks_numbery = sample.str.contains(r'^\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*$', regex=True, na=False).any()

                try:
                    if looks_money:
                        df[col] = self._to_money_float_series(df[col])
                    elif looks_percent:
                        # strip % and convert
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce') / 100.0
                    elif looks_numbery:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
                except Exception:
                    # ignore if conversion fails
                    pass
        return df
