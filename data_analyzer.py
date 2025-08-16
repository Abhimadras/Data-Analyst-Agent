import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
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
            logging.error(f"Analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _check_timeout(self) -> bool:
        """Check if we've exceeded the 3-minute time limit"""
        if self.start_time is None:
            return False
        return time.time() - self.start_time > self.max_duration
    
    def _parse_questions(self, questions: str) -> List[str]:
        """Parse questions text into individual tasks"""
        # Split by common delimiters
        tasks = []
        
        # Split by numbered questions (1., 2., etc.)
        numbered_pattern = r'\d+\.\s*'
        if re.search(numbered_pattern, questions):
            parts = re.split(numbered_pattern, questions)
            tasks = [part.strip() for part in parts if part.strip()]
        else:
            # Split by question marks or newlines
            parts = re.split(r'[?\n]+', questions)
            tasks = [part.strip() for part in parts if part.strip()]
        
        # If no clear separation, treat as single task
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
                    with open(filepath, 'r') as f:
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
                # If no data and no URLs, return a helpful message about what's needed
                return {
                    "error": "No data provided. Please upload data files (CSV, JSON, Parquet) or include URLs in your questions for web scraping."
                }
            
            # Determine task type and process accordingly
            if 'plot' in task_lower or 'chart' in task_lower or 'graph' in task_lower or 'scatterplot' in task_lower:
                return self._create_visualization(task, data_dict)
            elif 'correlation' in task_lower:
                return self._calculate_correlation(task, data_dict)
            elif 'regression' in task_lower:
                return self._perform_regression(task, data_dict)
            elif 'count' in task_lower or 'how many' in task_lower:
                return self._count_analysis(task, data_dict)
            elif 'earliest' in task_lower or 'latest' in task_lower or 'first' in task_lower or 'last' in task_lower:
                return self._find_extremes(task, data_dict)
            elif 'statistics' in task_lower or 'stats' in task_lower or 'summary' in task_lower:
                return self._calculate_statistics(task, data_dict)
            else:
                # General analysis - try to extract relevant information
                return self._general_analysis(task, data_dict)
                
        except Exception as e:
            logging.error(f"Task processing error: {str(e)}")
            return {"error": f"Task failed: {str(e)}"}
    
    def _scrape_wikipedia_table(self, url: str) -> pd.DataFrame:
        """Scrape tables from Wikipedia URL"""
        try:
            # Add timeout and headers for better reliability
            import requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # First try pandas read_html
            tables = pd.read_html(url, header=0)
            if tables and len(tables) > 0:
                # Return the largest table (most likely to be the main data)
                largest_table = max(tables, key=len)
                # Clean up column names
                if hasattr(largest_table, 'columns'):
                    largest_table.columns = [str(col).strip() for col in largest_table.columns]
                logging.info(f"Successfully scraped Wikipedia table with {len(largest_table)} rows and {len(largest_table.columns)} columns")
                return largest_table
            return None
        except Exception as e:
            logging.error(f"Wikipedia scraping error: {str(e)}")
            # Try alternative approach with requests + BeautifulSoup
            try:
                import requests
                from bs4 import BeautifulSoup
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                tables = soup.find_all('table', {'class': 'wikitable'})
                
                if tables:
                    # Convert first table to DataFrame
                    table = tables[0]
                    rows = []
                    headers = []
                    
                    # Extract headers
                    header_row = table.find('tr')
                    if header_row:
                        headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                    
                    # Extract data rows
                    for row in table.find_all('tr')[1:]:
                        cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                        if cells:
                            rows.append(cells)
                    
                    if rows and headers:
                        df = pd.DataFrame(rows, columns=headers)
                        logging.info(f"Successfully scraped Wikipedia table using BeautifulSoup with {len(df)} rows")
                        return df
                
            except Exception as e2:
                logging.error(f"Alternative scraping method also failed: {str(e2)}")
            
            return None
    
    def _create_visualization(self, task: str, data_dict: Dict[str, Any]) -> str:
        """Create visualization and return as base64 encoded image"""
        try:
            # Find the main dataset
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found for visualization"}
            
            plt.figure(figsize=(10, 6))
            
            # Determine plot type and create visualization
            task_lower = task.lower()
            
            if 'scatterplot' in task_lower or 'scatter' in task_lower:
                # Try to identify x and y columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    plt.scatter(df[x_col], df[y_col], alpha=0.6)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'Scatterplot: {x_col} vs {y_col}')
                    
                    # Add trend line if requested
                    if 'trend' in task_lower or 'regression' in task_lower:
                        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
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
                # Default to line plot or bar plot
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    if len(df) > 100:
                        # Line plot for time series or large datasets
                        plt.plot(df.index, df[numeric_cols[0]])
                        plt.xlabel('Index')
                        plt.ylabel(numeric_cols[0])
                        plt.title(f'Line Plot: {numeric_cols[0]}')
                    else:
                        # Bar plot for smaller datasets
                        df[numeric_cols[0]].plot(kind='bar')
                        plt.xlabel('Index')
                        plt.ylabel(numeric_cols[0])
                        plt.title(f'Bar Plot: {numeric_cols[0]}')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Check size limit (100KB = ~100,000 bytes)
            if len(image_base64) > 100000:
                # Reduce quality and try again
                buffer = io.BytesIO()
                plt.figure(figsize=(8, 5))
                # Recreate plot with lower quality
                plt.savefig(buffer, format='png', dpi=75, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logging.error(f"Visualization error: {str(e)}")
            return {"error": f"Visualization failed: {str(e)}"}
    
    def _calculate_correlation(self, task: str, data_dict: Dict[str, Any]) -> float:
        """Calculate correlation between specified columns"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}
            
            # Extract column names from task
            words = task.split()
            potential_cols = [word.strip('.,?!()') for word in words if word.isalnum()]
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            matching_cols = [col for col in numeric_cols if any(pot_col.lower() in col.lower() for pot_col in potential_cols)]
            
            if len(matching_cols) >= 2:
                corr = df[matching_cols[0]].corr(df[matching_cols[1]])
                return round(corr, 6)
            elif len(numeric_cols) >= 2:
                corr = df[numeric_cols[0]].corr(df[numeric_cols[1]])
                return round(corr, 6)
            else:
                return {"error": "Need at least 2 numeric columns for correlation"}
                
        except Exception as e:
            return {"error": f"Correlation calculation failed: {str(e)}"}
    
    def _perform_regression(self, task: str, data_dict: Dict[str, Any]) -> Dict[str, float]:
        """Perform regression analysis"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return {"error": "Need at least 2 numeric columns for regression"}
            
            x = df[numeric_cols[0]].dropna()
            y = df[numeric_cols[1]].dropna()
            
            # Ensure same length
            min_len = min(len(x), len(y))
            x = x.iloc[:min_len]
            y = y.iloc[:min_len]
            
            # Calculate regression
            coeffs = np.polyfit(x, y, 1)
            r_squared = np.corrcoef(x, y)[0, 1] ** 2
            
            return {
                "slope": round(coeffs[0], 6),
                "intercept": round(coeffs[1], 6),
                "r_squared": round(r_squared, 6)
            }
            
        except Exception as e:
            return {"error": f"Regression analysis failed: {str(e)}"}
    
    def _count_analysis(self, task: str, data_dict: Dict[str, Any]) -> int:
        """Count items based on criteria in the task"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}
            
            # Extract criteria from task
            task_lower = task.lower()
            
            # Look for monetary values
            money_pattern = r'\$([0-9.]+)\s*(bn|billion|m|million)?'
            money_matches = re.findall(money_pattern, task_lower)
            
            # Look for year criteria
            year_pattern = r'(before|after|in)\s*(\d{4})'
            year_matches = re.findall(year_pattern, task_lower)
            
            # Apply filters
            filtered_df = df.copy()
            
            if money_matches:
                for amount_str, unit in money_matches:
                    amount = float(amount_str)
                    if unit in ['bn', 'billion']:
                        amount *= 1e9
                    elif unit in ['m', 'million']:
                        amount *= 1e6
                    
                    # Find columns that might contain monetary values
                    money_cols = [col for col in df.columns if any(word in col.lower() for word in ['gross', 'revenue', 'box', 'earning', 'income'])]
                    if money_cols:
                        if 'over' in task_lower or 'above' in task_lower or '>' in task_lower:
                            filtered_df = filtered_df[filtered_df[money_cols[0]] > amount]
                        else:
                            filtered_df = filtered_df[filtered_df[money_cols[0]] >= amount]
            
            if year_matches:
                for relation, year_str in year_matches:
                    year = int(year_str)
                    year_cols = [col for col in df.columns if any(word in col.lower() for word in ['year', 'date', 'release'])]
                    if year_cols:
                        year_col = year_cols[0]
                        if relation == 'before':
                            filtered_df = filtered_df[filtered_df[year_col] < year]
                        elif relation == 'after':
                            filtered_df = filtered_df[filtered_df[year_col] > year]
                        else:  # 'in'
                            filtered_df = filtered_df[filtered_df[year_col] == year]
            
            return len(filtered_df)
            
        except Exception as e:
            return {"error": f"Count analysis failed: {str(e)}"}
    
    def _find_extremes(self, task: str, data_dict: Dict[str, Any]) -> Any:
        """Find earliest, latest, first, last items"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}
            
            task_lower = task.lower()
            
            # Look for criteria
            money_pattern = r'\$([0-9.]+)\s*(bn|billion|m|million)?'
            money_matches = re.findall(money_pattern, task_lower)
            
            # Apply filters first
            filtered_df = df.copy()
            
            if money_matches:
                for amount_str, unit in money_matches:
                    amount = float(amount_str)
                    if unit in ['bn', 'billion']:
                        amount *= 1e9
                    elif unit in ['m', 'million']:
                        amount *= 1e6
                    
                    money_cols = [col for col in df.columns if any(word in col.lower() for word in ['gross', 'revenue', 'box', 'earning', 'income'])]
                    if money_cols:
                        filtered_df = filtered_df[filtered_df[money_cols[0]] > amount]
            
            if filtered_df.empty:
                return {"error": "No records match the criteria"}
            
            # Find the extreme value
            if 'earliest' in task_lower or 'first' in task_lower:
                year_cols = [col for col in df.columns if any(word in col.lower() for word in ['year', 'date', 'release'])]
                if year_cols:
                    idx = filtered_df[year_cols[0]].idxmin()
                else:
                    idx = filtered_df.index[0]
            elif 'latest' in task_lower or 'last' in task_lower:
                year_cols = [col for col in df.columns if any(word in col.lower() for word in ['year', 'date', 'release'])]
                if year_cols:
                    idx = filtered_df[year_cols[0]].idxmax()
                else:
                    idx = filtered_df.index[-1]
            else:
                idx = filtered_df.index[0]
            
            # Return the relevant field (usually title/name)
            name_cols = [col for col in df.columns if any(word in col.lower() for word in ['title', 'name', 'film', 'movie'])]
            if name_cols:
                return filtered_df.loc[idx, name_cols[0]]
            else:
                return filtered_df.loc[idx].to_dict()
            
        except Exception as e:
            return {"error": f"Extreme value search failed: {str(e)}"}
    
    def _calculate_statistics(self, task: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic statistics"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is None:
                return {"error": "No suitable dataset found"}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return {"error": "No numeric columns found for statistics"}
            
            stats = {}
            for col in numeric_cols[:3]:  # Limit to first 3 columns
                col_stats = {
                    "count": int(df[col].count()),
                    "mean": round(df[col].mean(), 6),
                    "std": round(df[col].std(), 6),
                    "min": round(df[col].min(), 6),
                    "max": round(df[col].max(), 6),
                    "median": round(df[col].median(), 6)
                }
                stats[col] = col_stats
            
            return stats
            
        except Exception as e:
            return {"error": f"Statistics calculation failed: {str(e)}"}
    
    def _general_analysis(self, task: str, data_dict: Dict[str, Any]) -> Any:
        """Handle general analysis requests"""
        try:
            df = self._get_main_dataframe(data_dict)
            if df is not None:
                # Return basic info about the dataset
                return {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "data_types": df.dtypes.astype(str).to_dict(),
                    "sample_data": df.head().to_dict('records')
                }
            
            # If no DataFrame but we have scraped text, return text summary
            if 'scraped_text' in data_dict and data_dict['scraped_text']:
                text = data_dict['scraped_text']
                return {
                    "type": "text_content",
                    "length": len(text),
                    "preview": text[:500] + "..." if len(text) > 500 else text,
                    "word_count": len(text.split())
                }
            
            # If we have other data types, describe them
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
        """Check if we have any usable data for analysis"""
        # Check for dataframes (CSV, Parquet, scraped data)
        dataframes = [v for v in data_dict.values() if isinstance(v, pd.DataFrame)]
        if dataframes:
            return True
        
        # Check for JSON data
        json_data = [v for v in data_dict.values() if isinstance(v, (dict, list))]
        if json_data:
            return True
            
        # Check for scraped text content
        if 'scraped_text' in data_dict and data_dict['scraped_text']:
            return True
            
        return False
    
    def _get_main_dataframe(self, data_dict: Dict[str, Any]) -> pd.DataFrame:
        """Get the main dataframe from the data dictionary"""
        # Priority order: scraped_data, then largest dataframe
        if 'scraped_data' in data_dict and isinstance(data_dict['scraped_data'], pd.DataFrame):
            return data_dict['scraped_data']
        
        dataframes = [v for v in data_dict.values() if isinstance(v, pd.DataFrame)]
        if dataframes:
            # Return the largest dataframe
            return max(dataframes, key=len)
        
        # Try to convert JSON data to DataFrame if no direct DataFrames available
        for key, value in data_dict.items():
            if isinstance(value, list) and len(value) > 0:
                try:
                    # Try to create DataFrame from list of dictionaries
                    df = pd.DataFrame(value)
                    if not df.empty:
                        return df
                except:
                    pass
            elif isinstance(value, dict):
                try:
                    # Try to create DataFrame from dictionary
                    df = pd.DataFrame([value])
                    if not df.empty:
                        return df
                except:
                    pass
        
        return None
