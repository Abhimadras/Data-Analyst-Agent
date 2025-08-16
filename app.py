import os
import logging
import tempfile
import json
import pandas as pd
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from data_analyzer import DataAnalyzer

# Load environment variables from .env
load_dotenv()  # loads SESSION_SECRET

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'txt', 'csv', 'json', 'png', 'jpg', 'jpeg', 'parquet'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and coerce numeric-like columns to floats/ints
    so comparisons like > or < don't break.
    """
    for col in df.columns:
        if df[col].dtype == object:
            # Try converting to numbers where possible
            df[col] = (
                df[col]
                .replace('[\$,bnm]', '', regex=True)  # remove $, b, n, m
                .str.replace(',', '', regex=False)     # remove commas
            )
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


@app.route('/')
def index():
    """Render the main interface for testing the API"""
    return render_template('index.html')


@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint for data analysis"""
    try:
        if 'questions.txt' not in request.files:
            return jsonify({"error": "questions.txt is required"}), 400

        questions_file = request.files['questions.txt']
        if questions_file.filename == '':
            return jsonify({"error": "questions.txt cannot be empty"}), 400

        # Read questions
        questions_content = questions_file.read().decode('utf-8')
        if not questions_content.strip():
            return jsonify({"error": "questions.txt cannot be empty"}), 400

        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_files = {}

            # Save questions.txt
            questions_path = os.path.join(temp_dir, 'questions.txt')
            with open(questions_path, 'w', encoding='utf-8') as f:
                f.write(questions_content)
            uploaded_files['questions.txt'] = questions_path

            # Process other uploaded files
            for file_key in request.files:
                if file_key != 'questions.txt':
                    file = request.files[file_key]
                    if file and file.filename and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(temp_dir, filename)
                        file.save(filepath)

                        # If it's a CSV/JSON, coerce numerics right here
                        if filename.endswith(".csv"):
                            df = pd.read_csv(filepath)
                            df = coerce_numeric_columns(df)
                            df.to_csv(filepath, index=False)
                        elif filename.endswith(".json"):
                            df = pd.read_json(filepath)
                            df = coerce_numeric_columns(df)
                            df.to_json(filepath, orient="records")

                        uploaded_files[file_key] = filepath

            # Initialize data analyzer
            analyzer = DataAnalyzer()

            # Process the analysis request
            result = analyzer.analyze(questions_content, uploaded_files)

            return jsonify(result)

    except Exception as e:
        logging.error(f"Error in analyze_data: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route('/test', methods=['POST'])
def test_upload():
    """Test endpoint for the web interface"""
    try:
        if 'questions' not in request.files:
            flash('questions.txt is required', 'error')
            return redirect(url_for('index'))

        questions_file = request.files['questions']
        if questions_file.filename == '':
            flash('Please select a questions.txt file', 'error')
            return redirect(url_for('index'))

        # Read questions
        questions_content = questions_file.read().decode('utf-8')

        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_files = {}

            # Save questions.txt
            questions_path = os.path.join(temp_dir, 'questions.txt')
            with open(questions_path, 'w', encoding='utf-8') as f:
                f.write(questions_content)
            uploaded_files['questions.txt'] = questions_path

            # Process other uploaded files
            for file_key in request.files:
                if file_key != 'questions':
                    file = request.files[file_key]
                    if file and file.filename and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(temp_dir, filename)
                        file.save(filepath)

                        # Coerce numeric if CSV/JSON
                        if filename.endswith(".csv"):
                            df = pd.read_csv(filepath)
                            df = coerce_numeric_columns(df)
                            df.to_csv(filepath, index=False)
                        elif filename.endswith(".json"):
                            df = pd.read_json(filepath)
                            df = coerce_numeric_columns(df)
                            df.to_json(filepath, orient="records")

                        uploaded_files[file_key] = filepath

            # Initialize data analyzer
            analyzer = DataAnalyzer()

            # Process the analysis request
            result = analyzer.analyze(questions_content, uploaded_files)

            # Format result for display
            result_json = json.dumps(result, indent=2)

            return render_template('index.html', result=result_json, questions=questions_content)

    except Exception as e:
        logging.error(f"Error in test_upload: {str(e)}")
        flash(f'Analysis failed: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 100MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
