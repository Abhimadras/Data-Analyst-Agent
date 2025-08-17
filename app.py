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
        # Look for any .txt file in request.files
        questions_file = None
        for file_key, file in request.files.items():
            if file.filename.lower().endswith(".txt"):
                questions_file = file
                break

        if questions_file is None:
            return jsonify({"error": "At least one .txt file with questions is required"}), 400

        if questions_file.filename == '':
            return jsonify({"error": "Questions file cannot be empty"}), 400

        # Read questions
        questions_content = questions_file.read().decode('utf-8')
        if not questions_content.strip():
            return jsonify({"error": "Questions file cannot be empty"}), 400

        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_files = {}

            # Save the questions file
            questions_path = os.path.join(temp_dir, questions_file.filename)
            with open(questions_path, 'w', encoding='utf-8') as f:
                f.write(questions_content)
            uploaded_files['questions'] = questions_path

            # Process other uploaded files (CSV, JSON, etc.)
            for file_key in request.files:
                file = request.files[file_key]
                if file and file.filename and allowed_file(file.filename):
                    if file.filename == questions_file.filename:
                        continue  # skip, already handled
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(temp_dir, filename)
                    file.save(filepath)

                    # If it's a CSV/JSON, coerce numerics
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

@app.route('/', methods=['POST'])
def root_post():
    # Simply forward POST / to POST /api/
    return analyze_data()
@app.route('/test', methods=['POST'])
def test_upload():
    """Test endpoint for the web interface"""
    try:
        # Look for any .txt file in the uploaded files
        questions_file = None
        for file_key, file_obj in request.files.items():
            if file_obj and file_obj.filename and file_obj.filename.lower().endswith(".txt"):
                questions_file = file_obj
                break

        if not questions_file:
            flash('A .txt file containing questions is required', 'error')
            return redirect(url_for('index'))

        if questions_file.filename == '':
            flash('Questions file cannot be empty', 'error')
            return redirect(url_for('index'))

        # Read questions
        questions_content = questions_file.read().decode('utf-8')
        if not questions_content.strip():
            flash('Questions file cannot be empty', 'error')
            return redirect(url_for('index'))

        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_files = {}

            # Save the questions file (whatever its original name is)
            q_filename = secure_filename(questions_file.filename)
            questions_path = os.path.join(temp_dir, q_filename)
            with open(questions_path, 'w', encoding='utf-8') as f:
                f.write(questions_content)
            uploaded_files[q_filename] = questions_path

            # Process other uploaded files
            for file_key in request.files:
                if file_key != questions_file:  # skip the questions file itself
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
