# Data Analyst API

## Overview

This is a Flask-based web application that provides a data analysis API. The system accepts multiple file uploads (CSV, JSON, Parquet, images, and text files) along with analysis questions and returns structured results including data analysis, visualizations, and web scraping capabilities. The application is designed to handle various data analysis tasks within a 3-minute timeout window and can generate charts, perform statistical analysis, and scrape web content.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Web Framework
- **Flask** - Lightweight web framework chosen for its simplicity and flexibility in handling file uploads and API endpoints
- **Werkzeug ProxyFix** - Middleware for handling proxy headers in deployment environments
- Single main route (`/api/`) that accepts POST requests with multipart form data

### Data Processing Architecture
- **DataAnalyzer class** - Core analysis engine that processes questions and uploaded files
- **Task-based processing** - Questions are parsed into individual tasks that are processed sequentially
- **Timeout management** - 3-minute maximum execution time per analysis session
- **Multi-format data loading** - Supports CSV, JSON, Parquet, and image files

### Data Analysis Stack
- **Pandas** - Primary data manipulation and analysis library
- **NumPy** - Numerical computing support
- **DuckDB** - In-memory SQL database for efficient querying of large datasets
- **Matplotlib/Seaborn** - Visualization libraries with non-interactive backend for server deployment
- **PyArrow** - Parquet file format support

### Web Scraping Component
- **Trafilatura** - Primary web content extraction tool
- **BeautifulSoup** - HTML parsing fallback
- **Requests** - HTTP client with timeout handling
- Specialized function for extracting clean text content from websites

### File Handling
- **Temporary file management** - Uses Python's tempfile for secure file processing
- **File type validation** - Whitelist of allowed extensions (txt, csv, json, png, jpg, jpeg, parquet)
- **Size limits** - 100MB maximum file size per upload
- **Secure filename handling** - Werkzeug's secure_filename for sanitization

### Response Architecture
- **Flexible output formats** - JSON arrays, objects, base64-encoded images, or plain text
- **Base64 image encoding** - Charts and visualizations returned as data URIs under 100KB
- **Error handling** - Structured error responses with appropriate HTTP status codes
- **Single vs. multiple task handling** - Automatically formats response based on number of analysis tasks

### Frontend Interface
- **Bootstrap-based UI** - Dark theme with responsive design
- **File upload interface** - Multi-file drag-and-drop support
- **Real-time feedback** - Flash messages for user notifications
- **FontAwesome icons** - Enhanced visual interface elements

## External Dependencies

### Core Libraries
- **Flask** - Web framework for API endpoints and routing
- **Pandas** - Data manipulation and analysis
- **DuckDB** - High-performance analytical database
- **Matplotlib/Seaborn** - Data visualization and plotting
- **Trafilatura** - Web content extraction and scraping

### File Processing
- **PyArrow** - Parquet file format support
- **Pillow (PIL)** - Image processing capabilities
- **Werkzeug** - File upload utilities and security

### Web Technologies
- **Bootstrap CDN** - Frontend styling and responsive design
- **FontAwesome CDN** - Icon library for enhanced UI
- **BeautifulSoup** - HTML parsing for web scraping fallback

### Development Tools
- **Logging** - Built-in Python logging for debugging and monitoring
- **Requests** - HTTP client library for web scraping operations

### Environment Configuration
- **Environment variables** - Session secret and configuration management
- **Tempfile** - Secure temporary file handling
- **Base64** - Image encoding for API responses