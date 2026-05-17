# Financial Intelligence & Analytics Dashboard

An interactive web application built with Streamlit for financial data analysis, time-series forecasting, and anomaly detection.

## Prerequisites
Before installing, ensure you have the following software installed on your system:
* Python 3.10 or higher
* Pip (Python package manager)
* PostgreSQL (or your preferred SQL database engine)

## Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
2. Set Up a Virtual Environment

It is highly recommended to isolate this application using a virtual environment:
# Create environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux/Mac)
source venv/bin/activate
3. Install Dependencies

Install all required packages listed in the requirements file:
pip install -r requirements.txt
4. Environment Configuration

Copy the template configuration file and populate it with your local system credentials:
cp .env.example .env
To launch the dashboard interface locally, execute:
streamlit run app.py
The application will automatically open in your default browser at http://localhost:8501.
