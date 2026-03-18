# Time-Series-Forecasting-for-Energy-Consumption-project

⚡ Time Series Forecasting — Energy Consumption
A Streamlit web app for forecasting energy consumption using SARIMA and Facebook Prophet models on the AEP hourly dataset.
🚀 Live Demo
Deploy on Streamlit Cloud — free & instant.
📊 Features

Upload your own AEP_hourly.csv dataset
Exploratory Data Analysis (EDA) with interactive Plotly charts
Seasonal Decomposition (trend, seasonality, residuals)
ADF Stationarity Test
SARIMA forecasting
Prophet forecasting with confidence intervals
Model comparison (RMSE, MAE, MAPE)

🗂️ Project Structure
energy-forecast-app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # App theme configuration
🛠️ Run Locally
bash# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/energy-forecast-app.git
cd energy-forecast-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
Then open http://localhost:8501 in your browser.
☁️ Deploy to Streamlit Cloud (Free)

Push this repo to GitHub
Go to share.streamlit.io
Click New app → select your repo
Set Main file path to app.py
Click Deploy

📁 Dataset
This app uses the AEP Hourly Energy Consumption dataset.
Download AEP_hourly.csv from Kaggle and upload it via the sidebar when running the app.

⚠️ Do NOT push the CSV to GitHub if it's large. Use the upload feature instead.

📦 Dependencies

streamlit — web app framework
prophet — Facebook's time-series forecasting
statsmodels — SARIMA model
plotly — interactive charts
scikit-learn — evaluation metrics
pandas, numpy, matplotlib

📈 Model Results (Sample)
ModelRMSEMAEMAPE (%)SARIMA5735.844749.6129.57Prophet2922.452358.8612.75
Prophet outperforms SARIMA on this dataset.
👤 Author
Built with ❤️ using Python + StreamlitShareContentTime_Series_Forecasting_for_Energy_Consumption_project.ipynbipynb
