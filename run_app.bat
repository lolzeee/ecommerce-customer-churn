@echo off
cd /d "%~dp0"
echo ========================================================
echo Launching E-Commerce Churn Intelligence Streamlit App...
echo ========================================================
"C:\Users\salin\AppData\Local\Programs\Python\Python39\python.exe" -m streamlit run app.py
pause
