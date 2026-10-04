@echo off
cd /d "%~dp0"
echo Installing / checking packages (first run takes a few minutes)...
python -m pip install -q -r requirements.txt
echo Starting the app - it will open in your browser.
python -m streamlit run app.py
pause
