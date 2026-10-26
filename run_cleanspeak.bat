@echo off
title CleanSpeak - YouTube Video Analytics & Audience Insights
echo ========================================================
echo Starting CleanSpeak Application...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
