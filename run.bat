@echo off
echo 🌱 Iniciando Syngenta Safety Assistant...
echo 🤗 Powered by Hugging Face Transformers

cd /d "%~dp0"

echo 📦 Verificando dependencias...
pip install -r requirements-hf.txt

echo 🚀 Iniciando aplicación...
streamlit run src/app_hf.py

pause