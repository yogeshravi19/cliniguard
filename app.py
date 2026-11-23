"""
MedFactGuard Streamlit Application Entrypoint
Run with: streamlit run app.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Execute the MedFactGuard app
app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'medfactguard', 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    code = f.read()
exec(compile(code, app_path, 'exec'))
