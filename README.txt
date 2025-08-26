# BikeBot-Offline 🏍️

## How to Run
1. Extract the zip file.
2. Open terminal in the extracted folder.
3. Create and activate virtual environment (optional but recommended):
   python -m venv .venv
   .venv\Scripts\activate    (Windows)
   source .venv/bin/activate   (Linux/Mac)

4. Install dependencies:
   pip install -r requirements.txt

5. Run the app:
   streamlit run app.py

## Notes
- On first run, model (~4GB) will auto-download to `models/` folder with progress bar.
- If you already have mistral-7b-instruct-v0.2.Q4_K_M.gguf, place it in `models/` to skip download.
