import pandas as pd
import requests
import concurrent.futures
from functools import partial
from tqdm import tqdm  # <--- NEW IMPORT

# ---------------------------
# CONFIGURATION
# ---------------------------
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"

# Valid Binary Categories
CATEGORIES = ["FINANCIAL", "NOT_FINANCIAL"]

# PARALLELISM SETTINGS
# Note: If your progress bar speed is very slow, try lowering this.
# Running 10 concurrent LLM requests can choke local hardware.
MAX_WORKERS = 20
MAX_CHARS = 6000


# ---------------------------
# PROMPT & NORMALIZATION
# ---------------------------

def build_prompt(page_text: str) -> str:
    """
    Builds a Few-Shot prompt to classify text as FINANCIAL or NOT_FINANCIAL.
    """
    return f"""
You are a strict classifier. Your job is to read the webpage content and classify the business entity into exactly one of two categories: FINANCIAL or NOT_FINANCIAL.

DEFINITIONS:
1. FINANCIAL: The core business is managing money, investments, insurance, accounting, mortgages, or providing loans.
   - Includes: IFAs, Accountants, Banks, Insurance Brokers, Wealth Managers, Car Finance Brokers.
2. NOT_FINANCIAL: The core business is selling physical goods, non-financial services, hospitality, or trades.
   - Includes: Retail shops, Taxis, Railways, Manufacturers, Recruitment (unless specific to finance staffing, usually NOT_FINANCIAL), Web Design.

EXAMPLES:

Text: "We offer comprehensive wealth management, pension planning, and mortgage advice to help secure your future."
Category: FINANCIAL

Text: "Welcome to A2B Taxis. We offer airport transfers, school runs, and private hire vehicles 24/7."
Category: NOT_FINANCIAL

Text: "Specialist chartered accountants providing tax returns, bookkeeping, and payroll services for small businesses."
Category: FINANCIAL

Text: "We sell trophies, medals, and offer shoe repair and key cutting services. Visit our shop today."
Category: NOT_FINANCIAL

Text: "Our steam railway offers festive trips, dining experiences, and a journey through the Highlands."
Category: NOT_FINANCIAL

INSTRUCTIONS:
Read the text below and return ONLY the category name. Do not explain.

Text to classify:
\"\"\"{page_text}\"\"\"

Category:
""".strip()


def normalize_category(raw: str) -> str:
    """
    Normalizes the LLM output.
    CRITICAL: Check for 'NOT_FINANCIAL' (or 'NOT FINANCIAL') *before* 'FINANCIAL'
    because 'FINANCIAL' is a substring of 'NOT FINANCIAL'.
    """
    # Remove thoughts or reasoning if the model outputs them
    if "</think>" in raw:
        clean_raw = raw.split("</think>")[-1]
    else:
        clean_raw = raw

    raw_upper = clean_raw.strip().upper().replace("_", " ")

    # 1. Check Negative case first
    if "NOT FINANCIAL" in raw_upper:
        return "NOT_FINANCIAL"

    # 2. Check Positive case second
    if "FINANCIAL" in raw_upper:
        return "FINANCIAL"

    return "UNKNOWN"


# ---------------------------
# WORKER & PARALLEL CLASSIFIER
# ---------------------------

def process_single_row(row_tuple, text_col, model, max_chars):
    index, row = row_tuple

    # Ensure text is a string
    page_text = str(row[text_col]) if pd.notna(row[text_col]) else ""

    if len(page_text) > max_chars:
        page_text = page_text[:max_chars]

    prompt = build_prompt(page_text)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096,
            "temperature": 0,  # Deterministic
        },
    }

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        cat = normalize_category(raw)
        return index, cat

    except Exception as e:
        # Returning error text so we can see it in the CSV later if needed
        return index, "ERROR"


def classify_webpages_parallel(df: pd.DataFrame, text_col: str, model: str) -> pd.Series:
    print(f"\n--- Classifying {len(df)} webpages using '{text_col}' ---")

    rows_to_process = list(df.iterrows())
    total_rows = len(rows_to_process)

    # Create the partial function for the worker
    worker = partial(process_single_row, text_col=text_col, model=model, max_chars=MAX_CHARS)

    results_map = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(worker, r): r[0] for r in rows_to_process}

        # <--- TQDM WRAPPER HERE --->
        # tqdm wraps as_completed to provide the progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=total_rows, unit="row"):
            idx, result = future.result()
            results_map[idx] = result

    return pd.Series(results_map).reindex(df.index)


# ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":
    # 1. Load Data
    try:
        df = pd.read_csv(r"C:\Users\gmpal\PycharmProjects\Bristol_Datathon\problem1_2021_finance.csv")
    except Exception as e:
        print(f"Could not load input file: {e}")
        exit()

    # 2. Determine Text Column
    text_col = None
    candidates = ["clean_content", "content", "front_page"]

    for c in candidates:
        if c in df.columns:
            text_col = c
            break

    if not text_col:
        print(f"Error: Could not find text column. Available columns: {list(df.columns)}")
        exit()

    print(f"Using column: {text_col}")

    # 3. Run Classification (With Progress Bar)
    df["predicted_financial"] = classify_webpages_parallel(df, text_col, MODEL_NAME)

    # 4. Save Results
    output_filename = "classified_financial_results_12b.csv"
    df.to_csv(output_filename, index=False)

    # 5. Preview
    print("\n--- Results Preview ---")
    print(df[[text_col, "predicted_financial"]].head(10))
    print(f"\nSaved to {output_filename}")