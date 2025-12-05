import pandas as pd
import requests
import concurrent.futures
from functools import partial
from tqdm import tqdm

# ---------------------------
# CONFIGURATION
# ---------------------------
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"

# Parallelism Settings
MAX_WORKERS = 20
MAX_CHARS = 6000

# Input/Output Files
INPUT_FILE = "financial_subset_clean.csv"
OUTPUT_FILE = "financial_subset_clean_fintech_results.csv"

# ---------------------------
# PROMPT & NORMALIZATION
# ---------------------------

def build_fintech_prompt(page_text: str) -> str:
    """
    Builds a Few-Shot prompt to classify text as FINTECH or NOT_FINTECH
    based on the specific rules provided.
    """
    return f"""
You are a specialist classifier. Your job is to determine if a FINANCIAL company is specifically a FINTECH company.

DEFINITION:
FinTech (Financial Technology): Companies that develop, provide, or operate technology-based products, platforms, or software that deliver, enable, or transform financial services. The company must BUILD or OPERATE the technology — not simply USE fintech products or BROKER/REFER customers to financial services.

CLASSIFY AS FINTECH (YES) IF THE COMPANY:
- Builds/operates software, platforms, apps, or APIs for financial services
- Provides technology infrastructure for banks, lenders, insurers, or investors
- Offers digital-first financial products (neobanks, robo-advisors, crypto exchanges)
- Develops tools for compliance, identity verification, or regulatory reporting

CLASSIFY AS NOT FINTECH (NO) IF THE COMPANY:
- Is a broker, network, or appointed representative that REFERS customers to lenders/insurers
- Is a traditional financial adviser, accountant, or wealth manager using standard tools
- Simply ACCEPTS digital payments (e.g. PayPal, card machines) but doesn't provide payment tech
- MENTIONS fintech brands (Revolut, Monzo) but doesn't build fintech products
- Provides financial EDUCATION without a technology product
- Is a surveyor, valuation firm, or mortgage network

EXAMPLES:

Text: "We are a mortgage broker helping you find the best rates from high street banks."
Category: NOT_FINTECH

Text: "Our API allows developers to embed payment processing and identity verification into their apps instantly."
Category: FINTECH

Text: "Chartered accountants using Xero and QuickBooks to manage your tax returns."
Category: NOT_FINTECH

Text: "We built a proprietary AI-driven platform that automates loan underwriting for SME lenders."
Category: FINTECH

INSTRUCTIONS:
Read the text below and return ONLY the category name: 'FINTECH' or 'NOT_FINTECH'. Do not explain.

Text to classify:
\"\"\"{page_text}\"\"\"

Category:
""".strip()


def normalize_fintech_category(raw: str) -> str:
    """
    Normalizes the LLM output to FINTECH or NOT_FINTECH.
    """
    # Remove thoughts or reasoning if the model outputs them
    if "</think>" in raw:
        clean_raw = raw.split("</think>")[-1]
    else:
        clean_raw = raw

    raw_upper = clean_raw.strip().upper().replace("_", " ")

    # 1. Check Negative case first (Priority)
    if "NOT FINTECH" in raw_upper or "NO" in raw_upper:
        return "NOT_FINTECH"

    # 2. Check Positive case second
    if "FINTECH" in raw_upper or "YES" in raw_upper:
        return "FINTECH"

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

    prompt = build_fintech_prompt(page_text)

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
        cat = normalize_fintech_category(raw)
        return index, cat

    except Exception as e:
        return index, "ERROR"


def classify_webpages_parallel(df: pd.DataFrame, text_col: str, model: str) -> pd.Series:
    print(f"\n--- Classifying {len(df)} financial entities as Fintech/Not-Fintech ---")

    rows_to_process = list(df.iterrows())
    total_rows = len(rows_to_process)

    worker = partial(process_single_row, text_col=text_col, model=model, max_chars=MAX_CHARS)

    results_map = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(worker, r): r[0] for r in rows_to_process}

        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=total_rows, unit="row"):
            idx, result = future.result()
            results_map[idx] = result

    return pd.Series(results_map).reindex(df.index)


# ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":
    # 1. Load the Results from Step 1
    try:
        print(f"Loading {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Could not load input file: {e}")
        exit()

    # 2. Filter: Keep ONLY rows that were predicted as FINANCIAL
    # (Assuming the column name from previous script is 'predicted_financial')
    if "predicted_financial" not in df.columns:
        print("Error: Column 'predicted_financial' not found. Run the first script again.")
        exit()

    initial_count = len(df)
    df_fin = df[df["predicted_financial"] == "FINANCIAL"].copy()
    filtered_count = len(df_fin)

    print(f"Total rows loaded: {initial_count}")
    print(f"Rows retained (FINANCIAL only): {filtered_count}")
    print(f"Rows dropped: {initial_count - filtered_count}")

    if filtered_count == 0:
        print("No FINANCIAL rows found to process. Exiting.")
        exit()

    # 3. Determine Text Column
    text_col = None
    candidates = ["clean_content", "content", "front_page"]
    for c in candidates:
        if c in df_fin.columns:
            text_col = c
            break

    if not text_col:
        print(f"Error: Could not find text column. Available columns: {list(df_fin.columns)}")
        exit()

    print(f"Using column for text: {text_col}")

    # 4. Run Fintech Classification
    df_fin["is_fintech"] = classify_webpages_parallel(df_fin, text_col, MODEL_NAME)

    # 5. Save Results
    # We save only the financial rows with their new fintech label
    df_fin.to_csv(OUTPUT_FILE, index=False)

    # 6. Preview
    print("\n--- Fintech Classification Results Preview ---")
    print(df_fin[[text_col, "predicted_financial", "is_fintech"]].head(10))
    print(f"\nSaved to {OUTPUT_FILE}")