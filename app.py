# Agentic AP Automation — CrewAI MVP

import os, io, re, json, time, uuid, sqlite3, textwrap
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from pypdf import PdfReader
from rapidfuzz import process as fuzz_process

# Optional/modern AI libs (gracefully degrade if not configured)
USE_LLM = False
USE_CREW = os.getenv("USE_CREW", "false").lower() == "true"  # 👈 respect secret
USE_QDRANT = False

# CrewAI import
try:
    from crewai import Agent, Task, Crew, Process
    if USE_CREW:
        print("✅ CrewAI imports successful & enabled")
    else:
        print("ℹ️ CrewAI installed but disabled via config")
except Exception as e:
    print(f"❌ CrewAI not installed: {e}")
    USE_CREW = False

# LiteLLM import
try:
    import litellm
    USE_LLM = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
except Exception:
    USE_LLM = False

# Qdrant import + client
qdrant = None
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    if os.getenv("QDRANT_URL"):
        qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        USE_QDRANT = True
        print("✅ Qdrant client initialized")
    else:
        print("ℹ️ Qdrant not configured (no URL in secrets)")
except Exception as e:
    USE_QDRANT = False
    print(f"❌ Qdrant import/init failed: {e}")

APP_TITLE = "Agentic AP Automation — CrewAI MVP"
DB_PATH = os.path.join("storage", "ap_demo.db")
DATA_DIR = "data"

# -----------------------------
# Utilities & Bootstrapping
# -----------------------------

def ensure_dirs():
    os.makedirs("storage", exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def seed_csvs_if_missing():
    vendors_fp = os.path.join(DATA_DIR, "vendors.csv")
    pos_fp = os.path.join(DATA_DIR, "pos.csv")
    receipts_fp = os.path.join(DATA_DIR, "receipts.csv")
    policies_fp = os.path.join(DATA_DIR, "policies.md")

    if not os.path.exists(vendors_fp):
        pd.DataFrame([
            {"vendor_id": "V100", "vendor_name": "Acme Supplies Inc.", "trusted": 1, "default_gl": "6100", "payment_terms": "NET30"},
            {"vendor_id": "V200", "vendor_name": "Globex Marketing LLC", "trusted": 0, "default_gl": "6200", "payment_terms": "NET45"},
            {"vendor_id": "V300", "vendor_name": "Initech Hardware", "trusted": 1, "default_gl": "6150", "payment_terms": "NET15"},
        ]).to_csv(vendors_fp, index=False)

    if not os.path.exists(pos_fp):
        pd.DataFrame([
            {"po_number": "PO-91001", "vendor_id": "V100", "amount": 2450.00, "currency": "USD", "status": "OPEN"},
            {"po_number": "PO-91002", "vendor_id": "V200", "amount": 8800.00, "currency": "USD", "status": "OPEN"},
            {"po_number": "PO-91003", "vendor_id": "V300", "amount": 512.35,  "currency": "USD", "status": "OPEN"},
        ]).to_csv(pos_fp, index=False)

    if not os.path.exists(receipts_fp):
        pd.DataFrame([
            {"po_number": "PO-91001", "received_amount": 2450.00, "received_qty_ok": 1},
            {"po_number": "PO-91002", "received_amount": 8800.00, "received_qty_ok": 0},
            {"po_number": "PO-91003", "received_amount": 512.35,  "received_qty_ok": 1},
        ]).to_csv(receipts_fp, index=False)

    if not os.path.exists(policies_fp):
        with open(policies_fp, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(
                """
                # AP Policies (Demo)
                - Auto-approve invoices < $5,000 from trusted vendors.
                - Require 2-way match (PO Amount ~ Invoice Amount within ±2%).
                - Require 3-way match if Goods Receipt available; otherwise fallback to 2-way.
                - Escalate marketing invoices > $50,000.
                - Duplicate rule: (vendor_id + invoice_number) must be unique.
                """
            ))


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS invoices (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            vendor_id TEXT,
            vendor_name TEXT,
            invoice_number TEXT,
            po_number TEXT,
            invoice_date TEXT,
            currency TEXT,
            subtotal REAL,
            tax REAL,
            total REAL,
            status TEXT,
            stp INTEGER,
            exceptions_json TEXT,
            decisions_json TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS posted_invoices (
            id TEXT PRIMARY KEY,
            posted_at TEXT,
            erp_doc_id TEXT,
            amount REAL,
            currency TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# -----------------------------
# Data Access Layer (CSV + SQLite)
# -----------------------------

def load_master_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vendors = pd.read_csv(os.path.join(DATA_DIR, "vendors.csv"))
    pos = pd.read_csv(os.path.join(DATA_DIR, "pos.csv"))
    receipts = pd.read_csv(os.path.join(DATA_DIR, "receipts.csv"))
    return vendors, pos, receipts


# -----------------------------
# Parsing & Extraction
# -----------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(chunks)


def parse_invoice_text(text: str, vendors_df: pd.DataFrame) -> Dict[str, Any]:
    # Safer regex matcher
    def find(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, text, flags)
        if not m:
            return None
        # return the first non-empty capturing group
        for i in range(1, (m.lastindex or 0) + 1):
            if m.group(i):
                return m.group(i).strip()
        return None

    # Invoice number
    invoice_number = find(r"(?i)(?:invoice\s*(?:no\.?|number|#)?)\s*[:\-]?\s*([A-Za-z0-9\/\-\_]+)")

    # PO number
    po_number = find(r"(?i)(?:po\s*(?:no\.?|number|#)?|purchase\s*order)\s*[:\-]?\s*([A-Za-z0-9\/\-\_]+)")

    # Date (handles ISO, US, EU, written formats)
    invoice_date = find(
        r"(?i)(?:invoice\s*date|date)\s*[:\-]?\s*("
        r"(?:\d{4}[-/]\d{2}[-/]\d{2})|"                 # 2025-09-20
        r"(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|"          # 09/20/2025
        r"(?:[A-Za-z]{3,9}\s+\d{1,2},\s*\d{2,4})"      # Sep 20, 2025
        r")"
    )

    # Subtotal
    subtotal = find(r"(?i)(?:subtotal|sub\s*total|amount)\s*[:\-]?\s*\$?\s*([0-9,]+\.?[0-9]{0,2})")

    # Tax
    tax = find(r"(?i)(?:tax|gst|vat|sales\s*tax)\s*[:\-]?\s*\$?\s*([0-9,]+\.?[0-9]{0,2})")

    # Total
    total = find(r"(?i)(?:invoice\s*total|total\s*due|amount\s*due|\s*total)\s*[:\-]?\s*\$?\s*([0-9,]+\.?[0-9]{0,2})")

    # Vendor guess by fuzzy match
    vendor_name_guess = None
    choices = vendors_df["vendor_name"].tolist()
    best = fuzz_process.extractOne(text[:5000], choices)
    if best and best[1] > 60:
        vendor_name_guess = best[0]

    # Convert to floats safely
    def safe_float(val):
        try:
            return float(str(val).replace(",", "")) if val else None
        except Exception:
            return None

    subtotal_val = safe_float(subtotal)
    tax_val = safe_float(tax) or 0.0
    total_val = safe_float(total)

    # Fallback: if subtotal missing, compute from total - tax
    if subtotal_val is None and total_val is not None:
        subtotal_val = total_val - tax_val

    return {
        "invoice_number": invoice_number,
        "po_number": po_number,
        "invoice_date": invoice_date,
        "currency": "USD",  # can improve later
        "subtotal": subtotal_val,
        "tax": tax_val,
        "total": total_val,
        "vendor_name": vendor_name_guess,
    }

# -----------------------------
# Core AP Validations
# -----------------------------

def map_vendor_name_to_id(vendor_name: Optional[str], vendors_df: pd.DataFrame) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not vendor_name:
        return None, None
    row = vendors_df[vendors_df.vendor_name.str.lower() == vendor_name.lower()]
    if row.empty:
        # fuzzy fallback
        best = fuzz_process.extractOne(vendor_name, vendors_df.vendor_name.tolist())
        if best and best[1] > 80:
            row = vendors_df[vendors_df.vendor_name == best[0]]
    if row.empty:
        return None, None
    return str(row.iloc[0].vendor_id), row.iloc[0].to_dict()


def duplicate_check(conn, vendor_id: str, invoice_number: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT 1 FROM invoices WHERE vendor_id=? AND invoice_number=?", (vendor_id, invoice_number))
    return c.fetchone() is not None


def two_way_match(po_row: pd.Series, invoice_total: float, tolerance: float = 0.02) -> bool:
    if po_row is None or pd.isna(po_row["amount"]) or invoice_total is None:
        return False
    po_amt = float(po_row["amount"])
    return abs(invoice_total - po_amt) <= tolerance * max(po_amt, 1.0)


def three_way_match(receipt_row: pd.Series, invoice_total: float, po_row: pd.Series, tolerance: float = 0.02) -> bool:
    if receipt_row is None or receipt_row.empty:
        return False
    if invoice_total is None or po_row is None or po_row.empty:
        return False
    amt_ok = abs(float(receipt_row["received_amount"]) - invoice_total) <= tolerance * max(float(po_row["amount"]), 1.0)
    qty_ok = int(receipt_row.get("received_qty_ok", 0)) == 1
    return bool(amt_ok and qty_ok)


# -----------------------------
# Optional: LLM helpers (via LiteLLM)
# -----------------------------

def llm_decide(summary: str, policy: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    if not USE_LLM:
        # Rule-based fallback
        decision = {
            "action": "APPROVE" if "< $5,000" in policy and "trusted" in summary else "REVIEW",
            "rationale": "Rule-based fallback decision."
        }
        return decision
    prompt = f"""
    You are an AP approvals expert. Given the invoice validation summary and policy, decide:
    - action: APPROVE, REJECT, or REVIEW
    - rationale: 1-2 sentences
    Summary:\n{summary}\n\nPolicy:\n{policy}
    """
    try:
        resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
        content = resp.choices[0].message["content"].strip()
        action = "REVIEW"
        m = re.search(r"action\s*[:\-]?\s*(APPROVE|REJECT|REVIEW)", content, re.I)
        if m:
            action = m.group(1).upper()
        return {"action": action, "rationale": content[:500]}
    except Exception as e:
        return {"action": "REVIEW", "rationale": f"LLM error or not configured: {e}"}


# -----------------------------
# CrewAI Orchestration (optional)
# -----------------------------

def run_crewai_chain(extraction: Dict[str, Any], checks: Dict[str, Any], policy_text: str) -> Dict[str, Any]:
    if not USE_CREW or not USE_LLM:
        return {"crewai": False, "notes": "CrewAI/LLM not enabled; using rule-based path."}

    summary_text = textwrap.dedent(f"""
        Extraction: {json.dumps(extraction, indent=2)}
        Checks: {json.dumps(checks, indent=2)}
    """)

    capture_agent = Agent(
        role="Capture Analyst",
        goal="Confirm extracted invoice fields and flag missing values",
        backstory="You specialize in understanding unstructured invoices and normalizing fields.",
        llm="gpt-4o-mini",
        verbose=False,
    )

    validation_agent = Agent(
        role="Validation Specialist",
        goal="Evaluate duplicates, 2-way/3-way match, and master data consistency",
        backstory="You are rigorous about policy compliance and data quality.",
        llm="gpt-4o-mini",
        verbose=False,
    )

    approver_agent = Agent(
        role="Approver Agent",
        goal="Make an approval decision based on policy and risk",
        backstory="You mimic a seasoned AP manager making prudent decisions.",
        llm="gpt-4o-mini",
        verbose=False,
    )

    t1 = Task(
        description=f"Review the following extraction and highlight issues or missing fields.\n\n{summary_text}",
        expected_output="A short bullet list of extraction issues and confidence hints.",
        agent=capture_agent,
    )

    t2 = Task(
        description="Based on the checks, state validation outcome for duplicates, 2-way, 3-way, and master data.",
        expected_output="A compact JSON with booleans: {duplicate: bool, two_way: bool, three_way: bool, master_ok: bool}",
        agent=validation_agent,
    )

    t3 = Task(
        description=f"Using company policy below, recommend APPROVE, REJECT, or REVIEW and justify in 1-2 sentences.\n\nPolicy:\n{policy_text}",
        expected_output="A line starting with 'action: APPROVE|REJECT|REVIEW' followed by rationale.",
        agent=approver_agent,
    )

    crew = Crew(
        agents=[capture_agent, validation_agent, approver_agent],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return {"crewai": True, "result": str(result)[:2000]}


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")
    ensure_dirs()
    seed_csvs_if_missing()
    init_db()

    st.title(APP_TITLE)
    st.caption("Zero/Low‑touch invoice processing • CrewAI Agents • RAG‑ready • Weekend MVP")

    with st.sidebar:
        st.subheader("⚙️ Settings")
        st.write("LLM configured:", "✅" if USE_LLM else "❌")
        st.write("CrewAI enabled:", "✅" if (USE_CREW and USE_LLM) else "❌")
        st.write("Qdrant configured:", "✅" if USE_QDRANT else "❌")
        st.markdown("—")
        auto_post = st.checkbox("Auto-post approved invoices to ERP (simulated)", value=True)
        tolerance = st.slider("Match tolerance (±)", min_value=0.0, max_value=0.1, value=0.02, step=0.005)

    tabs = st.tabs(["📤 Upload & Process", "📊 Dashboard", "🪵 Logs / DB", "📚 Policies & Data"])

    # Tab 1: Upload & Process
    with tabs[0]:
        st.subheader("Upload Invoice (PDF)")
        file = st.file_uploader("Drop a PDF invoice", type=["pdf"])
        if file is not None and st.button("Process Invoice", type="primary"):
            process_invoice(file, tolerance, auto_post)

    # Tab 2: Dashboard
    with tabs[1]:
        render_dashboard()

    # Tab 3: Logs / DB
    with tabs[2]:
        render_db_views()

    # Tab 4: Policies/Data
    with tabs[3]:
        render_policies_and_data()


# -----------------------------
# Processing Pipeline
# -----------------------------

def process_invoice(file, tolerance: float, auto_post: bool):
    vendors_df, pos_df, receipts_df = load_master_data()
    conn = sqlite3.connect(DB_PATH)

    # 1) Extract text
    raw_text = extract_text_from_pdf(file.getvalue())
    extraction = parse_invoice_text(raw_text, vendors_df)

    # Map vendor name to ID
    vendor_id, vendor_row = map_vendor_name_to_id(extraction.get("vendor_name"), vendors_df)
    extraction["vendor_id"] = vendor_id

    # 2) Validate core rules
    exceptions = []
    checks = {"duplicate": False, "two_way": False, "three_way": False, "master_ok": False}

    if not vendor_id:
        exceptions.append("Vendor not recognized")
    else:
        checks["master_ok"] = True

    inv_no = extraction.get("invoice_number")
    if vendor_id and inv_no:
        is_dup = duplicate_check(conn, vendor_id, inv_no)
        checks["duplicate"] = is_dup
        if is_dup:
            exceptions.append("Duplicate invoice detected")
    else:
        exceptions.append("Missing invoice number")

    # PO / matching
    po_row = None
    if extraction.get("po_number"):
        po_row = pos_df[pos_df.po_number == extraction["po_number"]]
        po_row = None if po_row.empty else po_row.iloc[0]

    receipt_row = None
    if extraction.get("po_number"):
        r = receipts_df[receipts_df.po_number == extraction["po_number"]]
        receipt_row = None if r.empty else r.iloc[0]

    checks["two_way"] = two_way_match(po_row, extraction.get("total"), tolerance) if po_row is not None else False
    checks["three_way"] = three_way_match(receipt_row, extraction.get("total"), po_row, tolerance) if receipt_row is not None else False

    # 3) Decision via LLM or rules
    with open(os.path.join(DATA_DIR, "policies.md"), "r", encoding="utf-8") as f:
        policy_text = f.read()

    # Build a human-readable summary
    vendor_label = f"{vendor_row['vendor_name']} (trusted={bool(vendor_row['trusted'])})" if vendor_row else str(extraction.get("vendor_name"))
    summary = f"Vendor: {vendor_label}\nInvoice: {inv_no}\nPO: {extraction.get('po_number')}\nTotal: {extraction.get('total')} {extraction.get('currency')}\nChecks: {checks}\nExceptions: {exceptions}"

    if USE_CREW and USE_LLM:
        orchestration = run_crewai_chain(extraction, checks, policy_text)
        rationale_blob = orchestration.get("result", orchestration.get("notes"))
        decision = llm_decide(summary + "\n" + rationale_blob, policy_text)
    else:
        orchestration = {"crewai": False, "notes": "CrewAI/LLM disabled"}
        # Rule-based approval
        trusted = bool(vendor_row["trusted"]) if vendor_row else False
        small = (extraction.get("total") or 0) < 5000
        if checks["duplicate"]:
            action = "REVIEW"
            reason = "Duplicate detected"
        elif trusted and small and checks["two_way"]:
            action = "APPROVE"
            reason = "Trusted vendor < $5k with 2-way match"
        elif checks["three_way"]:
            action = "APPROVE"
            reason = "3-way match OK"
        else:
            action = "REVIEW"
            reason = "Needs human review per policy"
        decision = {"action": action, "rationale": reason}

    # 4) Persist & optionally post
    inv_id = str(uuid.uuid4())
    stp = 1 if decision["action"] == "APPROVE" and len(exceptions) == 0 else 0

    record = {
        "id": inv_id,
        "created_at": datetime.utcnow().isoformat(),
        "vendor_id": vendor_id,
        "vendor_name": extraction.get("vendor_name"),
        "invoice_number": inv_no,
        "po_number": extraction.get("po_number"),
        "invoice_date": extraction.get("invoice_date"),
        "currency": extraction.get("currency"),
        "subtotal": extraction.get("subtotal"),
        "tax": extraction.get("tax"),
        "total": extraction.get("total"),
        "status": decision["action"],
        "stp": stp,
        "exceptions_json": json.dumps(exceptions),
        "decisions_json": json.dumps({"decision": decision, "orchestration": orchestration}),
    }

    c = conn.cursor()
    cols = ",".join(record.keys())
    placeholders = ",".join(["?"] * len(record))
    c.execute(f"INSERT INTO invoices ({cols}) VALUES ({placeholders})", tuple(record.values()))

    if auto_post and decision["action"] == "APPROVE":
        c.execute(
            "INSERT OR REPLACE INTO posted_invoices (id, posted_at, erp_doc_id, amount, currency) VALUES (?,?,?,?,?)",
            (inv_id, datetime.utcnow().isoformat(), f"ERP-{inv_no or inv_id[:8]}", record["total"] or 0, record["currency"]),
        )
    conn.commit()
    conn.close()

    # UI Feedback
    st.success(f"Invoice processed. Decision: {decision['action']}")
    with st.expander("Decision details"):
        st.write(decision)
    with st.expander("Extraction"):
        st.json(extraction)
    with st.expander("Checks"):
        st.json(checks)
    with st.expander("Orchestration (CrewAI)"):
        st.json(orchestration)


# -----------------------------
# Dashboard & Admin Tabs
# -----------------------------

def render_dashboard():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM invoices ORDER BY created_at DESC", conn)
    conn.close()

    if df.empty:
        st.info("No invoices yet. Upload a PDF in the first tab.")
        return

    c1, c2, c3 = st.columns(3)
    stp_rate = 100.0 * df["stp"].mean()
    approve_rate = 100.0 * (df["status"] == "APPROVE").mean()
    exception_rate = 100.0 * df["exceptions_json"].apply(lambda x: 0 if x == "[]" else 1).mean()

    c1.metric("STP %", f"{stp_rate:.0f}%")
    c2.metric("Approval %", f"{approve_rate:.0f}%")
    c3.metric("Exception %", f"{exception_rate:.0f}%")

    st.subheader("Recent Invoices")
    show = df[["created_at", "vendor_name", "invoice_number", "po_number", "total", "currency", "status", "stp"]]
    st.dataframe(show, use_container_width=True)


def render_db_views():
    conn = sqlite3.connect(DB_PATH)
    inv = pd.read_sql_query("SELECT * FROM invoices ORDER BY created_at DESC", conn)
    posted = pd.read_sql_query("SELECT * FROM posted_invoices ORDER BY posted_at DESC", conn)
    conn.close()

    st.markdown("### Invoices Table")
    st.dataframe(inv, use_container_width=True)
    st.markdown("### Posted Invoices (ERP Simulation)")
    st.dataframe(posted, use_container_width=True)


def render_policies_and_data():
    vendors_df, pos_df, receipts_df = load_master_data()
    st.markdown("### Policies")
    st.code(open(os.path.join(DATA_DIR, "policies.md"), "r", encoding="utf-8").read())

    st.markdown("### Vendor Master")
    st.dataframe(vendors_df, use_container_width=True)

    st.markdown("### Purchase Orders")
    st.dataframe(pos_df, use_container_width=True)

    st.markdown("### Goods Receipts")
    st.dataframe(receipts_df, use_container_width=True)


if __name__ == "__main__":

    main()

