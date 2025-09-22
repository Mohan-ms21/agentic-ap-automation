# Agentic AP Automation — Pro Edition (Agents + RAG + HITL + Queues)

import os
import io
import re
import json
import uuid
import math
import time
import hashlib
import sqlite3
import textwrap
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from rapidfuzz import process as fuzz_process

# ==============================================================================
# Feature flags & paths
# ==============================================================================
APP_TITLE = "Agentic AP Automation — Pro Edition"
DB_PATH = os.path.join("storage", "ap_demo.db")
DATA_DIR = "data"

USE_CREW = os.getenv("USE_CREW", "false").strip().lower() == "true"
USE_LLM = False
USE_QDRANT = False
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")

# ==============================================================================
# Optional libraries (graceful import)
# ==============================================================================
try:
    from crewai import Agent, Task, Crew, Process
    if USE_CREW:
        print("✅ CrewAI enabled")
    else:
        print("ℹ️ CrewAI installed but disabled (USE_CREW=false)")
except Exception as e:
    print(f"❌ CrewAI import failed: {e}")
    USE_CREW = False

try:
    import litellm
    USE_LLM = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    print("✅ LLM configured" if USE_LLM else "ℹ️ No LLM keys; rules fallback")
except Exception as e:
    print(f"❌ litellm unavailable: {e}")
    USE_LLM = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    if os.getenv("QDRANT_URL"):
        USE_QDRANT = True
        print("✅ Qdrant configured")
    else:
        print("ℹ️ Qdrant not configured; using in-memory RAG")
except Exception:
    USE_QDRANT = False
    print("ℹ️ qdrant-client not installed; using in-memory RAG")

# ==============================================================================
# Bootstrapping
# ==============================================================================
def ensure_dirs():
    os.makedirs("storage", exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def seed_csvs_if_missing():
    vendors_fp = os.path.join(DATA_DIR, "vendors.csv")
    pos_fp = os.path.join(DATA_DIR, "pos.csv")
    receipts_fp = os.path.join(DATA_DIR, "receipts.csv")
    policies_fp = os.path.join(DATA_DIR, "policies.md")
    examples_fp = os.path.join(DATA_DIR, "examples.md")
    gl_map_fp = os.path.join(DATA_DIR, "gl_map.csv")  # seed GL hints for non-PO

    if not os.path.exists(vendors_fp):
        pd.DataFrame([
            {"vendor_id": "V100", "vendor_name": "Acme Supplies Inc.",  "trusted": 1, "default_gl": "6100", "payment_terms": "NET30", "email": "billing@acme.example", "category": "Office Supplies"},
            {"vendor_id": "V200", "vendor_name": "Globex Marketing LLC", "trusted": 0, "default_gl": "6200", "payment_terms": "NET45", "email": "ap@globex.example",     "category": "Marketing"},
            {"vendor_id": "V300", "vendor_name": "Initech Hardware",     "trusted": 1, "default_gl": "6150", "payment_terms": "NET15", "email": "finance@initech.example","category": "IT Hardware"},
        ]).to_csv(vendors_fp, index=False)

    if not os.path.exists(pos_fp):
        pd.DataFrame([
            {"po_number": "PO-91001", "vendor_id": "V100", "amount": 2450.00, "currency": "USD", "status": "OPEN", "owner_email": "owner.acme@example.com"},
            {"po_number": "PO-91002", "vendor_id": "V200", "amount": 8800.00, "currency": "USD", "status": "OPEN", "owner_email": "owner.globex@example.com"},
            {"po_number": "PO-91003", "vendor_id": "V300", "amount":  512.35, "currency": "USD", "status": "OPEN", "owner_email": "owner.initech@example.com"},
        ]).to_csv(pos_fp, index=False)

    if not os.path.exists(receipts_fp):
        pd.DataFrame([
            {"po_number": "PO-91001", "received_amount": 2450.00, "received_qty_ok": 1},
            {"po_number": "PO-91002", "received_amount": 8800.00, "received_qty_ok": 0},
            {"po_number": "PO-91003", "received_amount":  512.35, "received_qty_ok": 1},
        ]).to_csv(receipts_fp, index=False)

    if not os.path.exists(policies_fp):
        with open(policies_fp, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""\
                # AP Policies (Demo)
                - Auto-approve invoices < $5,000 from trusted vendors if 2-way or 3-way match succeeds.
                - Require 3-way match if GR available; otherwise fallback to 2-way.
                - Marketing invoices > $50,000 → escalate to Finance Director.
                - Duplicate rule: (vendor_id + invoice_number) must be unique.
                - Non-PO invoices require GL + Cost Center (confidence ≥ 0.9 → autopilot).
                - Suspicious bank detail change → hold & route to AP manager.
            """))

    if not os.path.exists(examples_fp):
        with open(examples_fp, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent("""\
                # Historical Decisions (Few-shot)
                - Trusted vendor under $5k with 2-way OK → APPROVE.
                - 3-way match amount & qty OK → APPROVE.
                - Duplicate or missing invoice number → REVIEW + vendor clarification.
                - Large mismatch to PO or tax anomaly → REVIEW/REJECT.
                - Marketing spend > $50k → escalate to FD approval.
            """))

    if not os.path.exists(gl_map_fp):
        pd.DataFrame([
            {"keyword": "marketing|ads|campaign", "gl": "6200", "cost_center": "MKT", "project": "", "score": 0.95},
            {"keyword": "office|supplies|stationery", "gl": "6100", "cost_center": "GNA", "project": "", "score": 0.93},
            {"keyword": "hardware|laptop|monitor|it", "gl": "6150", "cost_center": "IT", "project": "", "score": 0.92},
        ]).to_csv(gl_map_fp, index=False)

def _column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return column in [r[1] for r in cur.fetchall()]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
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
        decisions_json TEXT,
        rationale TEXT,
        crewai_rationale TEXT,
        gl_json TEXT,               -- GL Coding agent output
        route_json TEXT,            -- Routing agent output (approvers, SLA)
        anomaly_json TEXT,          -- Fraud/Anomaly agent output
        created_at_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS posted_invoices (
        id TEXT PRIMARY KEY,
        posted_at TEXT,
        erp_doc_id TEXT,
        amount REAL,
        currency TEXT
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        direction TEXT,   -- outbound|inbound
        status TEXT,      -- draft|sent|received|handled
        to_email TEXT,
        from_email TEXT,
        subject TEXT,
        body TEXT,
        related_invoice_id TEXT,
        created_at TEXT
    )""")

    # Migrations (safe add of new columns)
    migrations = {
        "gl_json":         "ALTER TABLE invoices ADD COLUMN gl_json TEXT",
        "route_json":      "ALTER TABLE invoices ADD COLUMN route_json TEXT",
        "anomaly_json":    "ALTER TABLE invoices ADD COLUMN anomaly_json TEXT",
        "rationale":       "ALTER TABLE invoices ADD COLUMN rationale TEXT",
        "crewai_rationale":"ALTER TABLE invoices ADD COLUMN crewai_rationale TEXT",
        "created_at_ts":   "ALTER TABLE invoices ADD COLUMN created_at_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    }
    for col, ddl in migrations.items():
        try:
            if not _column_exists(c, "invoices", col):
                print(f"⚙️ Adding missing column: {col}")
                c.execute(ddl)
        except Exception as e:
            print(f"⚠️ Migration skipped for {col}: {e}")

    conn.commit()
    conn.close()

# ==============================================================================
# Master data
# ==============================================================================
def load_master_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vendors = pd.read_csv(os.path.join(DATA_DIR, "vendors.csv"))
    pos     = pd.read_csv(os.path.join(DATA_DIR, "pos.csv"))
    receipts= pd.read_csv(os.path.join(DATA_DIR, "receipts.csv"))
    gl_map  = pd.read_csv(os.path.join(DATA_DIR, "gl_map.csv"))
    return vendors, pos, receipts, gl_map

# ==============================================================================
# Extraction / Parsing
# ==============================================================================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def parse_invoice_text(text: str, vendors_df: pd.DataFrame) -> Dict[str, Any]:
    def find(pattern, flags=re.IGNORECASE):
        m = re.search(pattern, text, flags)
        if not m: return None
        if m.lastindex:
            for i in range(1, m.lastindex + 1):
                if m.group(i): return m.group(i).strip()
        return m.group(0).strip()
    def to_float(s: Optional[str]) -> Optional[float]:
        try:
            return float(str(s).replace(",", "")) if s else None
        except Exception:
            return None

    invoice_number = find(r"(?:invoice\s*(?:no\.?|number|#)?)\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)")
    po_number      = find(r"(?:(?:po|purchase\s*order)\s*(?:no\.?|number|#)?)\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)")
    invoice_date   = find(r"(?:invoice\s*date|date)\s*[:\-]?\s*([A-Za-z0-9,\/\- ]+)")
    subtotal       = to_float(find(r"(?:subtotal|sub\s*total)\s*[:\-]?\s*\$?([0-9,]+\.\d{2})"))
    tax            = to_float(find(r"(?:tax|vat|gst)\s*[:\-]?\s*\$?([0-9,]+\.\d{2})")) or 0.0
    total          = to_float(find(r"(?:total|amount\s*due)\s*[:\-]?\s*\$?([0-9,]+\.\d{2})"))

    # Fuzzy vendor
    vendor_guess = None
    choices = vendors_df["vendor_name"].tolist()
    best = fuzz_process.extractOne(text[:5000], choices)
    if best and best[1] > 60:
        vendor_guess = best[0]

    if subtotal is None and total is not None:
        subtotal = total - tax

    return {
        "invoice_number": invoice_number,
        "po_number": po_number,
        "invoice_date": invoice_date,
        "currency": "USD",
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "vendor_name": vendor_guess,
        "description": text[:1500]  # crude description context for GL agent
    }

# ==============================================================================
# AP validations
# ==============================================================================
def map_vendor_name_to_id(vendor_name: Optional[str], vendors_df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
    if not vendor_name: return None, None
    row = vendors_df[vendors_df.vendor_name.str.lower() == vendor_name.lower()]
    if row.empty:
        best = fuzz_process.extractOne(vendor_name, vendors_df.vendor_name.tolist())
        if best and best[1] > 80:
            row = vendors_df[vendors_df.vendor_name == best[0]]
    if row.empty: return None, None
    return str(row.iloc[0].vendor_id), row.iloc[0]

def duplicate_check(conn: sqlite3.Connection, vendor_id: str, invoice_number: str) -> bool:
    c = conn.cursor()
    c.execute("SELECT 1 FROM invoices WHERE vendor_id=? AND invoice_number=?", (vendor_id, invoice_number))
    return c.fetchone() is not None

def two_way_match(po_row: Optional[pd.Series], invoice_total: Optional[float], tol: float = 0.02) -> bool:
    if po_row is None or pd.isna(po_row.get("amount")) or invoice_total is None: return False
    po_amt = float(po_row["amount"])
    return abs(invoice_total - po_amt) <= tol * max(po_amt, 1.0)

def three_way_match(receipt_row: Optional[pd.Series], invoice_total: Optional[float], po_row: Optional[pd.Series], tol: float = 0.02) -> bool:
    if receipt_row is None or po_row is None or invoice_total is None: return False
    amt_ok = abs(float(receipt_row["received_amount"]) - invoice_total) <= tol * max(float(po_row["amount"]), 1.0)
    qty_ok = int(receipt_row.get("received_qty_ok", 0)) == 1
    return bool(amt_ok and qty_ok)

# ==============================================================================
# RAG (hash embeddings + optional Qdrant)
# ==============================================================================
HASH_DIM = 1024

def hash_embed(text: str, dim: int = HASH_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[A-Za-z0-9]+", (text or "").lower()):
        hv = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16) % dim
        vec[hv] += 1.0
    n = np.linalg.norm(vec)
    return vec / (n + 1e-8)

def build_rag_corpus() -> List[Dict[str, Any]]:
    items = []
    for name in ["policies.md", "examples.md"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            txt = open(p, "r", encoding="utf-8").read()
            items.append({"id": f"doc:{name}", "type": "doc", "title": name, "text": txt})
    # historical decisions as weak supervision
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT vendor_name, po_number, total, currency, status, decisions_json FROM invoices", conn)
        for _, r in df.iterrows():
            items.append({
                "id": f"hist:{(r.get('po_number') or uuid.uuid4().hex[:8])}",
                "type": "history",
                "title": f"Decision {r['status']}",
                "text": f"{r['vendor_name']} {r['po_number']} {r['total']} {r['currency']}. {r['decisions_json']}"
            })
    except Exception:
        pass
    finally:
        conn.close()
    return items

def ensure_qdrant_collection(client: QdrantClient, name: str, size: int = HASH_DIM):
    try:
        client.get_collection(name)
    except Exception:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )

def upsert_rag_items(items: List[Dict[str, Any]]):
    if USE_QDRANT:
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        ensure_qdrant_collection(client, "ap_rag", HASH_DIM)
        points = []
        for it in items:
            vec = hash_embed(it["text"])
            points.append(PointStruct(id=abs(hash(it["id"])) % (10**12), vector=vec.tolist(), payload=it))
        if points:
            client.upsert("ap_rag", points)
        st.session_state["rag_mode"] = "qdrant"
        return
    # in-memory
    mat = []; meta = []
    for it in items:
        mat.append(hash_embed(it["text"]))
        meta.append(it)
    st.session_state["rag_mat"] = np.vstack(mat) if mat else np.zeros((0, HASH_DIM), dtype=np.float32)
    st.session_state["rag_meta"] = meta
    st.session_state["rag_mode"] = "memory"

def rag_search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    qv = hash_embed(query)
    mode = st.session_state.get("rag_mode", "memory")
    if mode == "qdrant" and USE_QDRANT:
        client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
        ensure_qdrant_collection(client, "ap_rag", HASH_DIM)
        hits = client.search(collection_name="ap_rag", query_vector=qv.tolist(), limit=k)
        return [{**h.payload, "_score": float(h.score)} for h in hits]
    # memory
    mat: np.ndarray = st.session_state.get("rag_mat", np.zeros((0, HASH_DIM), dtype=np.float32))
    meta: List[Dict[str, Any]] = st.session_state.get("rag_meta", [])
    if mat.shape[0] == 0: return []
    sims = mat @ qv
    idx = np.argsort(-sims)[:k]
    return [{**meta[i], "_score": float(sims[i])} for i in idx]

# ==============================================================================
# Messaging (Outbox/Inbox)
# ==============================================================================
def save_message(direction: str, status: str, to_email: str, from_email: str, subject: str, body: str, related_invoice_id: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    msg = {
        "id": str(uuid.uuid4()),
        "direction": direction, "status": status,
        "to_email": to_email, "from_email": from_email,
        "subject": subject, "body": body,
        "related_invoice_id": related_invoice_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    c.execute(f"INSERT INTO messages ({','.join(msg.keys())}) VALUES ({','.join(['?']*len(msg))})", tuple(msg.values()))
    conn.commit(); conn.close()
    return msg["id"]

def draft_vendor_email(issue: str, extraction: Dict[str, Any], vendor_row: Optional[pd.Series]) -> Dict[str, str]:
    to_email = vendor_row["email"] if vendor_row is not None else "vendor@example.com"
    inv = extraction.get("invoice_number") or "(missing)"
    po  = extraction.get("po_number") or "(missing)"
    subject = f"[Action Required] Invoice {inv} / PO {po} — {issue}"
    body = textwrap.dedent(f"""\
        Hello {extraction.get('vendor_name','Vendor')},

        Our Accounts Payable system detected an issue with invoice {inv} for PO {po}:

        • Issue: {issue}
        • Amount: {extraction.get('total')} {extraction.get('currency')}
        • Date: {extraction.get('invoice_date') or '(not found)'}
        • Vendor: {extraction.get('vendor_name')}

        Please reply with clarification or an updated invoice.
        Thank you,
        AP Automation Bot
    """)
    return {"to_email": to_email, "subject": subject, "body": body}

def draft_approver_nudge(to_email: str, inv_id: str, summary: str) -> Dict[str, str]:
    subject = f"[Approval Needed] {summary}"
    body = f"Hi,\n\nPlease approve invoice:\n{summary}\n\nLink ref: {inv_id}\n\nThanks,\nAP Bot"
    return {"to_email": to_email, "subject": subject, "body": body}

# ==============================================================================
# LLM fallback (simple)
# ==============================================================================
def llm_decide(summary: str, policy: str, model: str = None) -> Dict[str, Any]:
    if not USE_LLM:
        # Rules fallback
        action = "REVIEW"
        two_ok = "two_way': True" in summary
        three_ok = "three_way': True" in summary
        trusted = "trusted=True" in summary
        m = re.search(r"total=(\d+(?:\.\d+)?)", summary)
        tot = float(m.group(1)) if m else 1e9
        if (three_ok or (two_ok and trusted and tot < 5000)):
            action = "APPROVE"
        return {"action": action, "rationale": "Rule-based fallback (LLM disabled)."}

    model = model or DEFAULT_MODEL
    prompt = f"""You are an AP approvals expert. Decide APPROVE/REJECT/REVIEW with a one-line rationale.
Summary:
{summary}

Policy/examples:
{policy}
"""
    try:
        resp = litellm.completion(model=model, messages=[{"role":"user","content":prompt}], temperature=0)
        text = resp.choices[0].message["content"].strip()
        m = re.search(r"(APPROVE|REJECT|REVIEW)", text, re.I)
        action = m.group(1).upper() if m else "REVIEW"
        return {"action": action, "rationale": text[:500]}
    except Exception as e:
        return {"action": "REVIEW", "rationale": f"LLM error: {e}"}

# ==============================================================================
# Agents (CrewAI): Capture, Validation, Risk/Compliance, GL Coding, Approver Routing,
# Exception, Final Approver. We keep Final Approver as source-of-truth decision maker.
# ==============================================================================
def _sj(o): 
    try: return json.dumps(o, ensure_ascii=False, indent=2)
    except Exception: return str(o)

def _parse_crewai_json(raw: str) -> Dict[str, str]:
    try:
        j = json.loads(raw)
        act = j.get("action","").upper()
        if act not in {"APPROVE","REJECT","REVIEW"}: act = "REVIEW"
        rat = (j.get("rationale") or "").strip()[:800] or "No rationale provided."
        return {"action": act, "rationale": rat}
    except Exception:
        m = re.search(r"(APPROVE|REJECT|REVIEW)", raw, re.I)
        act = (m.group(1).upper() if m else "REVIEW")
        return {"action": act, "rationale": raw[:800]}

def run_agents(extraction: Dict[str,Any], checks: Dict[str,Any], context_chunks: List[Dict[str,Any]], gl_suggestion: Dict[str,Any], route_suggestion: Dict[str,Any], anomaly: Dict[str,Any]) -> Dict[str,Any]:
    """CrewAI orchestration; Final Approver returns strict JSON decision. Falls back to LLM/rules if disabled."""
    if not (USE_CREW and USE_LLM):
        return {"crewai": False, "decision": None, "raw": "CrewAI/LLM disabled"}

    context = "\n\n".join([f"({i+1}) {c['title']}:\n{c['text'][:1200]}" for i, c in enumerate(context_chunks)])
    summary = f"Extraction:\n{_sj(extraction)}\n\nChecks:\n{_sj(checks)}\n\nGL Suggestion:\n{_sj(gl_suggestion)}\n\nRouting:\n{_sj(route_suggestion)}\n\nAnomaly:\n{_sj(anomaly)}"

    # Agents
    capture  = Agent(role="Capture Analyst", goal="Validate extracted fields & quality", llm=DEFAULT_MODEL, verbose=False)
    validate = Agent(role="Validation Agent", goal="Evaluate duplicates & matching outcomes", llm=DEFAULT_MODEL, verbose=False)
    risk     = Agent(role="Risk & Compliance", goal="Assess policy risks from context", llm=DEFAULT_MODEL, verbose=False)
    router   = Agent(role="Approver Routing Agent", goal="Confirm approver list & SLA", llm=DEFAULT_MODEL, verbose=False)
    glcoder  = Agent(role="GL Coding Agent", goal="Confirm/adjust GL coding suggestion", llm=DEFAULT_MODEL, verbose=False)
    approver = Agent(role="Final Approver", goal="Make FINAL decision (strict JSON)", llm=DEFAULT_MODEL, verbose=False)

    # Tasks
    t1 = Task(description=f"Review extraction and list obscure/missing fields.\n{summary}", expected_output="Bullet issues or 'No issues'.", agent=capture)
    t2 = Task(description=f"Given checks JSON, confirm duplicate/2-way/3-way/master_ok clearly as compact JSON.", expected_output='{"duplicate":..., "two_way":..., "three_way":..., "master_ok":...}', agent=validate)
    t3 = Task(description=f"Use policy/examples to identify risks.\n{context}", expected_output="Short risk list; or 'No additional risk'.", agent=risk)
    t4 = Task(description=f"Given proposed routing JSON, confirm approvers & SLAs; adjust if needed; return compact JSON.", expected_output='{"approvers":[...],"sla_days":N}', agent=router)
    t5 = Task(description=f"Given proposed GL JSON for a non-PO invoice, confirm/adjust and return compact JSON.", expected_output='{"gl":"...","cost_center":"...","project":"...","confidence":0..1}', agent=glcoder)
    t6 = Task(description=textwrap.dedent("""\
        You are the FINAL approver. Decide APPROVE/REJECT/REVIEW. Return STRICT JSON ONLY:
        {"action":"APPROVE|REJECT|REVIEW","rationale":"<1-3 concise sentences>"}
        Consider all prior outputs, checks, routing, GL coding, and risks.
    """), expected_output='Strict JSON with "action" and "rationale"', agent=approver)

    crew = Crew(agents=[capture, validate, risk, router, glcoder, approver], tasks=[t1,t2,t3,t4,t5,t6], process=Process.sequential, verbose=False)
    try:
        raw = str(crew.kickoff())
        decision = _parse_crewai_json(raw)
        return {"crewai": True, "decision": decision, "raw": raw}
    except Exception as e:
        return {"crewai": True, "decision": {"action":"REVIEW","rationale":f"CrewAI error: {e}"}, "raw": f"Error: {e}"}

# ==============================================================================
# Pro Agents (heuristics; LLM can refine inside crew)
# ==============================================================================
def gl_coding_agent(extraction: Dict[str,Any], vendor_row: Optional[pd.Series], gl_map: pd.DataFrame, threshold: float = 0.9) -> Dict[str,Any]:
    """Suggest GL, cost center, project for Non-PO invoices. Simple keyword scoring + vendor defaults."""
    if extraction.get("po_number"):
        return {"mode":"PO","gl":None,"cost_center":None,"project":None,"confidence":1.0,"explain":"PO invoice; GL from PO/GR posting rules"}
    desc = (extraction.get("description") or "") + " " + (vendor_row["category"] if vendor_row is not None else "")
    best = {"gl": vendor_row["default_gl"] if vendor_row is not None else "6100", "cost_center": "GNA", "project":"", "score":0.5, "keyword":"default"}
    for _, row in gl_map.iterrows():
        kw = row["keyword"]
        if re.search(kw, desc, re.I):
            if row["score"] > best["score"]:
                best = {"gl": row["gl"], "cost_center": row["cost_center"], "project": row["project"], "score": float(row["score"]), "keyword": kw}
    return {
        "mode":"Non-PO",
        "gl": best["gl"], "cost_center": best["cost_center"], "project": best["project"],
        "confidence": round(best["score"],2),
        "explain": f"Matched keyword '{best['keyword']}' in description; vendor default {vendor_row['default_gl'] if vendor_row is not None else 'n/a'}"
    }

def approver_routing_agent(extraction: Dict[str,Any], vendors_df: pd.DataFrame, pos_df: pd.DataFrame) -> Dict[str,Any]:
    """Pick approver(s) from PO owner for PO invoices; or cost center owner for non-PO."""
    approvers = []
    sla_days = 3
    if extraction.get("po_number"):
        pr = pos_df[pos_df.po_number == extraction["po_number"]]
        if not pr.empty:
            owner = pr.iloc[0].get("owner_email") or "owner@example.com"
            approvers.append(owner)
    else:
        # fallback: route to cost center owner inferred from GL coding stage (front-end will merge)
        approvers.append("costcenter.owner@example.com")
    if (extraction.get("total") or 0) >= 50000 and "marketing" in (extraction.get("description") or "").lower():
        approvers.append("finance.director@example.com")
        sla_days = 2
    return {"approvers": list(dict.fromkeys(approvers)), "sla_days": sla_days, "explain":"PO owner or CC owner; escalate if marketing >$50k"}

def anomaly_agent(extraction: Dict[str,Any], vendor_row: Optional[pd.Series]) -> Dict[str,Any]:
    """Simple anomaly scoring: amount z-score by vendor vs historical mean; flags bank change risks (stub)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT total FROM invoices WHERE vendor_name=? AND total IS NOT NULL", conn, params=[extraction.get("vendor_name")])
    except Exception:
        df = pd.DataFrame(columns=["total"])
    finally:
        conn.close()
    x = extraction.get("total") or 0.0
    if df.empty:
        score = 0.1
    else:
        mu = float(df["total"].mean()); sd = float(df["total"].std() or 1.0)
        z = abs((x - mu) / sd)
        score = float(np.tanh(z/3))  # 0..1
    reasons = []
    if score > 0.8:
        reasons.append("Amount outlier vs vendor history")
    # Stub: vendor bank change detection would require vendor bank master
    return {"score": round(score,2), "reasons": reasons or ["None"]}

# ==============================================================================
# End-to-end processing
# ==============================================================================
def process_invoice(file, tolerance: float, auto_post: bool, gl_autopilot: float, route_autopilot: bool):
    vendors_df, pos_df, receipts_df, gl_map = load_master_data()
    conn = sqlite3.connect(DB_PATH)

    raw_text = extract_text_from_pdf(file.getvalue())
    st.session_state["last_uploaded_text"] = raw_text
    extraction = parse_invoice_text(raw_text, vendors_df)
    st.session_state["last_extraction"] = extraction

    vendor_id, vendor_row = map_vendor_name_to_id(extraction.get("vendor_name"), vendors_df)
    extraction["vendor_id"] = vendor_id
    trusted = bool(vendor_row["trusted"]) if vendor_row is not None else False

    # Checks
    exceptions: List[str] = []
    checks = {"duplicate": False, "two_way": False, "three_way": False, "master_ok": bool(vendor_id)}
    inv_no = extraction.get("invoice_number")

    if vendor_id and inv_no:
        if duplicate_check(conn, vendor_id, inv_no):
            checks["duplicate"] = True
            exceptions.append("Duplicate invoice detected")
    else:
        exceptions.append("Missing vendor or invoice number")

    po_row = None; receipt_row = None
    if extraction.get("po_number"):
        pr = pos_df[pos_df.po_number == extraction["po_number"]]
        po_row = None if pr.empty else pr.iloc[0]
        rr = receipts_df[receipts_df.po_number == extraction["po_number"]]
        receipt_row = None if rr.empty else rr.iloc[0]

    if po_row is not None:
        checks["two_way"] = two_way_match(po_row, extraction.get("total"), tolerance)
    if po_row is not None and receipt_row is not None:
        checks["three_way"] = three_way_match(receipt_row, extraction.get("total"), po_row, tolerance)

    # RAG
    items = build_rag_corpus()
    upsert_rag_items(items)
    rag_query = f"vendor={extraction.get('vendor_name')}, total={extraction.get('total')}, checks={checks}, desc={(extraction.get('description') or '')[:80]}"
    rag_ctx = rag_search(rag_query, k=3)

    # Pro Agents
    gl_suggestion = gl_coding_agent(extraction, vendor_row, gl_map, threshold=gl_autopilot)
    route_suggestion = approver_routing_agent(extraction, vendors_df, pos_df)
    anomaly = anomaly_agent(extraction, vendor_row)

    # Exception agent (vendor email draft) — if duplicate/missing criticals
    maybe_email_id = None
    if checks["duplicate"] or (extraction.get("invoice_number") is None) or (extraction.get("po_number") is None and gl_suggestion["mode"]=="PO"):
        issue = "Duplicate invoice" if checks["duplicate"] else "Missing critical fields"
        draft = draft_vendor_email(issue, extraction, vendor_row)
        maybe_email_id = save_message("outbound", "draft", draft["to_email"], "ap-bot@demo.local", draft["subject"], draft["body"], None)

    # Final Approver via CrewAI (or fallback)
    crew = run_agents(extraction, checks, rag_ctx, gl_suggestion, route_suggestion, anomaly)
    rationale_raw = crew.get("raw")
    if crew.get("crewai") and crew.get("decision"):
        decision = crew["decision"]
    else:
        pol = open(os.path.join(DATA_DIR,"policies.md"),"r",encoding="utf-8").read()
        ex  = open(os.path.join(DATA_DIR,"examples.md"),"r",encoding="utf-8").read()
        decision = llm_decide(
            f"trusted={trusted}, checks={checks}, total={extraction.get('total')}, vendor={extraction.get('vendor_name')}, gl={gl_suggestion}, route={route_suggestion}, anomaly={anomaly}",
            pol + "\n\n" + ex,
            model=DEFAULT_MODEL
        )

    # Persist
    inv_id = str(uuid.uuid4())
    stp = 1 if (decision["action"] == "APPROVE" and not exceptions) else 0

    record = {
        "id": inv_id,
        "created_at": datetime.utcnow().isoformat(),
        "vendor_id": vendor_id,
        "vendor_name": extraction.get("vendor_name"),
        "invoice_number": extraction.get("invoice_number"),
        "po_number": extraction.get("po_number"),
        "invoice_date": extraction.get("invoice_date"),
        "currency": extraction.get("currency"),
        "subtotal": extraction.get("subtotal"),
        "tax": extraction.get("tax"),
        "total": extraction.get("total"),
        "status": decision["action"],
        "stp": stp,
        "exceptions_json": json.dumps(exceptions),
        "decisions_json": json.dumps({"decision": decision, "checks": checks, "rag": rag_ctx}),
        "rationale": decision.get("rationale"),
        "crewai_rationale": rationale_raw,
        "gl_json": json.dumps(gl_suggestion),
        "route_json": json.dumps(route_suggestion),
        "anomaly_json": json.dumps(anomaly),
    }

    c = conn.cursor()
    c.execute(f"INSERT INTO invoices ({','.join(record.keys())}) VALUES ({','.join(['?']*len(record))})", tuple(record.values()))

    if maybe_email_id:
        c.execute("UPDATE messages SET related_invoice_id=? WHERE id=?", (inv_id, maybe_email_id))

    # Autopilot: post to ERP if approved
    if auto_post and decision["action"] == "APPROVE":
        c.execute(
            "INSERT OR REPLACE INTO posted_invoices (id, posted_at, erp_doc_id, amount, currency) VALUES (?,?,?,?,?)",
            (inv_id, datetime.utcnow().isoformat(), f"ERP-{extraction.get('invoice_number') or inv_id[:8]}", record["total"] or 0, record["currency"]),
        )
    conn.commit(); conn.close()

    # UI
    st.success(f"Final decision: {decision['action']}")
    c1, c2 = st.columns(2)
    with c1.expander("⚖️ Decision", expanded=True): st.json(decision)
    with c2.expander("📚 RAG Context"): st.json(rag_ctx)
    with st.expander("📋 Extraction"): st.json(extraction)
    with st.expander("🧪 Checks"): st.json(checks)
    with st.expander("🧾 GL Coding"): st.json(gl_suggestion)
    with st.expander("🧭 Routing (Approvers)"): st.json(route_suggestion)
    with st.expander("🕵️ Anomaly"): st.json(anomaly)
    if maybe_email_id:
        st.warning("Vendor email drafted in Outbox (draft). Review in Inbox/Outbox tab.")

# ==============================================================================
# Queues & Views (HITL)
# ==============================================================================
def render_dashboard():
    conn = sqlite3.connect(DB_PATH)
    try:
        try:
            df = pd.read_sql_query("SELECT * FROM invoices ORDER BY created_at_ts DESC", conn)
        except Exception:
            df = pd.read_sql_query("SELECT * FROM invoices ORDER BY created_at DESC", conn)
    finally:
        conn.close()
    if df.empty:
        st.info("No invoices yet.")
        return

    stp_rate = 100.0 * df["stp"].mean()
    approval_rate = 100.0 * (df["status"].str.upper() == "APPROVE").mean()
    exception_rate = 100.0 * df["exceptions_json"].apply(lambda x: 0 if x == "[]" else 1).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("STP %", f"{stp_rate:.0f}%")
    c2.metric("Approval %", f"{approval_rate:.0f}%")
    c3.metric("Exception %", f"{exception_rate:.0f}%")

    df["decision_rationale"] = df["decisions_json"].apply(lambda x: (json.loads(x).get("decision",{}) if x else {}).get("rationale"))
    show = df[["created_at","vendor_name","invoice_number","po_number","total","currency","status","stp","decision_rationale"]]
    st.subheader("Recent Invoices")
    st.dataframe(show, use_container_width=True)

def render_exception_queue():
    st.subheader("🚧 Exception Review Queue")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM invoices ORDER BY created_at DESC", conn)
    conn.close()
    if df.empty:
        st.info("No invoices.")
        return
    # Filter likely exceptions
    df["has_exception"] = df["exceptions_json"].apply(lambda x: x and x != "[]")
    q = df[(df["status"].str.upper()=="REVIEW") | (df["has_exception"]==True)]
    if q.empty:
        st.success("No exceptions pending — nice!")
        return
    st.dataframe(q[["created_at","vendor_name","invoice_number","po_number","total","status","exceptions_json","gl_json","route_json"]], use_container_width=True)

def render_approver_queue():
    st.subheader("🧑‍💼 Approver Queue (simulated)")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM invoices WHERE status='REVIEW' ORDER BY created_at DESC", conn)
    conn.close()
    if df.empty:
        st.info("No items waiting on approvers.")
        return
    st.dataframe(df[["created_at","vendor_name","invoice_number","po_number","total","route_json"]], use_container_width=True)

def render_db_views():
    conn = sqlite3.connect(DB_PATH)
    inv = pd.read_sql_query("SELECT * FROM invoices ORDER BY created_at DESC", conn)
    posted = pd.read_sql_query("SELECT * FROM posted_invoices ORDER BY posted_at DESC", conn)
    msgs = pd.read_sql_query("SELECT * FROM messages ORDER BY created_at DESC", conn)
    conn.close()
    st.markdown("### Invoices (raw)")
    st.dataframe(inv, use_container_width=True)
    st.markdown("### Posted Invoices (ERP Simulation)")
    st.dataframe(posted, use_container_width=True)
    st.markdown("### Messages (Inbox/Outbox)")
    st.dataframe(msgs, use_container_width=True)

def render_policies_and_data():
    vendors_df, pos_df, receipts_df, gl_map = load_master_data()
    st.markdown("### Policies")
    st.code(open(os.path.join(DATA_DIR,"policies.md"),"r",encoding="utf-8").read())
    st.markdown("### Examples (few-shot)")
    st.code(open(os.path.join(DATA_DIR,"examples.md"),"r",encoding="utf-8").read())
    st.markdown("### Vendor Master")
    st.dataframe(vendors_df, use_container_width=True)
    st.markdown("### Purchase Orders")
    st.dataframe(pos_df, use_container_width=True)
    st.markdown("### Goods Receipts")
    st.dataframe(receipts_df, use_container_width=True)
    st.markdown("### GL Keyword Map")
    st.dataframe(gl_map, use_container_width=True)

def render_inbox_outbox():
    st.subheader("📨 Inbox / Outbox (Simulated)")
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()

    st.markdown("#### Outbox")
    out_df = pd.read_sql_query("SELECT * FROM messages WHERE direction='outbound' ORDER BY created_at DESC", conn)
    st.dataframe(out_df, use_container_width=True)
    draft_ids = out_df[out_df["status"]=="draft"]["id"].tolist()
    if draft_ids:
        sel = st.selectbox("Select a draft to send", options=draft_ids, key="send_draft_sel")
        if st.button("Send selected draft"):
            c.execute("UPDATE messages SET status='sent' WHERE id=?", (sel,))
            conn.commit(); st.success("Draft sent.")

    st.divider()
    st.markdown("#### Simulate Vendor Reply (Inbound)")
    rel = st.text_input("Related Invoice ID (optional)", value="")
    from_email = st.text_input("From (vendor email)", value="vendor@example.com")
    subject = st.text_input("Subject", value="Re: Invoice clarification")
    body = st.text_area("Message body", value="Here is the missing PO number / corrected total ...")
    if st.button("Add inbound message"):
        save_message("inbound","received","ap-bot@demo.local",from_email,subject,body,rel or None)
        st.success("Inbound message saved.")

    st.divider()
    st.markdown("#### Inbox")
    in_df = pd.read_sql_query("SELECT * FROM messages WHERE direction='inbound' ORDER BY created_at DESC", conn)
    st.dataframe(in_df, use_container_width=True)
    if not in_df.empty:
        msg_id = st.selectbox("Pick an inbound message", options=[""] + in_df["id"].tolist())
        if msg_id:
            row = in_df[in_df["id"]==msg_id].iloc[0]
            st.code(row["body"])
            # Future: auto-parse corrections & re-run decision
            if st.button("Mark handled"):
                c.execute("UPDATE messages SET status='handled' WHERE id=?", (msg_id,))
                conn.commit(); st.success("Message marked handled.")
    conn.close()

# ==============================================================================
# Streamlit App
# ==============================================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")
    ensure_dirs(); seed_csvs_if_missing(); init_db()

    st.title(APP_TITLE)
    st.caption("Agentic multi-agent • RAG • GL coding • Routing • Fraud • HITL queues • Weekend MVP")

    with st.sidebar:
        st.header("⚙️ Control Center")
        env = os.getenv("ENV","development").capitalize()
        erp = os.getenv("ERP_SYSTEM","Simulated ERP")
        st.markdown(f"**Environment:** `{env}`")
        st.markdown(f"**ERP System:** `{erp}`")
        st.markdown(f"**LLM Model:** `{DEFAULT_MODEL}`")
        st.markdown("### Status")
        st.write("LLM configured:", "✅" if USE_LLM else "❌")
        st.write("CrewAI enabled:", "✅" if (USE_CREW and USE_LLM) else "❌")
        st.write("Qdrant configured:", "✅" if USE_QDRANT else "❌")
        st.divider()
        auto_post = st.checkbox("Auto-post approved invoices (ERP simulated)", value=True)
        tolerance = st.slider("Match tolerance (±)", 0.0, 0.1, 0.02, 0.005)
        gl_autopilot = st.slider("GL autopilot threshold", 0.5, 0.99, 0.90, 0.01)
        route_autopilot = st.checkbox("Auto-route to approver(s)", value=True)
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; font-size: 0.85em;'>"
            "🧾 <b>Agentic AP Automation</b><br>"
            "v0.3.0 • Built by <b>Mohanraja Sivakumar</b>"
            "</div>",
            unsafe_allow_html=True
        )

    tabs = st.tabs([
        "📤 Upload & Process",
        "📬 Inbox / Outbox",
        "📊 Dashboard",
        "🚦 Exception Queue",
        "✅ Approver Queue",
        "🪵 Logs / DB",
        "📚 Policies & Data",
        "🔎 Debug"
    ])

    with tabs[0]:
        st.subheader("Upload Invoice (PDF)")
        file = st.file_uploader("Drop a PDF invoice", type=["pdf"])
        if file is not None and st.button("Process Invoice", type="primary"):
            process_invoice(file, tolerance, auto_post, gl_autopilot, route_autopilot)

    with tabs[1]: render_inbox_outbox()
    with tabs[2]: render_dashboard()
    with tabs[3]: render_exception_queue()
    with tabs[4]: render_approver_queue()
    with tabs[5]: render_db_views()
    with tabs[6]: render_policies_and_data()

    with tabs[7]:
        st.subheader("Debug")
        if "last_uploaded_text" in st.session_state:
            st.text_area("Raw Extracted Text", st.session_state["last_uploaded_text"], height=260)
        if "last_extraction" in st.session_state:
            st.json(st.session_state["last_extraction"])

if __name__ == "__main__":
    main()
