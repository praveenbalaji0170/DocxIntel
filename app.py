
import os, re, json, time, math
from datetime import datetime
import pytz
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import fitz
from pdf2image import convert_from_path
import pytesseract
from docx import Document
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.decomposition import PCA
import torch

# -------------------------
# Configuration & State
# -------------------------
STORE_DIR = Path("/content/reasonedai_store")
STORE_DIR.mkdir(exist_ok=True)
CHUNKS_FILE = STORE_DIR / "chunks.json"
EMBED_FILE = STORE_DIR / "embeddings.npy"
META_FILE = STORE_DIR / "files_meta.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "stepfun/step-3.5-flash"

# Load Models Locally
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    # Base retrieval model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Advanced Intent Classification Model
    intent_model = SentenceTransformer("all-mpnet-base-v2")

    INTENT_LABELS = {
        "INFORMATIONAL": "Query asking about definitions, policies, documentation or explanations.",
        "ANALYTICAL": "Query asking to analyze logs, compare metrics, check compliance or detect deviations.",
        "ADVISORY": "Query asking for recommendations, corrective actions, or what should be done."
    }

    label_texts = list(INTENT_LABELS.values())
    label_keys = list(INTENT_LABELS.keys())
    label_embeddings = intent_model.encode(label_texts, convert_to_tensor=True)

    return nlp, embed_model, intent_model, label_keys, label_embeddings

nlp, embed_model, intent_model, intent_keys, intent_embeddings = load_models()

# Global States
if "CHUNKS" not in st.session_state: st.session_state.CHUNKS = []
if "files_meta" not in st.session_state: st.session_state.files_meta = {}
if "dfs" not in st.session_state: st.session_state.dfs = {}
if "VECTOR_INDEX" not in st.session_state: st.session_state.VECTOR_INDEX = None
if "EMBEDDINGS" not in st.session_state: st.session_state.EMBEDDINGS = None
if "chats" not in st.session_state: st.session_state.chats = []
if "auto_prompt_trigger" not in st.session_state: st.session_state.auto_prompt_trigger = []

# -------------------------
# Intent & LLM Helpers
# -------------------------
def classify_intent_semantic(query: str) -> str:
    try:
        query_embedding = intent_model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, intent_embeddings)
        best_index = torch.argmax(cosine_scores).item()
        return intent_keys[best_index]
    except Exception as e:
        print(f"Classification error: {e}")
        return "INFORMATIONAL"

def call_openrouter_chat(system_prompt: str, user_prompt: str, history=None):
    headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
    messages = [{"role": "system", "content": system_prompt}]

    # Inject memory chain (last 4 interactions to avoid token bloat)
    if history:
        for chat in history[-4:]:
            messages.append({"role": chat["role"], "content": chat["content"]})

    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": 1500
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# -------------------------
# Document Rules & DB Update
# -------------------------
def determine_doc_type(fname: str) -> str:
    lower_name = fname.lower()
    if "sop" in lower_name: return "Standard_Operating_Procedure"
    if "maintenance" in lower_name: return "Maintenance_Manual"
    if "fault" in lower_name: return "Historical_Fault_Log"
    if "rule" in lower_name: return "Equipment_Rulebook"
    if "log" in lower_name: return "Continuous_Data_Log"
    return "General_Document"

def infer_month(fname: str) -> str:
    lower_name = fname.lower()
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    for m in months:
        if m in lower_name: return m.capitalize()
    return "Global_Context"

def rebuild_vector_index():
    if not st.session_state.CHUNKS:
        st.session_state.VECTOR_INDEX = None
        st.session_state.EMBEDDINGS = None
        return
    texts = [c["text"] for c in st.session_state.CHUNKS]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if len(embeddings.shape) == 1: embeddings = embeddings.reshape(1, -1)

    st.session_state.EMBEDDINGS = np.array(embeddings).astype('float32')
    dim = st.session_state.EMBEDDINGS.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(st.session_state.EMBEDDINGS)
    st.session_state.VECTOR_INDEX = idx

def semantic_search(query: str, top_k:int=15):
    if st.session_state.VECTOR_INDEX is None: return []
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype('float32')
    D, I = st.session_state.VECTOR_INDEX.search(q_emb, k=top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(st.session_state.CHUNKS): continue
        results.append(st.session_state.CHUNKS[idx])
    return results

# -------------------------
# Dynamic Rule Engine Logic
# -------------------------
def extract_metric(df, parameter):
    param_clean = str(parameter).strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == param_clean:
            numeric_series = pd.to_numeric(df[col], errors='coerce').dropna()
            if not numeric_series.empty: return numeric_series.mean()
    return None

def evaluate_condition(value, limit, condition_type):
    try:
        val, lim = float(value), float(limit)
        if condition_type == "Greater Than": return val > lim
        elif condition_type == "Less Than": return val < lim
        elif condition_type == "Equal To": return val == lim
    except: pass
    return False

def run_dynamic_rule_engine(df, rulebook_df):
    results = []
    for _, rule in rulebook_df.iterrows():
        try:
            parameter = rule.get("Parameter")
            limit = rule.get("Limit_Value")
            condition_type = rule.get("Condition_Type")

            value = extract_metric(df, parameter)
            if value is None: continue

            violation = evaluate_condition(value, limit, condition_type)
            results.append({
                "Parameter": parameter,
                "Measured_Value": round(float(value), 2),
                "Limit": limit,
                "Condition": condition_type,
                "Violation": violation,
                "Escalation": rule.get("Escalation_Action", "Alert") if violation else "None"
            })
        except: continue
    return results

def get_ist_time():
    ist_tz = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist_tz).strftime("%I:%M %p")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide", page_title="ReasonedAI Core POC")

st.markdown("""
<style>
    .reasoning-box {
        color: #a0a0a0; font-size: 0.85em; border-left: 3px solid #666;
        padding-left: 15px; margin-bottom: 10px; margin-top: 10px; background-color: rgba(255,255,255,0.03);
        padding-top: 8px; padding-bottom: 8px; border-radius: 0px 5px 5px 0px; font-style: italic;
    }
    .kpi-card { background-color: #1e1e2e; padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid #ff4b4b;}
    .kpi-safe { color: #00fa9a; font-weight: bold; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

left_col, mid_col, right_col = st.columns([1.5, 2.8, 1.8])

# ================= LEFT PANEL =================
with left_col:
    st.title("📂 System Memory")
    uploaded_files = st.file_uploader("Upload Manuals, SOPs, Logs", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Extracting & Embedding into Vector DB..."):
            new_files_added = []
            for up in uploaded_files:
                fname = up.name
                if fname in st.session_state.files_meta: continue

                doc_type = determine_doc_type(fname)
                month_context = infer_month(fname)

                raw = ""
                is_tabular = False

                if fname.endswith((".csv", ".xlsx")):
                    is_tabular = True
                    df = pd.read_csv(up) if fname.endswith(".csv") else pd.read_excel(up)
                    st.session_state.dfs[fname] = df

                    row_strings = []
                    for index, row in df.iterrows():
                        row_repr = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                        row_strings.append(f"[File: {fname} | Context: {month_context}] Row {index+1}: {row_repr}")
                    raw = "\n".join(row_strings)
                elif fname.endswith(".pdf"):
                    doc = fitz.open(stream=up.read(), filetype="pdf")
                    raw = "\n".join([page.get_text() for page in doc])
                    raw = re.sub(r'\n\s*\n+', '\n\n', raw)
                else:
                    raw = up.getvalue().decode("utf-8")

                start_id = len(st.session_state.CHUNKS)
                parts = []
                if is_tabular:
                    lines = raw.split('\n')
                    for i in range(0, len(lines), 15): parts.append("\n".join(lines[i:i+15]))
                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    parts = splitter.split_text(raw)

                for i, p in enumerate(parts):
                    st.session_state.CHUNKS.append({
                        "id": start_id + i, "source": fname, "text": p,
                        "meta": {"document_type": doc_type, "month": month_context}
                    })

                local_time = get_ist_time()
                st.session_state.files_meta[fname] = {"type": doc_type, "chunks": len(parts), "uploaded": local_time}
                new_files_added.append(fname)

            if new_files_added:
                rebuild_vector_index()
                st.session_state.auto_prompt_trigger = new_files_added
                st.rerun()

    st.markdown("### 🗄️ Active Documents")
    if st.session_state.files_meta:
        for fn, info in st.session_state.files_meta.items():
            col1, col2 = st.columns([5, 1])
            col1.caption(f"📄 **{fn}**\n\n*(Added at {info['uploaded']} IST)*")
            if col2.button("❌", key=f"del_{fn}"):
                st.session_state.CHUNKS = [c for c in st.session_state.CHUNKS if c["source"] != fn]
                if fn in st.session_state.dfs: del st.session_state.dfs[fn]
                del st.session_state.files_meta[fn]
                rebuild_vector_index()
                st.rerun()
    else:
        st.caption("Memory is empty. Upload documents to begin.")

    st.markdown("---")
    with st.expander("🌌 View 3D Vector Space", expanded=False):
        if st.session_state.EMBEDDINGS is not None and len(st.session_state.EMBEDDINGS) > 3:
            pca = PCA(n_components=3)
            reduced = pca.fit_transform(st.session_state.EMBEDDINGS)
            df_pca = pd.DataFrame({
                'Dim 1': reduced[:,0], 'Dim 2': reduced[:,1], 'Dim 3': reduced[:,2],
                'Type': [c['meta']['document_type'] for c in st.session_state.CHUNKS],
                'Source': [c['source'] for c in st.session_state.CHUNKS]
            })
            fig_pca = px.scatter_3d(df_pca, x='Dim 1', y='Dim 2', z='Dim 3', color='Type', hover_data=['Source'])
            fig_pca.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""))
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.warning("Not enough data to visualize space.")

# ================= MIDDLE PANEL =================
with mid_col:
    st.title("🧠 Reasoning Engine")

    # Auto-Prompt Logic
    if st.session_state.auto_prompt_trigger:
        new_files = ", ".join(st.session_state.auto_prompt_trigger)
        st.session_state.auto_prompt_trigger = []

        auto_query = f"I just uploaded {new_files}. Provide a proactive executive summary identifying critical trends and comparing to existing rules/data."
        st.session_state.chats.append({"role": "user", "content": f"*(System Event)*: User uploaded new files: {new_files}. Analyze them against the database."})

        with st.status("🔄 Generating Automated Context Summary...", expanded=True) as status:
            retrieved = semantic_search(f"Summary and trends for {new_files}", top_k=15)
            context_str = "\n".join([f"Source: {r['source']}\n{r['text']}\n" for r in retrieved])

            sys_prompt = f"""You are an Elite Industrial AI Analyst.
            FORMATTING RULES:
            <REASONING>
            Provide a deep, highly intelligent chain of thought. Explain exactly how you are cross-referencing these new files with existing rules or previous months to find trends.
            </REASONING>
            <ANSWER>
            Start your answer EXACTLY with: "✅ **Vector Database Updated:** The new files have been successfully embedded and indexed."
            Provide your insightful summary below that.
            Do NOT use asterisks (*) or HTML tags (like <sup>) for citations. Simply write at the end of the claim.
            </ANSWER>
            """
            try:
                llm_answer = call_openrouter_chat(sys_prompt, f"Task: {auto_query}\n\nContext:\n{context_str}")
                st.session_state.chats.append({"role": "assistant", "content": llm_answer})
                status.update(label="Auto-Summary Complete", state="complete", expanded=False)
                st.rerun()
            except:
                status.update(label="Auto-Summary Failed", state="error")

    # Lock Chat in a Scrollable Container
    chat_container = st.container(height=600, border=True)
    with chat_container:
        for chat in st.session_state.chats:
            if chat["role"] == "user":
                st.chat_message("user").write(chat["content"])
            else:
                with st.chat_message("assistant"):
                    ans = chat["content"]
                    r_match = re.search(r'<REASONING>(.*?)</REASONING>', ans, re.DOTALL | re.IGNORECASE)
                    a_match = re.search(r'<ANSWER>(.*?)</ANSWER>', ans, re.DOTALL | re.IGNORECASE)

                    if r_match:
                        st.markdown(f"<div class='reasoning-box'><b>⚙️ Deep Chain of Thought:</b><br>{r_match.group(1).strip()}</div>", unsafe_allow_html=True)

                    a_text = a_match.group(1).strip() if a_match else ans
                    # Strip weird artifacts
                    a_text = a_text.replace('<sup>', '').replace('</sup>', '')
                    st.markdown(a_text)

    # User Input Chat
    query = st.chat_input("Ask about analytics, files, or rule limits...")
    if query:
        st.session_state.chats.append({"role": "user", "content": query})
        st.rerun()

    # Processing state for the latest query
    if st.session_state.chats and st.session_state.chats[-1]["role"] == "user" and not st.session_state.chats[-1]["content"].startswith("*(System Event)"):
        latest_query = st.session_state.chats[-1]["content"]
        with st.status("Processing AI Pipeline...", expanded=True) as status:
            t0 = time.time()
            intent = classify_intent_semantic(latest_query)
            st.write(f"✅ Semantic Intent: **{intent}**")

            retrieved = semantic_search(latest_query, top_k=15)
            context_str = "\n".join([f"Source: {r['source']}\n{r['text']}\n" for r in retrieved])

            sys_prompt = f"""You are the ReasonedAI Industrial Agent. Intent: {intent}.

            FORMATTING RULES:
            <REASONING>
            Provide a detailed, intelligent multi-step chain of thought. Explain what files you checked, how you compared data, and the logic of your deduction.
            </REASONING>
            <ANSWER>
            Provide the final answer entirely outside the reasoning block.
            Highlight key metrics using **bold**.
            Do NOT use asterisks (*) or HTML tags (like <sup>) for citations. Simply write perfectly at the end of the claim.
            </ANSWER>
            """
            try:
                # Passing history to maintain conversational memory
                llm_answer = call_openrouter_chat(sys_prompt, f"Query: {latest_query}\n\nContext:\n{context_str}", history=st.session_state.chats)
                st.session_state.chats.append({"role": "assistant", "content": llm_answer})
                status.update(label="Response Generated", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                st.error(f"API Error: {e}")

# ================= RIGHT PANEL: Aggregated Analytics =================
with right_col:
    st.title("📈 Global Analytics")

    # Extract Master Log Data for Visualizations
    master_log_list = []
    for fname, df in st.session_state.dfs.items():
        if "log" in fname.lower() or "fault" in fname.lower():
            temp = df.copy()
            temp['Source_Log'] = fname
            # Standardize Timestamp column
            if 'Date' in temp.columns: temp['Parsed_Time'] = pd.to_datetime(temp['Date'], errors='coerce')
            elif 'Timestamp' in temp.columns: temp['Parsed_Time'] = pd.to_datetime(temp['Timestamp'], errors='coerce')
            else: temp['Parsed_Time'] = pd.NaT
            master_log_list.append(temp)

    master_log = pd.concat(master_log_list, ignore_index=True) if master_log_list else pd.DataFrame()

    # 1. KPI Violations (Absolute Highest Priority)
    rule_df = next((df for fn, df in st.session_state.dfs.items() if "rule" in fn.lower()), None)

    if rule_df is not None and not master_log.empty:
        st.markdown("### 🚨 Critical KPI Violations")
        results = run_dynamic_rule_engine(master_log, rule_df)
        violations = [r for r in results if r["Violation"]]
        if violations:
            for v in violations:
                st.markdown(f"<div class='kpi-card'>⚠️ <b>{v['Parameter']}</b>: {v['Measured_Value']} (Limit: {v['Limit']})<br><i>Action: {v['Escalation']}</i></div>", unsafe_allow_html=True)
        else:
            st.success("All metrics are currently within Rulebook limits.")

    # 2. Recurrence / Top Faults
    if not master_log.empty and 'Component' in master_log.columns:
        st.markdown("### 🔧 Top Recurring Faults")
        fault_counts = master_log["Component"].value_counts().reset_index().head(5)
        fig_bar = px.bar(fault_counts, x="Component", y="count", color="count", color_continuous_scale="Reds")
        fig_bar.update_layout(margin=dict(l=0, r=0, b=0, t=10), height=200)
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Aggregated Parameter Trends (Scalable & Combined)
    if not master_log.empty and 'Parsed_Time' in master_log.columns:
        numeric_cols = master_log.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            st.markdown("### 📉 Multi-Log Parameter Trends")
            param_to_plot = st.selectbox("Track Parameter Across All Logs:", numeric_cols)

            plot_df = master_log.dropna(subset=['Parsed_Time', param_to_plot]).sort_values('Parsed_Time')
            if not plot_df.empty:
                fig_line = px.line(plot_df, x='Parsed_Time', y=param_to_plot, color='Source_Log', markers=True)

                # Draw Peak and Average Lines
                avg_val = plot_df[param_to_plot].mean()
                peak_val = plot_df[param_to_plot].max()
                fig_line.add_hline(y=avg_val, line_dash="dash", line_color="green", annotation_text=f"Avg: {avg_val:.2f}")
                fig_line.add_hline(y=peak_val, line_dash="dot", line_color="red", annotation_text=f"Peak: {peak_val:.2f}")

                fig_line.update_layout(margin=dict(l=0, r=0, b=0, t=10), height=280)
                st.plotly_chart(fig_line, use_container_width=True)

    # 4. Reports & Blueprints
    st.markdown("---")
    st.markdown("### 📋 Executive Reports")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("📄 Generate Audit Report"):
            with st.spinner("Compiling global audit..."):
                if st.session_state.VECTOR_INDEX:
                    context_str = "\n".join([c['text'] for c in st.session_state.CHUNKS[:20]])
                    audit_prompt = f"""Based on ALL provided context: {context_str}
                    Write a formal, Markdown-formatted Auditing Report. Include:
                    # Executive Summary
                    # Document Sources Indexed
                    # Critical Faults & KPI Violations Detected
                    # Operational Recommendations
                    Use bolding appropriately. Ensure clean formatting.
                    """
                    try:
                        res = call_openrouter_chat("Output clean Markdown.", audit_prompt)
                        st.download_button("📥 Download Report", res, "audit_report.md", "text/markdown")
                        st.success("Report Ready!")
                    except: st.error("Rate limit hit.")

    with c_btn2:
        if st.button("🤖 Agentic Blueprint"):
            with st.spinner("Generating architecture..."):
                if st.session_state.VECTOR_INDEX:
                    context_str = "\n".join([c['text'] for c in st.session_state.CHUNKS[:20]])
                    bp_prompt = f"""Based on these manual documents: {context_str}
                    Create a structured JSON blueprint for an Agentic AI Framework.
                    """
                    try:
                        res = call_openrouter_chat("Output ONLY valid JSON.", bp_prompt)
                        res_clean = res.replace("```json", "").replace("```", "").strip()
                        st.download_button("📥 Download JSON", res_clean, "blueprint.json", "application/json")
                        st.success("Blueprint Ready!")
                    except: st.error("Rate limit hit.")

    # Token/Chunk Stats moved to the bottom
    st.markdown("---")
    st.caption(f"⚙️ Memory Stats: {len(st.session_state.CHUNKS)} Chunks Indexed | ~{math.ceil(sum(len(c['text'].split()) for c in st.session_state.CHUNKS) / 0.75)} Tokens")
