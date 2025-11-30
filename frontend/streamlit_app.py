import os
import requests
import streamlit as st

# URL backendu FastAPI
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def ask_backend(question: str, top_k: int = 5):
    """Wysyła pytanie do backendu /ask_rag i zwraca odpowiedź + kontekst."""
    url = f"{BACKEND_URL}/ask_rag"
    payload = {"question": question, "top_k": top_k}

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data.get("context_documents", [])


def ask_backend_graph(question: str, top_k: int = 5):
    """Wysyła pytanie do backendu /ask_graph (Graph RAG) i zwraca odpowiedź + kontekst grafowy."""
    url = f"{BACKEND_URL}/ask_graph"
    payload = {"question": question, "top_k": top_k}

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data.get("context_subgraphs", [])


def upload_rfp(file) -> str:
    """Wysyła PDF do backendu /add_rfp."""
    url = f"{BACKEND_URL}/add_rfp"
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    resp = requests.post(url, files=files, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("status", "UNKNOWN")


# --- UI ---

st.set_page_config(page_title="Talent AI", page_icon="💬", layout="centered")
st.title("Talent AI")

# Sidebar
with st.sidebar:
    st.header("Ustawienia")
    backend = st.text_input("Backend URL", value=BACKEND_URL)

BACKEND_URL = backend

# --- Kontrola aktywnej zakładki ---

TAB_OPTIONS = ["chat", "graph", "rfp"]  # → RFP jako ostatnia
tab_choice = st.session_state.get("active_tab", "chat")

selected_tab = st.radio(
    "Wybierz zakładkę:",
    ["💬 Chat", "🕸️ Graph", "📄 Dodaj RFP (PDF)"],  # kolejność zgodna z wymaganiem
    horizontal=True
)

label_to_key = {
    "💬 Chat": "chat",
    "🕸️ Graph": "graph",
    "📄 Dodaj RFP (PDF)": "rfp",
}

current_tab_key = label_to_key[selected_tab]

# Reset historii tylko przy zmianie zakładki
if current_tab_key != tab_choice:
    st.session_state["messages"] = []
    st.session_state["active_tab"] = current_tab_key

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# --- Zakładka Chat (tekstowy RAG) ---

if current_tab_key == "chat":
    st.header("💬 Chat")

    top_k = st.slider("Liczba dokumentów (top_k)", 1, 10, 5)

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Zadaj pytanie o projekty lub CV...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Myślę..."):
                    answer, context_docs = ask_backend(user_input, top_k=top_k)
                    st.markdown(answer)

                    if context_docs:
                        with st.expander("Pokaż użyty kontekst (dokumenty z Chroma)"):
                            for i, doc in enumerate(context_docs, start=1):
                                st.markdown(f"**Dokument {i}:**")
                                st.write(doc)
                                st.markdown("---")

            st.session_state["messages"].append({"role": "assistant", "content": answer})

        except requests.RequestException as e:
            err = f"Błąd komunikacji z backendem: {e}"
            st.error(err)
            st.session_state["messages"].append({"role": "assistant", "content": err})


# --- Zakładka Graph (Graph RAG) ---

if current_tab_key == "graph":
    st.header("🕸️ Graph RAG")

    top_k = st.slider("Liczba elementów grafu (top_k)", 1, 10, 5)

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Zadaj pytanie do grafowej bazy wiedzy...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Analizuję graf..."):
                    answer, context_subgraphs = ask_backend_graph(user_input, top_k=top_k)
                    st.markdown(answer)

                    if context_subgraphs:
                        with st.expander("Pokaż użyty kontekst grafowy (podgrafy / ścieżki)"):
                            for i, sg in enumerate(context_subgraphs, start=1):
                                st.markdown(f"**Podgraf {i}:**")
                                st.write(sg)
                                st.markdown("---")

            st.session_state["messages"].append({"role": "assistant", "content": answer})

        except requests.RequestException as e:
            err = f"Błąd komunikacji z backendem (Graph): {e}"
            st.error(err)
            st.session_state["messages"].append({"role": "assistant", "content": err})


# --- Zakładka RFP (ostatnia) ---

if current_tab_key == "rfp":
    st.header("📄 Dodaj nowe RFP (PDF)")

    uploaded_file = st.file_uploader("Wybierz plik PDF:", type=["pdf"])

    if uploaded_file is not None:
        st.write(f"Wybrano: **{uploaded_file.name}**")

        if st.button("Wyślij do backendu"):
            try:
                with st.spinner("Wysyłam plik..."):
                    status = upload_rfp(uploaded_file)
                st.success(f"Status backendu: {status}")
            except requests.RequestException as e:
                st.error(f"❌ Błąd podczas wysyłania: {e}")
