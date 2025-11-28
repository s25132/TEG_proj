import os
import requests
import streamlit as st

# URL backendu FastAPI
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def ask_backend(question: str, top_k: int = 5):
    """Wysyła pytanie do backendu /ask_rag i zwraca odpowiedź + kontekst."""
    url = f"{BACKEND_URL}/ask_rag"
    payload = {
        "question": question,
        "top_k": top_k,
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Oczekiwany kształt: {"answer": "...", "context_documents": [...]}
    return data["answer"], data.get("context_documents", [])


# --- UI ---

st.set_page_config(page_title="Talent AI", page_icon="💬", layout="centered")
st.title("Talent AI")

# Pola konfiguracyjne
with st.sidebar:
    st.header("Ustawienia")
    backend = st.text_input("Backend URL", value=BACKEND_URL)
    top_k = st.slider("Liczba dokumentów (top_k)", 1, 10, 5)

# Aktualizujemy BACKEND_URL, jeśli user zmieni w sidebarze
BACKEND_URL = backend

# Inicjalizacja historii czatu
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # lista dictów: {"role": "user"/"assistant", "content": str}

# Wyświetl historię
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Pole wejściowe czatu
user_input = st.chat_input("Zadaj pytanie o projekty lub CV...")

if user_input:
    # 1. Dodaj wiadomość użytkownika do historii
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Wyślij do backendu
    try:
        with st.chat_message("assistant"):
            with st.spinner("Myślę..."):
                answer, context_docs = ask_backend(user_input, top_k=top_k)
                st.markdown(answer)

                # Opcjonalnie pokaż kontekst
                if context_docs:
                    with st.expander("Pokaż użyty kontekst (dokumenty z Chroma)"):
                        for i, doc in enumerate(context_docs, start=1):
                            st.markdown(f"**Dokument {i}:**")
                            st.write(doc)
                            st.markdown("---")

        # 3. Zapisz odpowiedź asystenta w historii
        st.session_state["messages"].append({"role": "assistant", "content": answer})

    except requests.RequestException as e:
        error_msg = f"Błąd komunikacji z backendem: {e}"
        st.error(error_msg)
        st.session_state["messages"].append({"role": "assistant", "content": error_msg})
