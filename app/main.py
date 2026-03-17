"""
IKEA RAG Chat — Streamlit conversational interface.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import List

import streamlit as st
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import SecretStr

# ── Make sure the project root is on sys.path ─────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import Settings  # noqa: E402
from src.embeddings.aws_embeddings import get_embeddings  # noqa: E402
from src.retrieval.retriever import SmartRetriever  # noqa: E402
from src.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IKEA RAG Chat",
    page_icon="🪑",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={},
)


# ── Per-session message store ─────────────────────────────────────────────────
def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "lc_store" not in st.session_state:
        st.session_state["lc_store"] = {}
    store: dict[str, ChatMessageHistory] = st.session_state["lc_store"]
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ── RAG system prompt ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """Eres un asistente experto en el catálogo de productos IKEA.
Responde siempre en el idioma en que te pregunten.
Para responder, usa primero el contexto proporcionado. Si la información no está en el contexto, \
búscala en el historial de la conversación antes de decir que no tienes información.
No hace falta que digas "según el contexto" o "según el historial", responde directamente.

Contexto:
{context}"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ── Cached resource: heavy components ────────────────────────────────────────
@st.cache_resource(show_spinner="Conectando a AWS Bedrock y ChromaDB…")
def _load_components():
    settings = Settings()  # type: ignore[call-arg]
    embeddings = get_embeddings(settings)
    vector_store = ChromaVectorStore(embeddings, settings)
    retriever = SmartRetriever(vector_store, settings)
    llm = ChatBedrock(
        model=settings.llm_model_id,
        region=settings.aws_region,
        aws_access_key_id=SecretStr(settings.aws_access_key_id),
        aws_secret_access_key=SecretStr(settings.aws_secret_access_key),
    )
    return settings, vector_store, retriever, llm


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def _ask(
    question: str, retriever: SmartRetriever, llm: ChatBedrock, session_id: str
) -> dict:
    docs = retriever.retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = RunnableWithMessageHistory(
        _PROMPT | llm | StrOutputParser(),
        _get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    answer = chain.invoke(
        {"context": context, "question": question},
        config={"configurable": {"session_id": session_id}},
    )
    return {"answer": answer, "source_documents": docs}


# ── Source rendering ──────────────────────────────────────────────────────────
def _render_sources(sources: List) -> None:
    if not sources:
        return
    with st.expander(f"📎 Fuentes consultadas ({len(sources)})", expanded=False):
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata
            filename = meta.get("filename") or meta.get("source", "Desconocido")
            file_type = meta.get("file_type", "?").upper()

            col_info, col_badge = st.columns([5, 1])
            with col_info:
                st.markdown(f"**{i}. {filename}**")
                detail_parts = []
                if "page" in meta:
                    detail_parts.append(f"Página {meta['page'] + 1}")
                if "chunk_index" in meta:
                    detail_parts.append(
                        f"Fragmento {meta['chunk_index'] + 1}/{meta.get('total_chunks', '?')}"
                    )
                if detail_parts:
                    st.caption(" · ".join(detail_parts))
                st.markdown(
                    f'<p class="source-preview">{doc.page_content[:350].strip()}…</p>',
                    unsafe_allow_html=True,
                )
            with col_badge:
                st.markdown(
                    f'<span class="source-badge">{file_type}</span>',
                    unsafe_allow_html=True,
                )
            if i < len(sources):
                st.divider()


# ── Chat ──────────────────────────────────────────────────────────────────────
def render_chat(
    retriever: SmartRetriever, llm: ChatBedrock, settings: Settings
) -> None:
    col_title, col_btn = st.columns([6, 1])
    with col_title:
        st.title("🪑 IKEA RAG Chat")
        st.caption(
            f"Modelo: `{settings.llm_model_id}` · Embeddings: `{settings.embedding_model_id}` · "
            f"Región: `{settings.aws_region}` · Chunks recuperados: `{settings.top_k}`"
        )
    with col_btn:
        st.write("")  # vertical alignment nudge
        if st.button("🗑️ Limpiar", use_container_width=True):
            session_id = _get_session_id()
            _get_session_history(session_id).clear()
            st.session_state["messages"] = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    if prompt := st.chat_input("Escribe tu pregunta aquí…"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en la base de conocimiento…"):
                try:
                    result = _ask(prompt, retriever, llm, _get_session_id())
                    answer = result["answer"]
                    sources = result["source_documents"]

                    st.markdown(answer)
                    if sources:
                        _render_sources(sources)

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    err = f"⚠️ Error al generar respuesta: {exc}"
                    st.error(err)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": err, "sources": []}
                    )


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    try:
        settings, vector_store, retriever, llm = _load_components()
        render_chat(retriever, llm, settings)
    except Exception as exc:
        st.error(f"❌ Error al inicializar el sistema: {exc}")
        st.info(
            "Asegúrate de que el archivo `.env` existe y tiene las credenciales de AWS. "
            "Consulta `.env.example` para ver las variables necesarias."
        )
        with st.expander("Detalle del error"):
            st.exception(exc)


if __name__ == "__main__":
    main()
