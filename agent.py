import json
import re
import random
from typing import List, TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from google.api_core.exceptions import ResourceExhausted
import time

from retriever import retriever
from memory import memory, remember
from tools import TOOLS

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    convert_system_message_to_human=True
)

class ChatState(TypedDict):
    messages: Annotated[List[str], ...]
    steps: int
    docs: str | None
    route: str | None
    error: str | None
    attitude: str


def juna_style(att: str) -> str:
    if att == "random":
        att = random.choice(["baik", "galak"])
    return (
        "Kamu menjawab seperti Gordon Ramsay: tegas, sinis, namun tetap sopan."
        if att in ("galak", "mean")
        else "Kamu menjawab ramah, antusias, dan suportif."
    )


SYSTEM_BASE = (
    "Kamu adalah chef virtual bernama Juna yang ahli resep masakan Indonesia.\n"
    "Gunakan hanya data dari KONTEKS RESEP yang diberikan.\n"
    "Jika perlu menggunakan tool, gunakan hanya tool yang disediakan.\n"
    "Format pemanggilan tool:\n"
    "<tool>CALL_nama_tool {\"arg1\": \"value1\"}</tool>\n"
)


def safe_llm_invoke(prompt: str):
    try:
        return llm.invoke(prompt).content.strip()
    except ResourceExhausted:
        time.sleep(12)
        return llm.invoke(prompt).content.strip()


def build_agent():
    g = StateGraph(ChatState)

    # ───── Retrieve langsung dari user message (hapus rewrite node)
    def retrieve_node(state: ChatState) -> ChatState:
        user_last = state["messages"][-1]
        docs = retriever.get_relevant_documents(user_last)
        state["docs"] = "\n\n".join(d.page_content for d in docs)
        return state

    # ───── Router tetap
    def router_node(state: ChatState) -> ChatState:
        last = state["messages"][-1].lower()

        if "juna" in last and any(w in last for w in
            ["mean", "galak", "random", "sikap"]):
            state["route"] = "att_change"
        else:
            state["route"] = "answer"

        return state

    # ───── Set Attitude (tanpa LLM)
    def att_set_node(state: ChatState) -> ChatState:
        user_msg = state["messages"][-1].lower()
        match = re.search(r"\b(baik|galak|mean|random)\b", user_msg)
        new_att = match.group(1) if match else "baik"
        state["attitude"] = new_att

        msg = f"Sikap Juna di-set ke '{new_att}'."
        state["messages"].append(msg)
        remember("ai", msg)
        return state

    # ───── Single LLM Answer Node (Gabungan decide + synth)
    def answer_node(state: ChatState) -> ChatState:
        hist = memory.load_memory_variables({}).get("history", "")

        prompt = (
            juna_style(state["attitude"]) + "\n" +
            SYSTEM_BASE +
            f"\n\nRiwayat:\n{hist}\n" +
            f"\nKONTEKS RESEP:\n{state['docs']}\n\n" +
            f"Pertanyaan:\n{state['messages'][-1]}\n\n" +
            "Jawaban ringkas dalam Bahasa Indonesia:"
        )

        ans = safe_llm_invoke(prompt)

        state["messages"].append(ans)
        remember("ai", ans)

        return state

    # ───── Error Node
    def error_node(state: ChatState) -> ChatState:
        msg = state.get("error", "Maaf, terjadi kesalahan.")
        state["messages"].append(msg)
        remember("ai", msg)
        return state

    # ===============================
    # GRAPH FLOW
    # ===============================
    g.add_node("retrieve", retrieve_node)
    g.add_node("router", router_node)
    g.add_node("att_set", att_set_node)
    g.add_node("answer", answer_node)
    g.add_node("error", error_node)

    g.add_edge("retrieve", "router")

    g.add_conditional_edges("router", lambda s: s["route"], {
        "att_change": "att_set",
        "answer": "answer"
    })

    g.add_edge("att_set", END)
    g.add_edge("answer", END)
    g.add_edge("error", END)

    g.set_entry_point("retrieve")

    return g.compile()
