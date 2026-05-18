from __future__ import annotations
from typing import Optional, Dict, List
from app.rag.retriever import LocalRetriever
from app.rag.generator import AnswerGenerator
from app.memory.conversation_store import append_message, load_history

class SugarCaneAgent:
    def __init__(self):
        self.retriever = LocalRetriever()
        self.generator = AnswerGenerator()
        self.last_prediction: Optional[Dict] = None

    def set_prediction(self, prediction: Dict):
        self.last_prediction = prediction

    def answer(self, question: str, prediction: Optional[Dict] = None, history: Optional[List[Dict]] = None, session_id: Optional[str] = None) -> str:
        pred = prediction or self.last_prediction
        effective_history = history or (load_history(session_id) if session_id else [])
        chunks = self.retriever.search(question, k=5, prediction=pred, history=effective_history)
        answer = self.generator.generate(question, chunks, pred, effective_history)
        if session_id:
            pred_meta = None
            if pred:
                pred_meta = {
                    k: v for k, v in pred.items()
                    if k not in {"gradcam_image"}
                }
                if pred.get("gradcam_image") is not None:
                    pred_meta["gradcam_available"] = True
            append_message(session_id, "user", question, {"prediction": pred_meta})
            append_message(session_id, "assistant", answer, {"sources": [{"source": c["source"], "chunk_id": c["chunk_id"], "score": c.get("score")} for c in chunks]})
        return answer
