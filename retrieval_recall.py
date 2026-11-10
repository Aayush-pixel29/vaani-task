# retrieval_recall.py
"""Offline retrieval check (no LLM calls): for each answerable question and each
retrieval setting, does the retrieved context contain the gold evidence, and from
the right paper? This bounds what the LLM can possibly get right per setting."""
import re
from retriever import PaperRetriever, RETRIEVAL_CONFIGS
from questions import EVAL_QUESTIONS

# Evidence strings that must appear in the retrieved text (verified in the parsed PDFs).
EVIDENCE = {
    "Q01": [r"N\s*=\s*6", r"512", r"h\s*=\s*8"],
    "Q02": [r"28\.4"],
    "Q03": [r"softmax", r"gradients"],
    "Q04": [r"0\.98", r"warmup_steps\s*=\s*4000"],
    "Q05": [r"2\.1T"],
    "Q06": [r"12\.5\s*Hz", r"80\s*ms"],
    "Q07": [r"Inner Monologue"],
    "Q08": [r"Temporal Transformer", r"Depth Transformer"],
    "Q09": [r"H-Net"],
    "Q10": [r"routing module", r"smoothing module"],
    "Q11": [r"ratio loss"],
    "Q12": [r"160\s*ms", r"200\s*ms"],
    "Q13": [r"cosine"],
}

if __name__ == "__main__":
    retrievers = {}
    totals = {}
    print(f"{'Q':4} " + " ".join(f"{k:>20}" for k in RETRIEVAL_CONFIGS))
    for q in EVAL_QUESTIONS:
        if q["expected_type"] != "answerable":
            continue
        row = []
        for name, cfg in RETRIEVAL_CONFIGS.items():
            r = retrievers.setdefault(cfg["piece_size"], PaperRetriever(cfg["piece_size"]))
            hits = r.retrieve(q["question"], top_k=cfg["top_k"], rerank=cfg["rerank"])
            ctx = " ".join(h["text"] for h in hits if h["paper_id"] == q["target_paper"])
            found = sum(bool(re.search(p, ctx, re.I)) for p in EVIDENCE[q["id"]])
            need = len(EVIDENCE[q["id"]])
            totals[name] = totals.get(name, 0) + (found == need)
            pages = ",".join(str(h["page"]) for h in hits)
            row.append(f"{found}/{need} p[{pages}]".rjust(20))
        print(f"{q['id']:4} " + " ".join(row))
    n = sum(1 for q in EVAL_QUESTIONS if q["expected_type"] == "answerable")
    print("\nFull evidence retrieved: " + " | ".join(f"{k}: {v}/{n}" for k, v in totals.items()))

