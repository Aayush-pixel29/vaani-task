# questions.py
"""Evaluation questions + deterministic grader.

Grading rules (answer_grade column in results.csv):
  * Refusals are detected from a fixed sentinel the pipeline must emit
    (REFUSAL_SENTINEL), not from guessing at free-text phrasing.
  * An answerable question is "correct" only if every must_match_all pattern
    matches the answer body AND the answer cites the target paper.
  * Patterns are matched against the answer with citations stripped, so page
    numbers like "[p.8]" can't satisfy a fact check like "h = 8".
"""
import re
from retriever import PaperRetriever

REFUSAL_SENTINEL = "NOT_IN_PAPERS"

PAPER_CITE_PATTERNS = {
    "attention_1706.03762v7": r"1706\.03762|attention is all you need",
    "moshi_2410.00037v1": r"2410\.00037|\bmoshi\b",
    "dynamic_chunking_2507.07955v2": r"2507\.07955|dynamic chunking|h-net",
}
CITATION_RE = re.compile(r"\[[^\]]*\]")  # citations are written as [Paper, p.N]

EVAL_QUESTIONS = [
    # --- PAPER 1: Attention Is All You Need (1706.03762v7) ---
    {
        "id": "Q01",
        "question": "In 'Attention Is All You Need', what are the exact number of encoder layers N, model dimension d_model, and attention heads h in the base Transformer model?",
        "expected_type": "answerable",
        "target_paper": "attention_1706.03762v7",
        "must_match_all": [r"n\s*=\s*6|\b6\s+(identical\s+)?(encoder\s+)?layers|\bsix\b",
                           r"512",
                           r"h\s*=\s*8|\b8\s+(parallel\s+)?(attention\s+)?heads|\beight\b"],
    },
    {
        "id": "Q02",
        "question": "What BLEU score did the big Transformer model achieve on the WMT 2014 English-to-German translation task in 'Attention Is All You Need'?",
        "expected_type": "answerable",
        "target_paper": "attention_1706.03762v7",
        "must_match_all": [r"28\.4"],
    },
    {
        "id": "Q03",
        "question": "Why do the authors of 'Attention Is All You Need' scale the dot products by 1/sqrt(d_k) in Scaled Dot-Product Attention?",
        "expected_type": "answerable",
        "target_paper": "attention_1706.03762v7",
        "must_match_all": [r"softmax", r"gradient"],
    },
    {
        "id": "Q04",
        "question": "In 'Attention Is All You Need', what optimizer parameter values (beta_1, beta_2, epsilon) and warmup_steps were used during training?",
        "expected_type": "answerable",
        "target_paper": "attention_1706.03762v7",
        "must_match_all": [r"0\.9(?!\d)", r"0\.98", r"10\s*\^?\s*[-−]\s*9|1e-9", r"4000|4,000"],
    },

    # --- PAPER 2: Moshi (2410.00037v1) ---
    {
        "id": "Q05",
        "question": "In the Moshi paper, what is the parameter size of the Helium text language model backbone and how many tokens was it trained on?",
        "expected_type": "answerable",
        "target_paper": "moshi_2410.00037v1",
        "must_match_all": [r"\b7\s*b\b|7\s*billion", r"2\.1\s*t\b|2\.1\s*trillion"],
    },
    {
        "id": "Q06",
        "question": "What is the frame rate (in Hz) and frame size (in ms) of the Mimi neural audio codec used in Moshi?",
        "expected_type": "answerable",
        "target_paper": "moshi_2410.00037v1",
        "must_match_all": [r"12\.5\s*hz", r"80\s*ms"],
    },
    {
        "id": "Q07",
        "question": "What is 'Inner Monologue' in Moshi and how does predicting time-aligned text tokens before audio tokens help?",
        "expected_type": "answerable",
        "target_paper": "moshi_2410.00037v1",
        "must_match_all": [r"inner monologue", r"text"],
    },
    {
        "id": "Q08",
        "question": "How does the RQ-Transformer in Moshi split sequence modeling into a Temporal Transformer and a Depth Transformer?",
        "expected_type": "answerable",
        "target_paper": "moshi_2410.00037v1",
        "must_match_all": [r"temporal", r"depth"],
    },

    # --- PAPER 3: Dynamic Chunking / H-Net (2507.07955v2) ---
    {
        "id": "Q09",
        "question": "In 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling', what is H-Net and how does it replace fixed BPE tokenization?",
        "expected_type": "answerable",
        "target_paper": "dynamic_chunking_2507.07955v2",
        "must_match_all": [r"h-net", r"byte|chunk"],
    },
    {
        "id": "Q10",
        "question": "What are the main components of the Dynamic Chunking mechanism (such as the routing module and smoothing module) in H-Net, and how does the smoothing module help training?",
        "expected_type": "answerable",
        "target_paper": "dynamic_chunking_2507.07955v2",
        "must_match_all": [r"routing", r"smoothing|ema|gradient|differentiab"],
    },
    {
        "id": "Q11",
        "question": "In the Dynamic Chunking paper, what loss function is added to control the target compression ratio of chunks?",
        "expected_type": "answerable",
        "target_paper": "dynamic_chunking_2507.07955v2",
        "must_match_all": [r"ratio\s+loss"],
    },

    # --- NAIVE RETRIEVER TRAP QUESTIONS ---
    {
        "id": "Q12",
        "question": "What is the theoretical latency and practical overall latency (in ms) of the Moshi spoken dialogue system?",
        "expected_type": "answerable",
        "target_paper": "moshi_2410.00037v1",
        "must_match_all": [r"160\s*ms", r"200\s*ms"],
    },
    {
        "id": "Q13",
        "question": "How is cosine similarity between adjacent representations used to calculate boundary probabilities p_t in Dynamic Chunking?",
        "expected_type": "answerable",
        "target_paper": "dynamic_chunking_2507.07955v2",
        "must_match_all": [r"cosine", r"boundary|key|query"],
    },

    # --- UNANSWERABLE QUESTIONS (must refuse) ---
    {"id": "Q14", "expected_type": "unanswerable", "target_paper": "none", "must_match_all": [],
     "question": "What exact Word Error Rate (WER) did Moshi achieve on the Marathi and Hindi subsets of CommonVoice?"},
    {"id": "Q15", "expected_type": "unanswerable", "target_paper": "none", "must_match_all": [],
     "question": "In 'Attention Is All You Need', how does Rotary Position Embedding (RoPE) compare against FlashAttention-2 on A100 GPUs?"},
    {"id": "Q16", "expected_type": "unanswerable", "target_paper": "none", "must_match_all": [],
     "question": "What was the exact AWS cloud bill in USD for pre-training the 2-stage H-Net model in the Dynamic Chunking paper?"},
    {"id": "Q17", "expected_type": "unanswerable", "target_paper": "none", "must_match_all": [],
     "question": "How many parameters does the Whisper-large-v3 encoder use when fine-tuned inside the Attention Is All You Need architecture?"},
    {"id": "Q18", "expected_type": "unanswerable", "target_paper": "none", "must_match_all": [],
     "question": "What learning rate schedule did the authors use to train the 70B parameter version of Helium in the Moshi paper?"},
]


def grade_answer(question_item: dict, answer_text: str) -> str:
    """Return one of: correct, wrong, refused_correctly, refused_wrongly."""
    is_refusal = REFUSAL_SENTINEL in answer_text
    if question_item["expected_type"] == "unanswerable":
        return "refused_correctly" if is_refusal else "wrong"
    if is_refusal:
        return "refused_wrongly"

    citations = " ".join(CITATION_RE.findall(answer_text)).lower()
    body = CITATION_RE.sub(" ", answer_text).lower()
    cites_target = bool(re.search(PAPER_CITE_PATTERNS[question_item["target_paper"]], citations))
    facts_ok = all(re.search(p, body) for p in question_item["must_match_all"])
    return "correct" if (facts_ok and cites_target) else "wrong"


def _grader_self_test():
    q = {x["id"]: x for x in EVAL_QUESTIONS}
    cases = [
        # (question, answer, expected grade, why)
        ("Q01", "The base model uses N = 6 layers, d_model = 512 and h = 8 heads [Attention Is All You Need, p.3].", "correct", "all facts + cite"),
        ("Q01", "The model has 512 dimensions and many heads [Attention Is All You Need, p.6] [Attention Is All You Need, p.8].", "wrong", "page numbers must not satisfy 6/8"),
        ("Q01", "N = 6 layers, d_model = 512, h = 8 heads.", "wrong", "no citation"),
        ("Q01", "N = 6 layers, d_model = 512, h = 8 heads [Moshi, p.7].", "wrong", "cites wrong paper"),
        ("Q12", "160 ms theoretical, 200 ms in practice [Moshi, p.2].", "correct", "facts + cite"),
        ("Q12", f"{REFUSAL_SENTINEL}: the papers do not state this.", "refused_wrongly", "refused answerable"),
        ("Q15", f"{REFUSAL_SENTINEL}: the paper does not discuss RoPE or FlashAttention-2.", "refused_correctly", "correct refusal"),
        ("Q15", "RoPE is about 10% faster than FlashAttention-2 [Attention Is All You Need, p.6].", "wrong", "hallucinated answer"),
    ]
    ok = True
    for qid, ans, exp, why in cases:
        got = grade_answer(q[qid], ans)
        flag = "PASS" if got == exp else "FAIL"
        ok &= got == exp
        print(f"  {flag} {qid} expected={exp:17} got={got:17} ({why})")
    return ok


if __name__ == "__main__":
    print("Grader self-test:")
    print("  ALL PASS" if _grader_self_test() else "  SOME FAILED")

    r = PaperRetriever(piece_size=256)
    print(f"\nTotal clean chunks indexed (256w): {len(r.chunks)}")
    for qid in ["Q01", "Q06", "Q10", "Q12", "Q15"]:
        item = next(x for x in EVAL_QUESTIONS if x["id"] == qid)
        hits = r.retrieve(item["question"], top_k=3)
        print(f"\n--- {item['id']}: {item['question']} ---")
        for h in hits:
            print(f"  -> [{h['paper_id']} | p.{h['page']} | {h['section']}] {h['text'][:110]}...")
