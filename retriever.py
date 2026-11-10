# retriever.py
"""BM25 retriever over the three papers with paper / page / section metadata.

Parsing notes (verified against PyMuPDF output, see scratch/inspect_headings.py):
  * Headings come out either split over two lines ('3.1' + 'Encoder and Decoder
    Stacks', Attention + Dynamic Chunking) or inline ('3.2 The Helium Text
    Language Model', Moshi).
  * Table cells produce look-alike pairs ('23.75' + 'Deep-Att + PosUnk [39]').
    These are rejected by only accepting a section number that is a valid
    *successor* of the current one (after 3.2: 3.2.1, 3.3 or 4).
  * Running page headers (e.g. Moshi's title on every page) are stripped so they
    don't inflate BM25 scores for every chunk.
  * Letter-spaced figure text ('n d o f t o k e s') is dropped line by line.
  * Everything after the 'References' heading is excluded from the index
    (bibliography entries match query words but never contain answers);
    lettered appendix headings after it re-enable indexing.
"""
import json
import os
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple

from rank_bm25 import BM25Okapi

PAPERS_META = {
    "attention_1706.03762v7": "Attention Is All You Need (1706.03762v7)",
    "moshi_2410.00037v1": "Moshi (2410.00037v1)",
    "dynamic_chunking_2507.07955v2": "Dynamic Chunking (2507.07955v2)",
}

FRONT_MATTER = "Abstract / front matter"
REFERENCES = "References"

_NUM_ONLY = re.compile(r"^(\d+(?:\.\d+)*)\.?$")
_NUM_INLINE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z].{1,80})$")
_APX_ONLY = re.compile(r"^([A-H](?:\.\d+)*)\.?$")
_APX_INLINE = re.compile(r"^([A-H](?:\.\d+)*)\.?\s+([A-Z].{1,80})$")


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _valid_successors(cur: Tuple[int, ...]) -> set:
    """Section numbers allowed to follow `cur`: first child, next sibling, or next ancestor."""
    if not cur:
        return {(1,)}
    succ = {cur + (1,)}
    for depth in range(len(cur), 0, -1):
        succ.add(cur[: depth - 1] + (cur[depth - 1] + 1,))
    return succ


def _looks_like_title(s: str) -> bool:
    return bool(re.match(r"^[A-Z]", s)) and len(s.split()) <= 10 and not s.endswith(".")


def _is_figure_garbage(line: str) -> bool:
    toks = line.split()
    if len(toks) < 6:
        return False
    single = sum(1 for t in toks if len(t) == 1)
    return single / len(toks) >= 0.6


def _is_garbage_chunk(piece) -> bool:
    """Chunk-level check: figure text often arrives one letter per line, which the
    line filter can't see. Drop chunks where >22% of words are single letters."""
    single = sum(1 for w in piece if len(w[0]) == 1 and w[0].isalpha())
    return single / max(1, len(piece)) > 0.22


_BOILERPLATE = ("hereby grants permission", "solely for use in journalistic")


class PaperRetriever:
    def __init__(self, piece_size: int = 256):
        self.piece_size = piece_size
        self.chunks: List[Dict] = []
        self.sections_found: Dict[str, List[Tuple[int, str]]] = {}
        self._build_chunks()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    # ---------- parsing ----------
    @staticmethod
    def _running_headers(pages: List[Dict]) -> set:
        """Lines that repeat on more than half the pages (titles, running heads)."""
        counts = Counter()
        for p in pages:
            counts.update({l.strip() for l in p["text"].split("\n") if len(l.strip()) > 15})
        return {l for l, n in counts.items() if n > len(pages) / 2}

    def _paper_words(self, paper_id: str, pages: List[Dict]) -> List[Tuple[str, int, str]]:
        """Return the paper as a flat list of (word, page, section)."""
        headers = self._running_headers(pages)
        cur_num: Tuple[int, ...] = ()
        cur_sec = FRONT_MATTER
        in_appendix = False
        found = []
        out = []

        for p in pages:
            lines = [l.strip() for l in p["text"].split("\n")]
            skip_next = False
            for i, line in enumerate(lines):
                if skip_next:
                    skip_next = False
                    continue
                if not line or line in headers or _is_figure_garbage(line):
                    continue
                if not re.search(r"[A-Za-z0-9]{2,}", line) and not (
                    _NUM_ONLY.match(line) or _APX_ONLY.match(line)
                ):
                    continue  # one-glyph lines from figure labels ('n', '!', "'!\"#"); keep '3.1'-style heading numbers
                if any(b in line.lower() for b in _BOILERPLATE):
                    continue
                nxt = lines[i + 1] if i + 1 < len(lines) else ""

                heading = None
                if line == "References" and cur_num:
                    cur_sec, heading = REFERENCES, REFERENCES
                elif not in_appendix and cur_sec != REFERENCES:
                    m_split = _NUM_ONLY.match(line)
                    m_inline = _NUM_INLINE.match(line)
                    if m_split and _looks_like_title(nxt):
                        num, title, skip_next = m_split.group(1), nxt, True
                    elif m_inline and _looks_like_title(m_inline.group(2)):
                        num, title = m_inline.group(1), m_inline.group(2)
                    else:
                        num = None
                    if num is not None:
                        tup = tuple(int(x) for x in num.split("."))
                        if tup in _valid_successors(cur_num):
                            cur_num, cur_sec = tup, f"{num} {title}"
                            heading = cur_sec
                        else:
                            skip_next = False  # table cell, not a heading
                if heading is None and cur_sec == REFERENCES or in_appendix:
                    m_split = _APX_ONLY.match(line)
                    m_inline = _APX_INLINE.match(line)
                    if m_split and _looks_like_title(nxt):
                        cur_sec, heading, skip_next = f"{m_split.group(1)} {nxt}", True, True
                        in_appendix = True
                    elif m_inline and _looks_like_title(m_inline.group(2)):
                        cur_sec, heading = line, True
                        in_appendix = True
                    if heading is True:
                        heading = cur_sec

                if heading:
                    found.append((p["page"], heading))
                    continue  # don't index the heading line itself as body text
                for w in line.split():
                    out.append((w, p["page"], cur_sec))

        self.sections_found[paper_id] = found
        return out

    def _build_chunks(self):
        for paper_id, title in PAPERS_META.items():
            path = f"data/parsed/{paper_id}.json"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {path}. Run fetch_papers.py first.")
            with open(path, "r", encoding="utf-8") as f:
                pages = json.load(f)

            words = [w for w in self._paper_words(paper_id, pages) if w[2] != REFERENCES]

            # Chunk within a page so every chunk has exactly one page citation.
            by_page: Dict[int, List[Tuple[str, int, str]]] = {}
            for w in words:
                by_page.setdefault(w[1], []).append(w)

            step = max(1, int(self.piece_size * 0.85))  # 15% overlap
            for page_num, pw in by_page.items():
                for i in range(0, len(pw), step):
                    piece = pw[i : i + self.piece_size]
                    if len(piece) < 30:
                        continue
                    text = " ".join(w[0] for w in piece)
                    if _is_garbage_chunk(piece):
                        continue  # figure labels extracted one letter per line
                    secs = list(dict.fromkeys(w[2] for w in piece))
                    self.chunks.append({
                        "paper_id": paper_id,
                        "paper_title": title,
                        "page": page_num,
                        "section": " → ".join(secs),
                        "text": text,
                    })

    # ---------- retrieval ----------
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        rerank: bool = False,
        paper_filter: Optional[str] = None,
    ) -> List[Dict]:
        scores = self.bm25.get_scores(tokenize(query))

        scored_chunks = []
        for idx, score in enumerate(scores):
            c = self.chunks[idx]
            if paper_filter and paper_filter != "all" and c["paper_id"] != paper_filter:
                continue
            scored_chunks.append((score, {**c, "bm25": float(score)}))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        candidates = [c for _, c in scored_chunks[:top_k]]

        if rerank and len(candidates) > 1:
            # Second-stage exact phrase / bigram + section heading proximity reranker
            q_words = [w for w in tokenize(query) if len(w) > 2]
            bigrams = [f"{q_words[i]} {q_words[i+1]}" for i in range(len(q_words) - 1)]

            def rerank_score(rank_and_chunk) -> float:
                rank_idx, chunk = rank_and_chunk
                t_lower = chunk["text"].lower()
                bg_hits = sum(3.0 for bg in bigrams if bg in t_lower)
                uniq_hits = sum(1.2 for w in set(q_words) if w in t_lower)
                sec_hits = sum(2.0 for w in set(q_words) if w in chunk["section"].lower())
                base_bonus = max(0.0, 2.0 - rank_idx * 0.3)  # keep a BM25 rank prior
                return bg_hits + uniq_hits + sec_hits + base_bonus

            ranked = sorted(enumerate(candidates), key=rerank_score, reverse=True)
            # Keep top 3 after reranking from top 5 to save LLM input tokens
            return [c for _, c in ranked[:3]]

        return candidates


RETRIEVAL_CONFIGS = {
    "256w_top3_norerank": {"piece_size": 256, "top_k": 3, "rerank": False},
    "512w_top5_norerank": {"piece_size": 512, "top_k": 5, "rerank": False},
    "512w_top5_rerank3":  {"piece_size": 512, "top_k": 5, "rerank": True},
}


if __name__ == "__main__":
    r = PaperRetriever(256)
    for pid, secs in r.sections_found.items():
        print(f"\n== {pid}: {len(secs)} headings")
        for page, s in secs:
            print(f"  p{page:>2}  {s}")
    print(f"\nChunks (256w): {len(r.chunks)}")
