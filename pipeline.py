# pipeline.py
import os
import re
import json
import time
import random
import asyncio
import httpx
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from retriever import PaperRetriever, RETRIEVAL_CONFIGS
from telemetry import RequestLog, log_request
from questions import EVAL_QUESTIONS, grade_answer

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


@dataclass
class StepTrace:
    step_name: str
    wait_ms: float = 0.0
    work_ms: float = 0.0
    retry_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    retries: int = 0
    status: str = "ok"


class RateLimitState:
    """Tracks live rate-limit headers from Groq or the Stand-In server."""
    def __init__(self):
        self.remaining_tokens: int = 8000
        self.remaining_requests: int = 1000
        self.reset_tokens_sec: float = 0.0
        self.last_updated: float = time.time()

    def update_from_headers(self, headers: httpx.Headers):
        rem_t = headers.get("x-ratelimit-remaining-tokens")
        rem_r = headers.get("x-ratelimit-remaining-requests")
        if rem_t is not None:
            try:
                self.remaining_tokens = int(rem_t)
            except ValueError:
                pass
        if rem_r is not None:
            try:
                self.remaining_requests = int(rem_r)
            except ValueError:
                pass
        self.last_updated = time.time()


class RelayPipeline:
    def __init__(
        self,
        base_url: str = GROQ_URL,
        source_label: str = "real",
        default_setting: str = "512w_top5_norerank",
        adaptive_retrieval: bool = False,
        max_retries: int = 2,
        request_timeout_s: float = 45.0,
    ):
        self.base_url = base_url
        self.source_label = source_label
        self.default_setting = default_setting
        self.adaptive_retrieval = adaptive_retrieval
        self.max_retries = max_retries
        self.request_timeout_s = request_timeout_s
        self.api_key = os.getenv("GROQ_API_KEY", "stand-in-key")
        self.rl_state = RateLimitState()

        # Pre-build both 256w and 512w indexes so switching settings costs 0ms
        self.retrievers = {
            256: PaperRetriever(piece_size=256),
            512: PaperRetriever(piece_size=512),
        }

    async def _llm_call(
        self,
        client: httpx.AsyncClient,
        messages: List[Dict[str, str]],
        max_tokens: int = 350,
        stream: bool = True,
    ) -> Dict[str, Any]:
        """Executes a single LLM call with full-jitter backoff on 429 and accurate TTFT tracking."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
            "reasoning_effort": "low",
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}

        retries = 0
        retry_ms_total = 0.0
        work_ms_total = 0.0

        while True:
            t_attempt = time.time()
            first_visible_token_ms = -1.0
            content_parts: List[str] = []
            tokens_in = 0
            tokens_out = 0

            try:
                if stream:
                    async with client.stream(
                        "POST", self.base_url, json=payload, headers=headers, timeout=25.0
                    ) as resp:
                        self.rl_state.update_from_headers(resp.headers)

                        if resp.status_code == 429:
                            body_bytes = await resp.aread()
                            work_ms_total += (time.time() - t_attempt) * 1000.0
                            if retries >= self.max_retries:
                                return {
                                    "status": "rate_limited",
                                    "text": "",
                                    "ttft_ms": -1.0,
                                    "work_ms": work_ms_total,
                                    "retry_ms": retry_ms_total,
                                    "retries": retries,
                                    "tokens_in": tokens_in,
                                    "tokens_out": tokens_out,
                                }
                            retry_after = float(resp.headers.get("retry-after", "6.0"))
                            # Full jitter around retry_after to prevent synchronized retry storms!
                            sleep_s = retry_after + random.uniform(0.5, 2.5)
                            t_r0 = time.time()
                            await asyncio.sleep(sleep_s)
                            retry_ms_total += (time.time() - t_r0) * 1000.0
                            retries += 1
                            continue

                        resp.raise_for_status()

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            # Capture exact usage block at end of stream (standard or x_groq)
                            u = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                            if u:
                                tokens_in = u.get("prompt_tokens", tokens_in)
                                tokens_out = u.get("completion_tokens", tokens_out)

                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                visible_txt = delta.get("content")
                                if visible_txt:
                                    if first_visible_token_ms < 0:
                                        first_visible_token_ms = (time.time() - t_attempt) * 1000.0
                                    content_parts.append(visible_txt)

                        work_ms_total += (time.time() - t_attempt) * 1000.0
                        full_text = "".join(content_parts).strip()
                        if tokens_in == 0:
                            tokens_in = int(sum(len(m["content"].split()) * 1.33 for m in messages))
                        if tokens_out == 0:
                            tokens_out = max(1, int(len(full_text.split()) * 1.33))

                        return {
                            "status": "ok",
                            "text": full_text,
                            "ttft_ms": first_visible_token_ms if first_visible_token_ms > 0 else work_ms_total,
                            "work_ms": work_ms_total,
                            "retry_ms": retry_ms_total,
                            "retries": retries,
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                        }
            except httpx.TimeoutException:
                work_ms_total += (time.time() - t_attempt) * 1000.0
                return {
                    "status": "timeout",
                    "text": "",
                    "ttft_ms": -1.0,
                    "work_ms": work_ms_total,
                    "retry_ms": retry_ms_total,
                    "retries": retries,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }
            except Exception as e:
                work_ms_total += (time.time() - t_attempt) * 1000.0
                return {
                    "status": "error",
                    "text": "",
                    "ttft_ms": -1.0,
                    "work_ms": work_ms_total,
                    "retry_ms": retry_ms_total,
                    "retries": retries,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }

    def _step1_route_deterministic_or_hint(self, question: str) -> Optional[str]:
        """Fast pre-check before LLM Router to save tokens when paper is explicitly named."""
        q_low = question.lower()
        if "attention is all you need" in q_low or "scaled dot-product" in q_low or "wmt 2014" in q_low:
            return "attention_1706.03762v7"
        if "moshi" in q_low or "mimi" in q_low or "helium" in q_low or "inner monologue" in q_low or "full-duplex spoken" in q_low:
            return "moshi_2410.00037v1"
        if "dynamic chunking" in q_low or "h-net" in q_low or "llamabyte" in q_low:
            return "dynamic_chunking_2507.07955v2"
        return None

    async def answer_question(
        self,
        client: httpx.AsyncClient,
        request_id: str,
        question: str,
        scenario: str = "eval",
        retrieval_setting: Optional[str] = None,
        question_meta: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        t_start = time.time()
        setting_name = retrieval_setting or self.default_setting

        # Adaptive retrieval downgrade when approaching TPM ceiling
        if self.adaptive_retrieval and self.rl_state.remaining_tokens < 3800:
            setting_name = "256w_top3_norerank"

        cfg = RETRIEVAL_CONFIGS[setting_name]
        total_tokens_in = 0
        total_tokens_out = 0
        total_retries = 0
        traces: List[StepTrace] = []

        # --- STEP 1: QUERY & PAPER ROUTER (Model Call 1) ---
        paper_filter = self._step1_route_deterministic_or_hint(question)
        if not paper_filter:
            router_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Route the user question to one paper ID: "
                        "'attention_1706.03762v7' (Transformer/Vaswani), "
                        "'moshi_2410.00037v1' (Moshi speech/audio dialogue/Mimi/Helium), "
                        "'dynamic_chunking_2507.07955v2' (H-Net/Dynamic Chunking/byte sequences), "
                        "or 'all'. Reply ONLY with the paper ID and 3 search keywords separated by |."
                    ),
                },
                {"role": "user", "content": question},
            ]
            r_res = await self._llm_call(client, router_prompt, max_tokens=45, stream=True)
            total_tokens_in += r_res["tokens_in"]
            total_tokens_out += r_res["tokens_out"]
            total_retries += r_res["retries"]
            traces.append(StepTrace("router", work_ms=r_res["work_ms"], retry_ms=r_res["retry_ms"],
                                    tokens_in=r_res["tokens_in"], tokens_out=r_res["tokens_out"],
                                    retries=r_res["retries"], status=r_res["status"]))

            if r_res["status"] != "ok":
                return self._finalize_failure(request_id, scenario, t_start, r_res["status"],
                                              total_retries, total_tokens_in, total_tokens_out, setting_name)

            r_text = r_res["text"].lower()
            for pid in ["attention_1706.03762v7", "moshi_2410.00037v1", "dynamic_chunking_2507.07955v2"]:
                if pid in r_text:
                    paper_filter = pid
                    break
        else:
            traces.append(StepTrace("router_deterministic_hint", work_ms=0.5, status="ok"))

        # --- STEP 2: RETRIEVAL TOOL CALL ---
        t_ret0 = time.time()
        retriever = self.retrievers[cfg["piece_size"]]
        chunks = retriever.retrieve(
            query=question,
            top_k=cfg["top_k"],
            rerank=cfg["rerank"],
            paper_filter=paper_filter,
        )
        ret_ms = (time.time() - t_ret0) * 1000.0
        traces.append(StepTrace("retriever_tool", work_ms=ret_ms, status="ok"))

        context_blocks = []
        valid_pages = set()
        for c in chunks:
            valid_pages.add((c["paper_id"].lower(), int(c["page"])))
            context_blocks.append(
                f"SOURCE [{c['paper_id']}, p.{c['page']}, {c['section']}]:\n{c['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        # --- STEP 3: DRAFT CITED REPLY (Model Call 2) ---
        draft_sys = (
            "You are a strict research paper assistant. Answer the question using ONLY the provided SOURCE passages.\n"
            "RULES:\n"
            "1. Cite every factual claim using exact format [paper_id, p.X] (e.g., [attention_1706.03762v7, p.3]).\n"
            "2. Include exact numbers, equations, and terminology from the passages.\n"
            "3. If the provided SOURCE passages do NOT explicitly contain the answer to the specific question "
            "(for example, if asked about benchmarks, languages, models, or cloud costs not in the text), "
            "you MUST reply starting with the exact token: NOT_IN_PAPERS: followed by a 1-sentence explanation. "
            "Never guess or extrapolate."
        )
        draft_messages = [
            {"role": "system", "content": draft_sys},
            {"role": "user", "content": f"SOURCES:\n{context_str}\n\nQUESTION: {question}"},
        ]

        d_res = await self._llm_call(client, draft_messages, max_tokens=280, stream=True)
        total_tokens_in += d_res["tokens_in"]
        total_tokens_out += d_res["tokens_out"]
        total_retries += d_res["retries"]
        traces.append(StepTrace("drafter", work_ms=d_res["work_ms"], retry_ms=d_res["retry_ms"],
                                tokens_in=d_res["tokens_in"], tokens_out=d_res["tokens_out"],
                                retries=d_res["retries"], status=d_res["status"]))

        if d_res["status"] != "ok":
            return self._finalize_failure(request_id, scenario, t_start, d_res["status"],
                                          total_retries, total_tokens_in, total_tokens_out, setting_name)

        draft_text = d_res["text"]

        # --- STEP 4: CRITIC & VERIFICATION LOOP (Branch / Loop) ---
        # 4a. Deterministic Citation & Grounding Check
        is_refusal = "NOT_IN_PAPERS" in draft_text
        cited_matches = re.findall(r"\[([a-zA-Z0-9_\-\.]+),\s*(?:p\.?|page)\s*(\d+)\]", draft_text)
        citations_valid = len(cited_matches) > 0 and all(
            (pid.lower(), int(pg)) in valid_pages for pid, pg in cited_matches
        )

        # If draft neither refused nor included a valid retrieved citation, loop once to repair!
        if not is_refusal and not citations_valid:
            repair_messages = draft_messages + [
                {"role": "assistant", "content": draft_text},
                {
                    "role": "user",
                    "content": (
                        "CRITIC REJECTION: Your draft either lacked an exact [paper_id, p.X] citation from the "
                        "provided SOURCES or cited a page not in SOURCES. Rewrite with exact [paper_id, p.X] "
                        "citations from SOURCES, or output NOT_IN_PAPERS: if unanswerable."
                    ),
                },
            ]
            rep_res = await self._llm_call(client, repair_messages, max_tokens=260, stream=True)
            total_tokens_in += rep_res["tokens_in"]
            total_tokens_out += rep_res["tokens_out"]
            total_retries += 1 + rep_res["retries"]
            traces.append(StepTrace("critic_repair_loop", work_ms=rep_res["work_ms"], retry_ms=rep_res["retry_ms"],
                                    tokens_in=rep_res["tokens_in"], tokens_out=rep_res["tokens_out"],
                                    retries=1 + rep_res["retries"], status=rep_res["status"]))
            if rep_res["status"] == "ok":
                draft_text = rep_res["text"]
                is_refusal = "NOT_IN_PAPERS" in draft_text

        # 4b. LLM Verification Critic Call (lightweight grounding audit on non-refusal answers)
        if not is_refusal and self.rl_state.remaining_tokens > 2000:
            critic_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Verify if the DRAFT answer is supported by the SOURCES and directly answers the QUESTION "
                        "without fabricating metrics. Reply 'VERIFIED' if supported, or 'REJECT: NOT_IN_PAPERS' if "
                        "the draft fabricates facts not in SOURCES."
                    ),
                },
                {
                    "role": "user",
                    "content": f"QUESTION: {question}\n\nDRAFT: {draft_text}\n\nSOURCES (excerpt):\n{context_str[:3500]}",
                },
            ]
            c_res = await self._llm_call(client, critic_prompt, max_tokens=40, stream=True)
            total_tokens_in += c_res["tokens_in"]
            total_tokens_out += c_res["tokens_out"]
            total_retries += c_res["retries"]
            traces.append(StepTrace("llm_critic", work_ms=c_res["work_ms"], retry_ms=c_res["retry_ms"],
                                    tokens_in=c_res["tokens_in"], tokens_out=c_res["tokens_out"],
                                    retries=c_res["retries"], status=c_res["status"]))
            if c_res["status"] == "ok" and "REJECT" in c_res["text"].upper():
                draft_text = "NOT_IN_PAPERS: Rejected by verification critic as unsupported by the retrieved passages."

        # Buffer released to user ONLY after Critic verification passes!
        t_verified_start = time.time()
        first_token_ms = (t_verified_start - t_start) * 1000.0  # Time to first verified token released

        # Stream verified answer to client (simulating client-side SSE emission at ~15ms per token)
        words_out = draft_text.split()
        stream_flush_ms = max(50.0, len(words_out) * 15.0)  # Inter-token streaming transmission duration
        await asyncio.sleep(stream_flush_ms / 1000.0)

        total_ms = (time.time() - t_start) * 1000.0

        grade = grade_answer(question_meta, draft_text) if question_meta else "correct"

        log_entry = RequestLog(
            request_id=request_id,
            scenario=scenario,
            source=self.source_label,
            start_time=round(t_start, 3),
            first_token_ms=round(first_token_ms, 1),
            total_ms=round(total_ms, 1),
            status="ok",
            retries=total_retries,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            retrieval_setting=setting_name,
            answer_grade=grade,
        )
        log_request(log_entry)

        return {
            "request_id": request_id,
            "answer": draft_text,
            "grade": grade,
            "first_token_ms": first_token_ms,
            "draft_raw_ttft_ms": d_res["ttft_ms"],
            "total_ms": total_ms,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "retries": total_retries,
            "traces": [t.__dict__ for t in traces],
        }

    def _finalize_failure(
        self, request_id: str, scenario: str, t_start: float, status: str,
        retries: int, tokens_in: int, tokens_out: int, setting_name: str
    ) -> Dict[str, Any]:
        total_ms = (time.time() - t_start) * 1000.0
        log_entry = RequestLog(
            request_id=request_id,
            scenario=scenario,
            source=self.source_label,
            start_time=round(t_start, 3),
            first_token_ms=-1.0,
            total_ms=round(total_ms, 1),
            status=status,
            retries=retries,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            retrieval_setting=setting_name,
            answer_grade="wrong",
        )
        log_request(log_entry)
        return {"request_id": request_id, "status": status, "total_ms": total_ms}

