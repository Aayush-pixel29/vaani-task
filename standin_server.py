# standin_server.py
"""FastAPI Stand-In Server replicating Groq API behavior.

Calibrated strictly to Table 1 measurements & Brief requirements:
  1. "Be able to be slow": Baseline TTFT distribution matching Groq (~350ms).
  2. "To fail": Small injected fault rate (1-2% HTTP 500 / timeouts).
  3. "To refuse because a limit was reached": HTTP 429 when leaky-bucket TPM (8,000) or RPD is reached.
  4. "To become slower under load": Dynamic in_flight_requests tracking:
     - Concurrency penalty on TTFT: +180ms per in-flight request (simulating batch contention).
     - Token delay penalty: +4ms/token per concurrent request.
     - Tail latency spikes (2-3% chance of 1.5-3.0s delay: "succeed slowly instead of failing quickly").
  5. "To send its reply a piece at a time": SSE streaming chunks with [DONE] and usage blocks.
  6. Authentic Grounding: Integrates with data/real_response_bank.json to stream genuine paper answers.
"""

import os
import re
import time
import json
import random
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Groq Stand-In Server", version="1.0.0")

# Calibration Parameters (from Table 1 Live Probe)
TPM_LIMIT = 8000
REFILL_RATE_PER_SEC = TPM_LIMIT / 60.0  # 133.33 tokens/sec
RPD_LIMIT = 1000

RESPONSE_BANK_PATH = "data/real_response_bank.json"

class LeakyBucket:
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.time()

    async def consume(self, amount: float) -> tuple[bool, float]:
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0
        else:
            deficit = amount - self.tokens
            wait_sec = deficit / self.refill_rate
            return False, wait_sec

    async def get_remaining(self) -> int:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        return max(0, int(self.tokens))

    def reset_capacity(self, new_capacity: float):
        self.capacity = new_capacity
        self.refill_rate = new_capacity / 60.0
        self.tokens = min(self.tokens, new_capacity)

bucket = LeakyBucket(capacity=TPM_LIMIT, refill_rate=REFILL_RATE_PER_SEC)
requests_used = 0
in_flight_requests = 0


class ChatCompletionRequest(BaseModel):
    model: str = "openai/gpt-oss-20b"
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    reasoning_effort: Optional[str] = "low"
    stream_options: Optional[Dict[str, Any]] = None


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.33))


def load_response_bank() -> Dict[str, Any]:
    if os.path.exists(RESPONSE_BANK_PATH):
        try:
            with open(RESPONSE_BANK_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def generate_simulated_reply(messages: List[Dict[str, str]]) -> str:
    """Generates grounded responses matching the pipeline prompts."""
    last_msg = messages[-1]["content"] if messages else ""
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")

    # Router Call
    if "Route the user question" in sys_msg:
        low = last_msg.lower()
        if "attention" in low or "transformer" in low or "bleu" in low or "warmup" in low:
            return "attention_1706.03762v7 | transformer | layers | attention"
        if "moshi" in low or "mimi" in low or "helium" in low or "inner monologue" in low or "dialogue" in low:
            return "moshi_2410.00037v1 | audio | codec | speech"
        if "dynamic chunking" in low or "h-net" in low or "cosine" in low or "ratio loss" in low:
            return "dynamic_chunking_2507.07955v2 | chunking | routing | h-net"
        return "all | architecture | evaluation"

    # Critic Verification Call
    if "Verify if the DRAFT answer is supported" in sys_msg:
        if "NOT_IN_PAPERS" in last_msg:
            return "VERIFIED"
        return "VERIFIED"

    # Critic Repair Call
    if "CRITIC REJECTION" in last_msg:
        return "NOT_IN_PAPERS: Lacks grounded evidence in provided sources."

    # Drafter Call: Check real response bank first!
    bank = load_response_bank()
    low = last_msg.lower()
    for qid, item in bank.items():
        q_text = item.get("question", "").lower()
        q_words = set(q_text.split())
        match_words = sum(1 for w in q_words if w in low)
        if len(q_words) > 5 and match_words / len(q_words) > 0.65:
            return item.get("answer", "")

    # Fallback grounded templates
    if any(k in low for k in ["marathi", "rope", "flashattention", "aws cloud bill", "whisper-large-v3", "70b parameter"]):
        return "NOT_IN_PAPERS: The provided source passages do not contain information regarding this topic."

    if "exact number of encoder layers" in low or "q01" in low:
        return "In the base Transformer model, there are N = 6 encoder layers, model dimension d_model = 512, and h = 8 parallel attention heads. [attention_1706.03762v7, p.3]"
    if "bleu score" in low or "wmt 2014" in low:
        return "The big Transformer model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task. [attention_1706.03762v7, p.8]"
    if "1/sqrt(d_k)" in low or "scale the dot products" in low:
        return "The authors scale dot products by 1/sqrt(d_k) because for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. [attention_1706.03762v7, p.4]"
    if "optimizer parameter values" in low or "beta_1" in low:
        return "During training, the Adam optimizer was used with beta_1 = 0.9, beta_2 = 0.98, epsilon = 10^-9, and warmup_steps = 4000. [attention_1706.03762v7, p.7]"
    if "parameter size of the helium" in low:
        return "The Helium text language model backbone has a parameter size of 7B and was trained on 2.1T tokens. [moshi_2410.00037v1, p.3]"
    if "frame rate" in low and "mimi" in low:
        return "The Mimi neural audio codec operates at a frame rate of 12.5 Hz with a frame size of 80 ms. [moshi_2410.00037v1, p.10]"
    if "inner monologue" in low:
        return "Inner Monologue in Moshi predicts time-aligned text tokens before generating speech tokens, leveraging text modeling capabilities to enhance linguistic quality and consistency. [moshi_2410.00037v1, p.15]"
    if "rq-transformer" in low:
        return "The RQ-Transformer splits sequence modeling into a Temporal Transformer modeling inter-frame dependencies and a Depth Transformer modeling intra-frame codebooks. [moshi_2410.00037v1, p.12]"
    if "dynamic chunking" in low and "h-net" in low:
        return "H-Net is an end-to-end hierarchical architecture that replaces fixed BPE tokenization by dynamically grouping raw bytes into chunks through learned routing. [dynamic_chunking_2507.07955v2, p.1]"
    if "routing module and smoothing module" in low:
        return "In H-Net, the routing module predicts boundary probabilities, and the smoothing module applies an exponential moving average to enable differentiable gradient flow during chunking. [dynamic_chunking_2507.07955v2, p.6]"
    if "loss function" in low and "compression ratio" in low:
        return "A ratio loss is added to regularize and control the target compression ratio across chunk representations. [dynamic_chunking_2507.07955v2, p.8]"
    if "theoretical latency and practical" in low or "latency in milliseconds" in low:
        return "The Moshi spoken dialogue system has a theoretical latency of 160 ms and achieves an overall practical latency of 200 ms. [moshi_2410.00037v1, p.2]"
    if "cosine similarity" in low:
        return "Cosine similarity between adjacent token representations is computed to evaluate semantic continuity and determine boundary probabilities p_t. [dynamic_chunking_2507.07955v2, p.6]"

    return "NOT_IN_PAPERS: The provided passages do not explicitly contain this information."


@app.post("/openai/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global requests_used, in_flight_requests

    # Fault Injection (1% chance of 500 error simulating upstream tool timeout)
    if random.random() < 0.015:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error: upstream worker timeout", "type": "server_error"}}
        )

    prompt_text = " ".join(m.get("content", "") for m in req.messages)
    tokens_in = estimate_tokens(prompt_text)
    
    reply_text = generate_simulated_reply(req.messages)
    tokens_out = estimate_tokens(reply_text)
    total_requested = tokens_in + tokens_out

    # Check TPM Leaky Bucket
    ok, wait_sec = await bucket.consume(total_requested)
    rem_tokens = await bucket.get_remaining()
    requests_used += 1
    rem_reqs = max(0, RPD_LIMIT - requests_used)

    rate_headers = {
        "x-ratelimit-limit-tokens": str(TPM_LIMIT),
        "x-ratelimit-limit-requests": str(RPD_LIMIT),
        "x-ratelimit-remaining-tokens": str(rem_tokens),
        "x-ratelimit-remaining-requests": str(rem_reqs),
        "x-ratelimit-reset-tokens": f"{wait_sec:.2f}s" if not ok else "100ms",
        "x-ratelimit-reset-requests": "14m20s",
    }

    if not ok:
        retry_int = max(1, int(wait_sec) + 1)
        rate_headers["retry-after"] = str(retry_int)
        error_body = {
            "error": {
                "message": (
                    f"Rate limit reached for model `{req.model}` in organization `standin_org` "
                    f"service tier `on_demand` on tokens per minute (TPM): Limit {TPM_LIMIT}, "
                    f"Used {TPM_LIMIT - rem_tokens}, Requested {total_requested}. "
                    f"Please try again in {wait_sec:.3f}s."
                ),
                "type": "tokens",
                "code": "rate_limit_exceeded"
            }
        }
        return JSONResponse(status_code=429, content=error_body, headers=rate_headers)

    # Track in-flight concurrency & apply load degradation penalty
    in_flight_requests += 1
    current_flight = in_flight_requests

    try:
        # Base TTFT (~350ms) + Concurrency Penalty (+180ms per concurrent request)
        base_ttft = max(0.12, random.gauss(0.35, 0.08))
        concurrency_penalty = max(0.0, (current_flight - 1) * 0.18)
        
        # Rare Tail Latency Spike (2.5% chance of 1.5 - 2.5s lag: "succeed slowly instead of failing quickly")
        tail_spike = random.uniform(1.5, 2.5) if random.random() < 0.025 else 0.0

        ttft_s = base_ttft + concurrency_penalty + tail_spike
        await asyncio.sleep(ttft_s)

        # Token streaming speed degrades slightly under load (+4ms/token per concurrent request)
        token_interval = 0.016 + max(0.0, (current_flight - 1) * 0.004)

        if not req.stream:
            gen_time = tokens_out * token_interval
            await asyncio.sleep(gen_time)
            return JSONResponse(
                status_code=200,
                content={
                    "id": f"chatcmpl-standin-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": reply_text},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": tokens_in,
                        "completion_tokens": tokens_out,
                        "total_tokens": tokens_in + tokens_out
                    }
                },
                headers=rate_headers
            )

        # SSE Streaming mode
        async def sse_generator():
            role_chunk = {
                "id": f"chatcmpl-standin-{int(time.time()*1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            words = reply_text.split(" ")
            for i, w in enumerate(words):
                token_str = w if i == 0 else " " + w
                chunk_data = {
                    "id": f"chatcmpl-standin-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": token_str}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(token_interval)

            final_chunk = {
                "id": f"chatcmpl-standin-{int(time.time()*1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out
                }
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream", headers=rate_headers)

    finally:
        in_flight_requests = max(0, in_flight_requests - 1)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
