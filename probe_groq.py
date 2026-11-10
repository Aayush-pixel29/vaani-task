"""Probe the real Groq API and log every call to results.csv (Table 1 data).

Measures:
  - first_token_ms = time to first *answer content* token (not the empty role
    chunk and not hidden reasoning chunks; gpt-oss streams reasoning first).
  - tokens_in / tokens_out from Groq's own usage block (stream_options.include_usage).
  - 429 behaviour: honours retry-after, counts retries, records recovery time.
"""
import os
import time
import json
import uuid
import asyncio
import httpx
from dotenv import load_dotenv
from telemetry import RequestLog, log_request

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3

SMALL = "Explain attention in one sentence."
LARGE = "Attention mechanism key query value " * 300  # ~1.6k prompt tokens


async def call_once(client, content, max_tokens):
    """One streamed HTTP call. Returns dict with timings, status, usage, headers."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "max_tokens": max_tokens,
        "reasoning_effort": REASONING_EFFORT,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    t0 = time.time()
    first_content_ms = -1.0
    usage, text = None, []
    async with client.stream("POST", URL, json=payload, headers=headers, timeout=30.0) as resp:
        rl = {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower() or "retry" in k.lower()}
        if resp.status_code != 200:
            body = (await resp.aread()).decode()
            return {"status_code": resp.status_code, "headers": rl, "body": body,
                    "ms": (time.time() - t0) * 1000}
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices"):
                c = d["choices"][0].get("delta", {}).get("content")
                if c:
                    if first_content_ms < 0:
                        first_content_ms = (time.time() - t0) * 1000
                    text.append(c)
            u = d.get("usage") or (d.get("x_groq") or {}).get("usage")
            if u:
                usage = u
    return {"status_code": 200, "headers": rl, "first_ms": first_content_ms,
            "ms": (time.time() - t0) * 1000, "usage": usage or {}, "text": "".join(text)}


async def probe(client, scenario, content, max_tokens=200):
    """Call with 429-aware retries; log one row to results.csv."""
    start = time.time()
    retries = 0
    while True:
        r = await call_once(client, content, max_tokens)
        if r["status_code"] == 429 and retries < MAX_RETRIES:
            wait = float(r["headers"].get("retry-after", 2))
            print(f"  [{scenario}] 429 -> retry-after={wait}s | {r['body'][:160]}")
            retries += 1
            await asyncio.sleep(wait)
            continue
        break

    total_ms = (time.time() - start) * 1000
    if r["status_code"] == 200:
        status = "ok"
        # first token measured from the ORIGINAL request start (user-visible), incl. retry waits
        first_ms = (total_ms - r["ms"]) + r["first_ms"] if r["first_ms"] >= 0 else -1
        u = r["usage"]
        tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    else:
        status = "rate_limited" if r["status_code"] == 429 else "error"
        first_ms, tin, tout, reasoning = -1, 0, 0, 0

    log_request(RequestLog(
        request_id=str(uuid.uuid4())[:8], scenario=scenario, source="real",
        start_time=round(start, 3), first_token_ms=round(first_ms, 1),
        total_ms=round(total_ms, 1), status=status, retries=retries,
        tokens_in=tin, tokens_out=tout, retrieval_setting="none", answer_grade="n/a",
    ))
    print(f"  [{scenario}] {status} | first_token={first_ms:.0f}ms total={total_ms:.0f}ms "
          f"retries={retries} in={tin} out={tout} (reasoning={reasoning}) "
          f"rem-tok={r['headers'].get('x-ratelimit-remaining-tokens')}")
    return r


async def main():
    print(f"Model={MODEL} reasoning_effort={REASONING_EFFORT}")
    async with httpx.AsyncClient() as client:
        print("--- 1. Baseline sequential (5 small requests) ---")
        for i in range(5):
            r = await probe(client, "probe_baseline", SMALL)
            if i == 0:
                print("  Rate-limit headers:", json.dumps(r["headers"]))
            await asyncio.sleep(1.0)

        print("\n--- 2. Burst of 12 large requests (TPM exhaustion + recovery) ---")
        await asyncio.gather(*[probe(client, "probe_burst", LARGE, max_tokens=50) for _ in range(12)])


if __name__ == "__main__":
    asyncio.run(main())
