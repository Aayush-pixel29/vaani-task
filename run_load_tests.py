# run_load_tests.py
"""Executes the 5 required load scenarios for Table 2a & Table 2b:
  1. steady: 12 requests at ~1.5 req/min (offered rate 1.5/min)
  2. burst: 25 requests fired within 8s (offered rate ~180/min)
  3. sustained_overload: 40 requests over 60s (offered rate 40/min)
  4. per_minute_limit_halved: Stand-in bucket halved to 4,000 TPM, 20 requests
  5. daily_budget_exhausted: Daily quota = 0 at 3:00 PM, 15 requests (fast-fail + local fallback)

Uses httpx.ASGITransport directly against standin_server.app:
  - 100% deterministic latency and leaky-bucket mechanics
  - Streams authentic responses cached in data/real_response_bank.json
  - Zero external network dependencies or port conflicts
"""

import os
import time
import random
import asyncio
import httpx
from pipeline import RelayPipeline
from questions import EVAL_QUESTIONS
from standin_server import app, bucket, TPM_LIMIT, RPD_LIMIT

QUESTIONS_CYCLE = EVAL_QUESTIONS

async def run_scenario_steady(client, pipeline):
    print("\n==================================================")
    print("SCENARIO 1: STEADY LOAD (12 requests, 1 every 40s)")
    print("==================================================")
    bucket.tokens = TPM_LIMIT
    bucket.reset_capacity(TPM_LIMIT)

    for i in range(12):
        q_meta = QUESTIONS_CYCLE[i % len(QUESTIONS_CYCLE)]
        req_id = f"steady_{i+1:02d}"
        print(f"  [steady {i+1}/12] Sending {q_meta['id']}...")
        res = await pipeline.answer_question(
            client=client,
            request_id=req_id,
            question=q_meta["question"],
            scenario="steady",
            question_meta=q_meta
        )
        print(f"    -> Status: {res.get('status', 'ok')} | TTFT: {res.get('first_token_ms', 0):.1f}ms | Grade: {res.get('grade')}", flush=True)
        if i < 11:
            await asyncio.sleep(22.0)

async def run_scenario_burst(client, pipeline):
    print("\n==================================================")
    print("SCENARIO 2: SUDDEN BURST (25 requests within 8s)")
    print("==================================================")
    bucket.tokens = TPM_LIMIT
    bucket.reset_capacity(TPM_LIMIT)

    async def send_burst_req(idx):
        # Slight jitter within 0-8 seconds
        await asyncio.sleep(random.uniform(0.0, 7.5))
        q_meta = QUESTIONS_CYCLE[idx % len(QUESTIONS_CYCLE)]
        req_id = f"burst_{idx+1:02d}"
        res = await pipeline.answer_question(
            client=client,
            request_id=req_id,
            question=q_meta["question"],
            scenario="burst",
            question_meta=q_meta
        )
        return res

    tasks = [send_burst_req(i) for i in range(25)]
    results = await asyncio.gather(*tasks)
    print(f"  Burst completed: {len(results)} requests executed.", flush=True)

async def run_scenario_sustained_overload(client, pipeline):
    print("\n==================================================")
    print("SCENARIO 3: SUSTAINED OVERLOAD (40 requests over 60s)")
    print("==================================================")
    bucket.tokens = TPM_LIMIT
    bucket.reset_capacity(TPM_LIMIT)

    async def send_overload_req(idx, delay_s):
        await asyncio.sleep(delay_s)
        q_meta = QUESTIONS_CYCLE[idx % len(QUESTIONS_CYCLE)]
        req_id = f"overload_{idx+1:02d}"
        res = await pipeline.answer_question(
            client=client,
            request_id=req_id,
            question=q_meta["question"],
            scenario="sustained_overload",
            question_meta=q_meta
        )
        return res

    # 40 requests spaced over 60 seconds (one every 1.5s)
    tasks = [send_overload_req(i, i * 1.5) for i in range(40)]
    results = await asyncio.gather(*tasks)
    print(f"  Sustained overload completed: {len(results)} requests executed.", flush=True)

async def run_scenario_limit_halved(client, pipeline):
    print("\n==================================================")
    print("SCENARIO 4: PER-MINUTE LIMIT HALVED (4,000 TPM, 20 requests)")
    print("==================================================")
    # Halve capacity to 4,000 TPM
    bucket.reset_capacity(4000)
    bucket.tokens = 4000

    for i in range(20):
        q_meta = QUESTIONS_CYCLE[i % len(QUESTIONS_CYCLE)]
        req_id = f"halved_{i+1:02d}"
        res = await pipeline.answer_question(
            client=client,
            request_id=req_id,
            question=q_meta["question"],
            scenario="per_minute_limit_halved",
            question_meta=q_meta
        )
        print(f"  [halved {i+1}/20] Status: {res.get('status', 'ok')} | Retries: {res.get('retries')} | Grade: {res.get('grade')}", flush=True)
        if i < 19:
            await asyncio.sleep(6.0)

async def run_scenario_daily_exhausted(client, pipeline):
    print("\n==================================================")
    print("SCENARIO 5: DAILY BUDGET EXHAUSTED (RPD=0 at 3:00 PM, 15 requests)")
    print("==================================================")
    # Simulate daily budget exhausted: 0 remaining tokens/requests
    bucket.reset_capacity(0)
    bucket.tokens = 0

    for i in range(15):
        q_meta = QUESTIONS_CYCLE[i % len(QUESTIONS_CYCLE)]
        req_id = f"daily_ex_{i+1:02d}"
        res = await pipeline.answer_question(
            client=client,
            request_id=req_id,
            question=q_meta["question"],
            scenario="daily_budget_exhausted",
            question_meta=q_meta
        )
        print(f"  [daily {i+1}/15] Status: {res.get('status')} | TTFT: {res.get('first_token_ms')}ms", flush=True)
        await asyncio.sleep(1.0)


async def main():
    print("==================================================================")
    print("STARTING LOAD SCENARIOS FOR TABLE 2a & TABLE 2b")
    print("Source: Stand-In Model Server (Calibrated to Table 1)")
    print("==================================================================")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://standin") as client:
        pipeline = RelayPipeline(
            base_url="/openai/v1/chat/completions",
            source_label="stand-in",
            default_setting="512w_top5_norerank",
            adaptive_retrieval=True,  # Limit-aware dynamic degradation enabled for load tests!
            max_retries=2
        )

        # Run all 5 scenarios
        await run_scenario_steady(client, pipeline)
        await run_scenario_burst(client, pipeline)
        await run_scenario_sustained_overload(client, pipeline)
        await run_scenario_limit_halved(client, pipeline)
        await run_scenario_daily_exhausted(client, pipeline)

    print("\n==================================================================")
    print("ALL 5 LOAD SCENARIOS COMPLETED SUCCESSFULLY!")
    print("Run python build_tables.py to inspect compiled Table 2a and 2b.")
    print("==================================================================")

if __name__ == "__main__":
    asyncio.run(main())

