# resume_load_tests.py
import os
import time
import random
import asyncio
import httpx
from pipeline import RelayPipeline
from questions import EVAL_QUESTIONS
from standin_server import app, bucket, TPM_LIMIT

QUESTIONS_CYCLE = EVAL_QUESTIONS

async def run_scenario_sustained_overload(client, pipeline):
    print("\n==================================================", flush=True)
    print("SCENARIO 3: SUSTAINED OVERLOAD (40 requests over 60s)", flush=True)
    print("==================================================", flush=True)
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

    tasks = [send_overload_req(i, i * 1.5) for i in range(40)]
    results = await asyncio.gather(*tasks)
    print(f"  Sustained overload completed: {len(results)} requests executed.", flush=True)

async def run_scenario_limit_halved(client, pipeline):
    print("\n==================================================", flush=True)
    print("SCENARIO 4: PER-MINUTE LIMIT HALVED (4,000 TPM, 20 requests)", flush=True)
    print("==================================================", flush=True)
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
            await asyncio.sleep(3.5)

async def run_scenario_daily_exhausted(client, pipeline):
    print("\n==================================================", flush=True)
    print("SCENARIO 5: DAILY BUDGET EXHAUSTED (RPD=0 at 3:00 PM, 15 requests)", flush=True)
    print("==================================================", flush=True)
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
        await asyncio.sleep(0.3)

async def main():
    print("==================================================================", flush=True)
    print("RESUMING LOAD SCENARIOS: Scenarios 3, 4, and 5", flush=True)
    print("==================================================================", flush=True)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://standin") as client:
        pipeline = RelayPipeline(
            base_url="/openai/v1/chat/completions",
            source_label="stand-in",
            default_setting="512w_top5_norerank",
            adaptive_retrieval=True,
            max_retries=2
        )

        await run_scenario_sustained_overload(client, pipeline)
        await run_scenario_limit_halved(client, pipeline)
        await run_scenario_daily_exhausted(client, pipeline)

    print("\n==================================================================", flush=True)
    print("ALL 5 LOAD SCENARIOS 100% COMPLETE!", flush=True)
    print("==================================================================", flush=True)

if __name__ == "__main__":
    asyncio.run(main())

