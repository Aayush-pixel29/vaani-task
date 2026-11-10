# run_table3.py
"""Executes the Table 3 Retrieval Evaluation across all 3 settings and 18 questions on real Groq.

Pacing is tuned to leaky-bucket refill math (133.3 tokens/sec):
  - 256w_top3_norerank: ~1,200 prompt tokens -> 20s sleep
  - 512w_top5_norerank: ~4,200 prompt tokens -> 35s sleep
  - 512w_top5_rerank3:  ~2,800 prompt tokens -> 26s sleep

Saves all grounded answers into data/real_response_bank.json for the Stand-In server.
"""

import os
import json
import time
import asyncio
import httpx
from pipeline import RelayPipeline
from questions import EVAL_QUESTIONS

RESPONSE_BANK_PATH = "data/real_response_bank.json"

SETTINGS_TO_RUN = [
    ("256w_top3_norerank", 20.0),
    ("512w_top5_norerank", 35.0),
    ("512w_top5_rerank3", 26.0),
]

async def main():
    print("==================================================================")
    print("STARTING TABLE 3 EVALUATION (18 Questions x 3 Retrieval Settings)")
    print("Target: 54 real Groq requests | scenario='retrieval_eval'")
    print("==================================================================\n")

    pipeline = RelayPipeline(
        source_label="real",
        adaptive_retrieval=False,  # Strict isolation per setting (no auto-downgrade)
        max_retries=2
    )

    response_bank = {}
    if os.path.exists(RESPONSE_BANK_PATH):
        try:
            with open(RESPONSE_BANK_PATH, "r", encoding="utf-8") as f:
                response_bank = json.load(f)
        except Exception:
            response_bank = {}

    total_runs = len(SETTINGS_TO_RUN) * len(EVAL_QUESTIONS)
    completed = 0

    async with httpx.AsyncClient() as client:
        for setting_name, pace_sec in SETTINGS_TO_RUN:
            print(f"\n>>> EVALUATING SETTING: {setting_name} (Pacing: {pace_sec}s) <<<")
            
            for q_idx, q_meta in enumerate(EVAL_QUESTIONS):
                qid = q_meta["id"]
                req_id = f"eval_{setting_name[:4]}_{qid.lower()}"
                
                print(f"[{completed+1}/{total_runs}] {setting_name} | {qid}: {q_meta['question'][:60]}...")
                
                res = await pipeline.answer_question(
                    client=client,
                    request_id=req_id,
                    question=q_meta["question"],
                    scenario="retrieval_eval",
                    retrieval_setting=setting_name,
                    question_meta=q_meta
                )

                grade = res.get("grade", "unknown")
                ft = res.get("first_token_ms", 0.0)
                tot = res.get("total_ms", 0.0)
                tin = res.get("tokens_in", 0)
                tout = res.get("tokens_out", 0)
                ans = res.get("answer", "")

                print(f"   -> Grade: {grade} | TTFT: {ft:.1f}ms | Total: {tot:.1f}ms | Tokens: {tin}+{tout}")

                # Cache good answers in response bank
                if qid not in response_bank or (grade in ["correct", "refused_correctly"]):
                    response_bank[qid] = {
                        "question": q_meta["question"],
                        "answer": ans,
                        "grade": grade,
                        "target_paper": q_meta.get("target_paper", "none"),
                        "tokens_in": tin,
                        "tokens_out": tout,
                    }
                    with open(RESPONSE_BANK_PATH, "w", encoding="utf-8") as f:
                        json.dump(response_bank, f, indent=2)

                completed += 1

                # Pace between calls if not the very last one
                if completed < total_runs:
                    print(f"   [Pacing] Cooling down {pace_sec}s for token refill...")
                    await asyncio.sleep(pace_sec)

    print("\n==================================================================")
    print("TABLE 3 EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Total requests logged to results.csv: {completed}")
    print(f"Response bank saved to: {RESPONSE_BANK_PATH}")
    print("==================================================================")

if __name__ == "__main__":
    asyncio.run(main())

