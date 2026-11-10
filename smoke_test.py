import asyncio
import time
import json
import httpx
from pipeline import RelayPipeline
from questions import EVAL_QUESTIONS

async def run_smoke():
    pipeline = RelayPipeline(
        source_label="real",
        default_setting="512w_top5_norerank",
        adaptive_retrieval=True
    )
    test_ids = ["Q01", "Q12", "Q15"]
    questions_to_test = [q for q in EVAL_QUESTIONS if q["id"] in test_ids]

    print("==================================================")
    print("STARTING 3-QUESTION SMOKE TEST ON REAL GROQ API")
    print(f"Model: {pipeline.default_setting} | Adaptive: True")
    print("==================================================\n")

    async with httpx.AsyncClient() as client:
        for idx, q_meta in enumerate(questions_to_test):
            qid = q_meta["id"]
            question = q_meta["question"]
            print(f"[{qid}] Running question: {question}")
            
            res = await pipeline.answer_question(
                client=client,
                request_id=f"smoke_{qid.lower()}",
                question=question,
                scenario="smoke_test",
                question_meta=q_meta
            )

            print(f"[{qid}] Result:")
            print(f"  Grade: {res.get('grade')} | Retries: {res.get('retries')}")
            print(f"  First Token (Verified Release): {res.get('first_token_ms', 0):.1f}ms")
            print(f"  Draft Raw TTFT: {res.get('draft_raw_ttft_ms', 0):.1f}ms")
            print(f"  Total Duration: {res.get('total_ms', 0):.1f}ms")
            print(f"  Tokens In: {res.get('tokens_in')} | Tokens Out: {res.get('tokens_out')}")
            print(f"  Answer:\n    {res.get('answer')}")
            print("  Traces Breakdown:")
            for tr in res.get("traces", []):
                print(f"    - {tr['step_name']}: work={tr['work_ms']:.1f}ms, retry={tr['retry_ms']:.1f}ms, tokens={tr['tokens_in']}+{tr['tokens_out']}")

            if idx < len(questions_to_test) - 1:
                print("\n[Pacing] Sleeping 25 seconds to respect 8,000 TPM limit...")
                await asyncio.sleep(25.0)
                print()

    print("\n==================================================")
    print("SMOKE TEST COMPLETE!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_smoke())

