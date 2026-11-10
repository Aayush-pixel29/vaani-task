import os
import json
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
URL = "https://api.groq.com/openai/v1/chat/completions"

async def test_draft_and_critic():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    sample_context = """[Excerpt 1 | attention_1706.03762v7 | Page 3 | 3.1 Encoder and Decoder Stacks]
The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers.
The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
We employ a residual connection around each of the two sub-layers, followed by layer normalization.
To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512.
In the multi-head attention, we employ h = 8 parallel attention layers, or heads."""

    question = "In 'Attention Is All You Need', what are the exact number of encoder layers N, model dimension d_model, and attention heads h in the base Transformer model?"

    draft_prompt = f"""You are a precise technical research assistant answering questions using ONLY the provided excerpts.
Excerpts:
{sample_context}

Question: {question}

Instructions:
1. If the excerpts contain the answer, answer concisely and include citations in the format [Paper Name, p.X].
2. If the excerpts do NOT contain sufficient information, reply ONLY with: NOT_IN_PAPERS: <reason>.
3. Never guess or extrapolate beyond the text."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 3: Draft
        resp = await client.post(URL, headers=headers, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": draft_prompt}],
            "max_tokens": 150,
            "reasoning_effort": "low"
        })
        draft = resp.json()["choices"][0]["message"]["content"]
        print("DRAFT ANSWER:\n", draft)

        # Step 4: Critic
        critic_prompt = f"""You are a rigorous Fact-Checking Critic. Verify if the draft answer is 100% faithful to the provided excerpts and has valid citations.

Excerpts:
{sample_context}

Question: {question}
Draft Answer: {draft}

Rules:
- Check if every factual claim in the draft is directly stated in the excerpts.
- Check if citations match the excerpt page numbers.
- If the draft is faithful and properly cited, output:
VERDICT: PASS
FINAL_ANSWER: <the verified draft answer>
- If the draft has unsupported claims, incorrect citations, or hallucinated facts, output:
VERDICT: REVISED
FINAL_ANSWER: <corrected version strictly based on excerpts with correct citations, or NOT_IN_PAPERS if unsupported>"""

        resp_critic = await client.post(URL, headers=headers, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": critic_prompt}],
            "max_tokens": 200,
            "reasoning_effort": "low"
        })
        critic_out = resp_critic.json()["choices"][0]["message"]["content"]
        print("\nCRITIC OUTPUT:\n", critic_out)

if __name__ == "__main__":
    asyncio.run(test_draft_and_critic())

