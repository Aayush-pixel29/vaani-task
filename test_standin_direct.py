import asyncio
import httpx
from standin_server import app

async def test_standin():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://standin") as client:
        # Test 1: Simple completion
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": "Route the user question"},
                {"role": "user", "content": "How many layers in Attention Is All You Need?"}
            ],
            "stream": True
        }
        resp = await client.post("/openai/v1/chat/completions", json=payload)
        print("Status code:", resp.status_code)
        print("Rate limit headers:", {k: v for k, v in resp.headers.items() if "ratelimit" in k})
        
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                print("SSE Chunk:", line[:60])
                break

if __name__ == "__main__":
    asyncio.run(test_standin())

