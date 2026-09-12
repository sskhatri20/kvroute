import asyncio
import httpx


async def main():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8001/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        ) as response:
            async for chunk in response.aiter_bytes():
                print(chunk)


asyncio.run(main())