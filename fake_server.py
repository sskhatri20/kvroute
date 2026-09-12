import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


async def fake_stream():
    for word in ["hello", "from", "fake", "model"]:
        print(word)
        yield word.encode()
        await asyncio.sleep(0.5)


@app.post("/v1/chat/completions")
async def fake_chat():
    return StreamingResponse(fake_stream())
