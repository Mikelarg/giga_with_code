from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from jupyter_client import AsyncKernelManager

from app.run_jupyter import async_run_code

load_dotenv()

app = FastAPI()


class CodeRequest(BaseModel):
    script: str


@app.post("/code")
async def code(request: CodeRequest):
    km = AsyncKernelManager()
    await km.start_kernel()
    response = await async_run_code(km, request.script)
    return {
        "result": response[0],
        "is_exception": bool(response[1]),
        "exception": response[1],
        "attachments": response[3],
    }
