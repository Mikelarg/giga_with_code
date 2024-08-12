from langchain.pydantic_v1 import BaseModel, Field

from jupyter_client_api import JupyterClient
from langchain_core.tools import StructuredTool
import re
import os


class CodeInput(BaseModel):
    code: str = Field(..., description="Код Python")


client = JupyterClient(
    base_url=os.getenv("JUPYTER_CLIENT_API", "http://127.0.0.1:9090")
)

INPUT_REGEX = re.compile(r"input\(.+?\)")
FILE_NOT_FOUND_REGEX = re.compile(r"FileNotFoundError:.+?No such file or directory")


def execute(code: str):
    if INPUT_REGEX.search(code):
        return {
            "message": (
                "Перепиши код без использования функции input. "
                "Сгенерируй синтетические данные сам"
            ),
            "attachments": [],
            "is_exception": True,
        }
    response = client.execute("import pandas as pd\nimport numpy as np\n" + code)
    result = response["result"]
    results = []
    if result is not None:
        results.append(result.strip())
    if len(response["attachments"]):
        for attachment in response["attachments"]:
            if "application/vnd.plotly.v1+json" in attachment:
                results.append("В результате выполнения был сгенерирован график")
            elif "image/png" in attachment:
                results.append("В результате выполнения было сгенерировано изображение")
    result = "\n".join(results)
    if response["is_exception"]:
        if FILE_NOT_FOUND_REGEX.search(response["exception"]):
            message = (
                "Перепиши код без открытия файлов. "
                "Сгенерируй синтетические данные исходя из задачи."
            )
        else:
            # Убираем лишние строки кода из ошибки, для улучшения качества исправления
            exc = re.sub(
                r"(.+?\/.+?py.+\n(.+\n)+\n)", "", response["exception"], 0, re.MULTILINE
            )
            message = (
                f'Во время исполнения кода произошла ошибка: "{exc}".\n'
                "Исправь код и напиши полностью исправленный код."
            )
    else:
        message = (
            f'Результат выполнения: "{result.strip()}".\n'
            "Проверь результат выполнения. "
            "Если он правильный, используй результат в своем ответе "
            "(не пиши код и не пиши изображения)."
        )
    return {
        "message": message,
        "attachments": response["attachments"],
        "is_exception": response["is_exception"],
    }


repl = StructuredTool(
    name="python",
    description=(
        "Компилятор ipython. Возвращает результат выполнения. "
        "Если произошла ошибка напиши исправленный код "
    ),
    func=execute,
    args_schema=CodeInput,  # type: ignore
)

tools = [repl]
