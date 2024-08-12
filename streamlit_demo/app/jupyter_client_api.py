import httpx
from langchain.pydantic_v1 import BaseModel


class JupyterClient(BaseModel):
    base_url: str

    def execute(self, code):
        r = httpx.post(f"{self.base_url}/code", json={"script": code}, timeout=60.0)
        return r.json()


if __name__ == "__main__":
    script = """
    print('gdfgfdf')
    """
    response = JupyterClient().execute(script)
    print(response)
