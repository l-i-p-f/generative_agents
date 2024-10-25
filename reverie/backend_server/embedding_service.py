from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel  # 数据模型自动验证及异常处理框架
from sentence_transformers import SentenceTransformer
import uvicorn  # 异步服务器框架

# 加载模型
model_name = "bge-m3"
model = SentenceTransformer(r'D:\model\bge-m3')

app = FastAPI()


# 定义请示体数据模型
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]


# 定义返回体数据模型
class EmbeddingResponse(BaseModel):
    object: str
    data: List[dict]
    model: str
    usage: dict


@app.get("/")
async def root():
    return {"message": "Embedding service is up and running!"}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embedding(request: EmbeddingRequest):
    text = request.input
    if isinstance(text, list):
        embeddings = model.encode(text).tolist()
    else:
        embeddings = model.encode([text]).tolist()[0]

    response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embeddings,
                "index": 0
            }

        ],
        "model": model_name,
        "usage": {
            "prompt_tokens": len(text.split()) if isinstance(text, str) else sum(len(t.split()) for t in text),
            "total_tokens": len(text.split()) if isinstance(text, str) else sum(len(t.split()) for t in text),
        }
    }
    return response


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
