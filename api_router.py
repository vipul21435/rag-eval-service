"""
REST API 模块（使用FastAPI实现）
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import asyncio
from contextlib import asynccontextmanager
from version import __version__

# 从重构后的模块导入
from config import SILICONFLOW_API_KEY, MAGICK_API_KEY, is_configured_api_key
from core.generator import query_answer
from core.ingest import SourceFile, ingest_files
from core.vector_store import vector_store
from features.web_search import check_serpapi_key
from utils.network import is_port_available

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API 服务启动")
    yield
    logger.info("API 服务已关闭")


app = FastAPI(
    title="本地RAG API服务",
    description="提供基于本地大模型、云端模型服务和SERPAPI的文档问答API接口",
    version=__version__,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    enable_web_search: bool = False
    model_choice: str = "siliconflow"


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class FileProcessResult(BaseModel):
    status: str
    message: str
    file_info: Optional[Dict[str, Any]] = None


@app.post("/api/upload", response_model=FileProcessResult)
async def upload_file(file: UploadFile = File(...)):
    """处理文档并存入向量数据库"""
    filename = file.filename or "upload"
    suffix = os.path.splitext(filename)[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        report = await asyncio.to_thread(
            ingest_files, [SourceFile(path=Path(tmp_path), name=filename)]
        )
        result = report.files[0]
        if result.ok:
            message = f"{filename}: indexed {result.chunks} chunk(s)"
        else:
            message = f"{filename}: {result.error}"
        return {
            "status": "success" if result.ok else "error",
            "message": message,
            "file_info": {"filename": filename, "chunks": result.chunks}
        }
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        raise HTTPException(500, f"文档处理失败: {str(e)}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """问答接口"""
    if not req.question:
        raise HTTPException(400, "问题不能为空")
    try:
        answer = await asyncio.to_thread(query_answer, req.question, req.enable_web_search, req.model_choice)
        sources = []
        url_matches = re.findall(r'\[(网络来源|本地文档):[^\]]+\]\s*(?:\(URL:\s*([^)]+)\))?', answer)
        for source_type, url in url_matches:
            sources.append({"type": source_type, "url": url} if url else {"type": source_type})

        return {
            "answer": answer, "sources": sources,
            "metadata": {"enable_web_search": req.enable_web_search, "model": req.model_choice}
        }
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(500, f"问答处理失败: {str(e)}") from e


@app.get("/api/status")
async def check_status():
    return {
        "status": "healthy",
        "siliconflow_configured": is_configured_api_key(SILICONFLOW_API_KEY),
        "magick_configured": is_configured_api_key(MAGICK_API_KEY),
        "serpapi_configured": check_serpapi_key(),
        "vector_store_ready": vector_store.is_ready,
        "total_chunks": vector_store.total_chunks,
        "version": __version__
    }


if __name__ == "__main__":
    import uvicorn
    port = next((p for p in [17995, 17996, 17997, 17998, 17999] if is_port_available(p)), 17995)
    logger.info(f"启动API服务，端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
