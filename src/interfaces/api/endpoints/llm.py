"""
LLM APIエンドポイント - LLMエンジンへのAPIアクセスを提供

このモジュールは、UIからLLMエンジンへのアクセスを提供するためのFastAPIエンドポイントを実装します。
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import asyncio
import json

from pydantic import BaseModel, Field

# LLMエンジンをインポート
from src.core.llm.engine import LLMEngine

router = APIRouter(
    prefix="/llm",
    tags=["LLM"],
    responses={404: {"description": "Not found"}},
)

# リクエスト・レスポンスモデルの定義
class GenerateRequest(BaseModel):
    """テキスト生成リクエスト"""
    prompt: str
    provider: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    
class GenerateResponse(BaseModel):
    """テキスト生成レスポンス"""
    text: str
    
class ChatMessage(BaseModel):
    """チャットメッセージ"""
    role: str
    content: str
    
class ChatRequest(BaseModel):
    """チャット生成リクエスト"""
    messages: List[ChatMessage]
    provider: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    
class ChatResponse(BaseModel):
    """チャット生成レスポンス"""
    response: str
    
class EmbedRequest(BaseModel):
    """埋め込み生成リクエスト"""
    text: str
    provider: Optional[str] = None
    
class EmbedResponse(BaseModel):
    """埋め込み生成レスポンス"""
    embedding: List[float]

# LLMエンジンインスタンスの取得（依存性注入）
async def get_llm_engine():
    """LLMエンジンインスタンスを取得"""
    # 実際の実装では、アプリケーションの起動時に一度だけ初期化するべき
    engine = LLMEngine()
    try:
        yield engine
    finally:
        # クリーンアップが必要な場合
        pass

# エンドポイント定義
@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    llm_engine: LLMEngine = Depends(get_llm_engine)
):
    """テキスト生成エンドポイント"""
    try:
        text = await llm_engine.generate(
            prompt=request.prompt,
            provider=request.provider,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return GenerateResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation error: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_engine: LLMEngine = Depends(get_llm_engine)
):
    """チャット形式でのテキスト生成エンドポイント"""
    try:
        # メッセージ形式をLLMエンジンの期待する形式に変換
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # LLMエンジンにリクエスト
        result = await llm_engine.chat(
            messages=messages,
            provider=request.provider,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # レスポンスの抽出
        response = result.get("content", "No response generated")
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM chat error: {str(e)}")

@router.post("/embed", response_model=EmbedResponse)
async def embed_text(
    request: EmbedRequest,
    llm_engine: LLMEngine = Depends(get_llm_engine)
):
    """テキスト埋め込みベクトル生成エンドポイント"""
    try:
        embedding = await llm_engine.embed(
            text=request.text,
            provider=request.provider
        )
        return EmbedResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM embedding error: {str(e)}")
