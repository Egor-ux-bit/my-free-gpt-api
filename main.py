import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx

app = FastAPI(
    title="Free GPT API",
    description="Бесплатный API-интерфейс для генерации текста",
    version="1.0.0"
)

class Message(BaseModel):
    role: str  # "user" или "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "free-model"

@app.get("/")
async def root():
    return {"status": "online", "message": "API работает! Используйте POST на /v1/chat/completions"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Список сообщений (messages) не может быть пустым")

    # Берём последнее сообщение пользователя
    user_message = request.messages[-1].content

    # Публичный бесплатный эндпоинт для демонстрации (Hugging Face Serverless API)
    # Используем одну из лучших открытых моделей
    HF_API_URL = "https://huggingface.co"
    
    # Рекомендуется зарегистрироваться на Hugging Face (это бесплатно) и вставить токен в переменные окружения,
    # чтобы избежать жестких лимитов на количество запросов:
    HF_TOKEN = os.getenv("HF_TOKEN", "") 
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    payload = {
        "inputs": f"<s>[INST] {user_message} [/INST]",
        "parameters": {"max_new_tokens": 512, "temperature": 0.7}
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(HF_API_URL, json=payload, headers=headers, timeout=30.0)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Ошибка внешнего провайдера: {response.text}"
                )
            
            result = response.json()
            
            # Извлекаем сгенерированный текст (у HF формат ответа — список словарей)
            if isinstance(result, list) and len(result) > 0:
                full_text = result[0].get("generated_text", "")
                # Отрезаем промпт, если модель вернула его вместе с ответом
                generated_text = full_text.split("[/INST]")[-1].strip()
            else:
                generated_text = "Не удалось разобрать ответ от модели."

            # Возвращаем стандартный JSON-ответ в стиле OpenAI API
            return {
                "id": "chatcmpl-free-api",
                "object": "chat.completion",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }
                ]
            }

        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Ошибка сети при запросе к ИИ: {exc}")
