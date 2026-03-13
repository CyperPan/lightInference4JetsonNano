"""
在线模型推理服务：基于 HuggingFace Transformers 的 HTTP API 部署。
支持 Llama、Qwen 等 CausalLM 模型，通过环境变量指定模型路径。
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 延迟导入，避免启动时未安装 transformers 时报错
def _get_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


_model_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_tokenizer
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")
    print(f"正在加载模型: {model_name} ...")
    _model_tokenizer = _get_model_and_tokenizer()
    print("模型加载完成，服务就绪。")
    yield
    _model_tokenizer = None


app = FastAPI(title="KuiperLLama 在线推理", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="输入文本/提示")
    max_new_tokens: int = Field(128, ge=1, le=2048, description="最大生成 token 数")
    do_sample: bool = Field(False, description="是否采样")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="重复惩罚")
    temperature: float = Field(1.0, ge=0.01, le=2.0, description="采样温度（do_sample=True 时有效）")
    skip_special_tokens: bool = Field(True, description="解码时是否跳过特殊 token")


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    max_new_tokens: int


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "model_loaded": _model_tokenizer is not None}


def _sse_event(data: str, event: str | None = None) -> str:
    # SSE 协议：每个事件以空行分隔
    lines = []
    if event:
        lines.append(f"event: {event}")
    for ln in str(data).splitlines() or [""]:
        lines.append(f"data: {ln}")
    return "\n".join(lines) + "\n\n"


@app.post("/generate_stream")
def generate_stream(req: GenerateRequest):
    """流式文本生成接口（SSE），逐步返回生成的 token 文本片段。"""
    if _model_tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    model, tokenizer = _model_tokenizer

    # 延迟导入避免服务启动时强依赖 torch/transformers 子模块
    from transformers import TextIteratorStreamer
    import threading

    def event_iter():
        try:
            inputs = tokenizer(req.prompt, return_tensors="pt")
            inputs = inputs.to(model.device)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=req.skip_special_tokens,
            )
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
                repetition_penalty=req.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                streamer=streamer,
            )
            if req.do_sample:
                gen_kwargs["temperature"] = req.temperature

            t = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
            t.start()

            yield _sse_event({"type": "start"})
            for new_text in streamer:
                if new_text:
                    yield _sse_event(new_text, event="delta")
            yield _sse_event({"type": "done"}, event="done")
        except Exception as e:
            yield _sse_event(str(e), event="error")

    return StreamingResponse(event_iter(), media_type="text/event-stream")


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """文本生成接口"""
    if _model_tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    model, tokenizer = _model_tokenizer
    try:
        inputs = tokenizer(req.prompt, return_tensors="pt")
        inputs = inputs.to(model.device)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=req.do_sample,
            repetition_penalty=req.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
        if req.do_sample:
            gen_kwargs["temperature"] = req.temperature
        pred = model.generate(**gen_kwargs)
        text = tokenizer.decode(pred.cpu()[0], skip_special_tokens=req.skip_special_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return GenerateResponse(text=text, prompt=req.prompt, max_new_tokens=req.max_new_tokens)


if __name__ == "__main__":
    import socket
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    # 未设置 PORT 或 PORT=0 时，在 8001~8010 中自动选可用端口，避免 address already in use
    port_cfg = os.environ.get("PORT", "0")
    port = int(port_cfg)
    if port <= 0:
        for p in range(8001, 8011):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((host if host != "0.0.0.0" else "127.0.0.1", p))
                port = p
                sock.close()
                break
            except OSError:
                sock.close()
        else:
            port = 8001
    print(f"服务地址: http://127.0.0.1:{port}")
    uvicorn.run(app, host=host, port=port)
