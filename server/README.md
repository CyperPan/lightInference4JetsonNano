# 在线推理服务

基于 FastAPI 的 HTTP 推理接口，使用 HuggingFace Transformers 加载 Llama/Qwen 等模型。

## 安装

```bash
pip install -r requirements.txt
```

若遇 numpy/scipy 版本冲突，建议使用虚拟环境并安装兼容版本：
```bash
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## 启动

```bash
# 使用 HF 模型名（会按需下载）
export MODEL_NAME=Qwen/Qwen2.5-0.5B
python -m server.app

# 使用本地已下载目录
export MODEL_NAME=/path/to/Qwen/Qwen2.5-0.5B
python -m server.app

# 默认端口 8001；指定端口与 host
HOST=0.0.0.0 PORT=8080 python -m server.app
```

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /generate | 文本生成 |

**POST /generate 请求体：**
```json
{
  "prompt": "你好",
  "max_new_tokens": 128,
  "do_sample": false,
  "repetition_penalty": 1.0,
  "temperature": 1.0,
  "skip_special_tokens": true
}
```

## 调用示例

```bash
curl -X POST http://127.0.0.1:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好","max_new_tokens":64}'
```
