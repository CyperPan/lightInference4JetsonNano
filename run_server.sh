#!/bin/bash
# 在线推理服务启动脚本（默认端口 8001，避免 8000 被占用）
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "请先创建虚拟环境并安装依赖："
  echo "  python3 -m venv .venv && source .venv/bin/activate"
  echo "  pip install \"numpy<2\" \"scipy>=1.10\" -r requirements.txt"
  exit 1
fi
source .venv/bin/activate
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
export PORT="${PORT:-8001}"
echo "模型: $MODEL_NAME  端口: $PORT"
python -m server.app
