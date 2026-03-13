import argparse
import json
import sys


def _print_flush(s: str):
    sys.stdout.write(s)
    sys.stdout.flush()


def chat_loop(base_url: str, max_new_tokens: int, do_sample: bool, temperature: float):
    try:
        import requests
    except Exception:
        raise SystemExit("缺少依赖 requests，请先 pip install requests")

    endpoint = base_url.rstrip("/") + "/generate_stream"
    _print_flush(f"连接到: {endpoint}\n")
    _print_flush("输入内容回车发送，输入 /exit 退出。\n\n")

    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
        }
        with requests.post(endpoint, json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            _print_flush("assistant: ")
            buf = ""
            for raw in r.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip("\r")
                if not line:
                    # event separator
                    if buf.startswith("data: "):
                        data = buf[len("data: ") :]
                        # 服务端 start/done 会发送字典字符串，这里仅用于调试时打印
                        if data.startswith("{") and data.endswith("}"):
                            pass
                        else:
                            _print_flush(data)
                    buf = ""
                    continue
                # 只关心 data 行；delta/done/error 事件都走 data
                if line.startswith("data: "):
                    buf = line
            _print_flush("\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8002", help="服务地址（不含路径）")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    chat_loop(
        base_url=args.base_url,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

