import argparse
import json
import re
import uuid
from datetime import datetime

import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from icecream import ic

app = Flask(__name__)
CORS(app)

# 解析命令行参数
parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str, default='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjMzNjA2MzgsInVzZXJuYW1lIjoiMjAyNDEzNDAyMiJ9.b4E5VzUxkn0Kc1pxkKVipybRFCw47NcppBognTD39e8',
                    help='GenAI API Access Token')
parser.add_argument('--port', type=int, default=5000,
                    help='Flask server port (default: 5000)')
args = parser.parse_args()

# GenAI API 配置
GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
GENAI_HEADERS = {
    "Accept": "*/*, text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://genai.shanghaitech.edu.cn",
    "Referer": "https://genai.shanghaitech.edu.cn/dialogue",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "X-Access-Token": args.token,
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

MODEL_SPECS = [
    {
        "public_id": "deepseek-r1",
        "request_id": "deepseek-r1:671b",
        "actual_id": "deepseek-r1:671b",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "deepseek-v3",
        "request_id": "deepseek-v3:671b",
        "actual_id": "deepseek-v3:671b",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "glm-5.1",
        "request_id": "chatglm",
        "actual_id": "glm-chat",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "minimax-m1",
        "request_id": "MiniMax-M1",
        "actual_id": "minimax",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "qwen3.5-397b-a17b",
        "request_id": "qwen-instruct",
        "actual_id": "qwen-instruct",
        "root_ai_type": "xinference",
    },
    {
        "public_id": "gpt-5.5",
        "request_id": "GPT-5.5",
        "actual_id": "gpt-5.5-2026-04-24",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-5.4",
        "request_id": "GPT-5.4",
        "actual_id": "gpt-5.4-2026-03-05",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-5.2",
        "request_id": "GPT-5.2",
        "actual_id": "gpt-5.2-2025-12-11",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-5",
        "request_id": "GPT-5",
        "actual_id": "gpt-5-2025-08-07",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-4.1",
        "request_id": "GPT-4.1",
        "actual_id": "gpt-4.1-2025-04-14",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-4.1-mini",
        "request_id": "GPT-4.1-mini",
        "actual_id": "gpt-4.1-mini-2025-04-14",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-o4-mini",
        "request_id": "o4-mini",
        "actual_id": "o4-mini-2025-04-16",
        "root_ai_type": "azure",
    },
    {
        "public_id": "gpt-o3",
        "request_id": "o3",
        "actual_id": "o3-2025-04-16",
        "root_ai_type": "azure",
    },
]


def build_model_alias_lookup():
    """构建模型别名查找表。

    将对外公开名称、上游请求名称和上游实际模型名称统一映射到同一份
    模型规格上，便于后续按任意别名解析。

    Returns:
        dict[str, dict]: 以小写别名为键、模型规格字典为值的查找表。
    """
    alias_lookup = {}
    for spec in MODEL_SPECS:
        for alias in {spec["public_id"], spec["request_id"], spec["actual_id"]}:
            alias_lookup[alias.lower()] = spec
    return alias_lookup


MODEL_ALIAS_LOOKUP = build_model_alias_lookup()


def resolve_model(model_name):
    """解析模型名称到上游请求参数。

    Args:
        model_name (Any): 调用方传入的模型名，可能是 public id、request id
            或 actual id。

    Returns:
        tuple[Any, str]: 第一个元素为实际发给上游的 `aiType`，第二个元素为
        `rootAiType`。
    """
    if not isinstance(model_name, str):
        return model_name, infer_root_ai_type(model_name)

    spec = MODEL_ALIAS_LOOKUP.get(model_name.lower())
    if spec is None:
        return model_name, infer_root_ai_type(model_name)
    return spec["request_id"], spec["root_ai_type"]


def infer_root_ai_type(model_name):
    """为未知模型推断上游路由类型。

    Args:
        model_name (Any): 调用方传入的模型名。

    Returns:
        str: 推断得到的 `rootAiType`，当前仅返回 `azure` 或 `xinference`。
    """
    if not isinstance(model_name, str):
        return "xinference"

    normalized = model_name.lower()
    # OpenAI / Azure 系列模型目前统一走 azure 路由。
    azure_markers = (
        "gpt-",
        "gpt",
        "o3",
        "o4-mini",
    )
    return "azure" if normalized.startswith(azure_markers) else "xinference"


def convert_messages_to_genai_format(messages):
    """从消息列表中提取 GenAI 所需的 `chatInfo`。

    当前上游实际请求中 `chatInfo` 只使用最后一条用户消息内容，因此这里
    仅做最小提取。

    Args:
        messages (list[dict]): OpenAI 风格的消息列表。

    Returns:
        str: 最后一条用户消息的文本内容；若不存在则返回空字符串。
    """
    # 上游会单独接收一份 chatInfo，这里取最后一条 user 消息与网页行为对齐。
    chat_info = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            chat_info = msg.get("content", "")
            break
    
    return chat_info

def extract_delta_from_genai(response_data):
    """从 GenAI 增量响应中提取正文和思维链字段。

    Args:
        response_data (dict): 单条 GenAI SSE 数据解析后的 JSON 对象。

    Returns:
        dict[str, str | None]: 包含 `reasoning` 与 `content` 两个字段；若缺失则
        返回 `None`。
    """
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            return {
                "reasoning": delta.get("reasoning"),
                "content": delta.get("content"),
            }
    except (KeyError, IndexError, TypeError):
        pass
    return {"reasoning": None, "content": None}


def stream_genai_events(messages, model, max_tokens):
    """调用 GenAI 流式接口并产出统一事件流。

    该函数是整个协议转换的底层入口，负责：
    1. 解析模型别名
    2. 调用上游 GenAI SSE 接口
    3. 将上游原始事件规范化为内部事件类型

    Args:
        messages (list[dict]): 发送给上游的消息列表。
        model (str): 调用方指定的模型名。
        max_tokens (int | None): 最大输出 token 数。

    Yields:
        dict: 统一事件对象，`type` 可能为 `delta`、`done`、`meta` 或 `error`。
    """
    upstream_model, root_ai_type = resolve_model(model)

    # 这里保持与网页端接近的请求体结构，避免上游校验差异。
    genai_data = {
        "chatInfo": "",
        "messages": messages,
        "type": "3",
        "stream": True,
        "aiType": upstream_model,
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": root_ai_type,
        "maxToken": max_tokens or 30000
    }

    try:
        response = requests.post(
            GENAI_URL,
            headers=GENAI_HEADERS,
            json=genai_data,
            stream=True,
            timeout=60
        )

        if response.status_code != 200:
            yield {
                "type": "error",
                "error": f"GenAI API error: {response.status_code}",
            }
            return

        finished = False
        for line in response.iter_lines():
            if finished:
                break

            if line:
                try:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    # 兼容标准 SSE 的 `data:` 前缀。
                    if line_str.startswith('data:'):
                        line_str = line_str[5:].strip()

                    if line_str:
                        genai_json = json.loads(line_str)

                        # 上游偶尔会返回补充元数据，先保留为内部 meta 事件。
                        if genai_json.get("other"):
                            yield {
                                "type": "meta",
                                "other": genai_json.get("other"),
                            }

                        # 只要上游给出 finish_reason，就视为本轮流式输出结束。
                        if "choices" in genai_json and len(genai_json["choices"]) > 0:
                            choice = genai_json["choices"][0]
                            if choice.get("finish_reason") is not None:
                                finished = True

                        if finished:
                            yield {
                                "type": "done",
                                "upstream_model": genai_json.get("model"),
                            }
                            break

                        delta = extract_delta_from_genai(genai_json)
                        reasoning = delta.get("reasoning")
                        content = delta.get("content")
                        # 内部统一拆成 reasoning 和 content，便于上层复用。
                        if reasoning is not None or content is not None:
                            yield {
                                "type": "delta",
                                "upstream_model": genai_json.get("model"),
                                "reasoning": reasoning,
                                "content": content,
                            }

                except json.JSONDecodeError:
                    pass

        yield {
            "type": "done",
            "upstream_model": None,
        }

    except Exception as e:
        # 流式链路统一转成 error 事件，交由上层协议各自包装。
        yield {
            "type": "error",
            "error": str(e),
        }


def stream_chat_completions_response(messages, model, max_tokens):
    """将内部事件流转换为 Chat Completions SSE。

    Args:
        messages (list[dict]): OpenAI 风格消息列表。
        model (str): 调用方传入的模型名。
        max_tokens (int | None): 最大输出 token 数。

    Yields:
        str: 符合 OpenAI Chat Completions SSE 格式的文本片段。
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())

    for event in stream_genai_events(messages, model, max_tokens):
        if event["type"] == "error":
            yield f"data: {json.dumps({'error': event['error']})}\n\n"
            return

        if event["type"] == "delta":
            delta_payload = {}
            # 对外沿用 DeepSeek 常见字段名 reasoning_content。
            if event.get("reasoning") is not None:
                delta_payload["reasoning_content"] = event["reasoning"]
            if event.get("content") is not None:
                delta_payload["content"] = event["content"]

            if delta_payload:
                openai_response = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta_payload,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(openai_response)}\n\n"

        if event["type"] == "done":
            final_response = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_response)}\n\n"
            yield "data: [DONE]\n\n"
            return


def collect_genai_response(messages, model, max_tokens):
    """收集完整响应并聚合为非流式结果。

    Args:
        messages (list[dict]): OpenAI 风格消息列表。
        model (str): 调用方传入的模型名。
        max_tokens (int | None): 最大输出 token 数。

    Returns:
        dict[str, str | None]: 聚合后的正文、思维链和上游模型名。

    Raises:
        RuntimeError: 当上游事件流返回错误事件时抛出。
    """
    content_parts = []
    reasoning_parts = []
    upstream_model = None

    for event in stream_genai_events(messages, model, max_tokens):
        if event["type"] == "error":
            raise RuntimeError(event["error"])
        if event["type"] == "delta":
            upstream_model = event.get("upstream_model") or upstream_model
            if event.get("reasoning"):
                reasoning_parts.append(event["reasoning"])
            if event.get("content"):
                content_parts.append(event["content"])
        if event["type"] == "done":
            break

    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "upstream_model": upstream_model,
    }


def build_response_input_messages(input_value):
    """将 Responses API 输入归一化为消息列表。

    当前仅处理文本输入，兼容字符串输入以及包含文本片段的数组输入。

    Args:
        input_value (str | list | Any): `/v1/responses` 的 `input` 字段。

    Returns:
        list[dict]: 可直接发给上游的消息列表。
    """
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if isinstance(input_value, list):
        messages = []
        for item in input_value:
            if not isinstance(item, dict):
                continue

            role = item.get("role", "user")
            content = item.get("content")

            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                # 仅提取文本片段，忽略当前版本尚未支持的其他 item 类型。
                text_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in {"input_text", "text", "output_text"}:
                        text = part.get("text")
                        if text:
                            text_parts.append(text)
                if text_parts:
                    messages.append({"role": role, "content": "\n".join(text_parts)})

        return messages

    return []


def stream_responses_api(messages, model, max_tokens):
    """将内部事件流转换为最小 Responses API SSE。

    Args:
        messages (list[dict]): 发送给上游的消息列表。
        model (str): 调用方传入的模型名。
        max_tokens (int | None): 最大输出 token 数。

    Yields:
        str: 符合最小 Responses API SSE 格式的文本片段。
    """
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(datetime.now().timestamp())
    reasoning_id = f"rs_{uuid.uuid4().hex[:12]}"
    output_index = 0

    created_event = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "in_progress",
            "model": model,
        }
    }
    yield f"data: {json.dumps(created_event)}\n\n"

    for event in stream_genai_events(messages, model, max_tokens):
        if event["type"] == "error":
            error_event = {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created,
                    "status": "failed",
                    "model": model,
                },
                "error": {
                    "message": event["error"],
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"
            return

        if event["type"] == "delta":
            # Responses 接口将 reasoning 和正文拆成不同事件类型。
            if event.get("reasoning") is not None:
                reasoning_event = {
                    "type": "response.reasoning.delta",
                    "response_id": response_id,
                    "output_index": output_index,
                    "item_id": reasoning_id,
                    "delta": event["reasoning"],
                }
                yield f"data: {json.dumps(reasoning_event)}\n\n"

            if event.get("content") is not None:
                content_event = {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "output_index": output_index,
                    "delta": event["content"],
                }
                yield f"data: {json.dumps(content_event)}\n\n"

        if event["type"] == "done":
            completed_event = {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created,
                    "status": "completed",
                    "model": model,
                }
            }
            yield f"data: {json.dumps(completed_event)}\n\n"
            yield "data: [DONE]\n\n"
            return

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理 OpenAI Chat Completions 兼容请求。

    Returns:
        Response: Flask JSON 响应或 SSE 流式响应。
    """
    try:
        req_data = request.get_json()
        
        # Chat Completions 至少需要消息数组。
        if not req_data or 'messages' not in req_data:
            return jsonify({'error': 'Missing messages field'}), 400
        
        messages = req_data.get('messages', [])
        model = req_data.get('model', 'gpt-3.5-turbo')
        stream = req_data.get('stream', False)
        max_tokens = req_data.get('max_tokens', 30000)
        
        # 转换消息格式
        chat_info = convert_messages_to_genai_format(messages)
        
        if not chat_info:
            return jsonify({'error': 'No user message found'}), 400

        if stream:
            return Response(
                stream_with_context(stream_chat_completions_response(messages, model, max_tokens)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                }
            )

        # 非流式模式先完整收集，再一次性组装 OpenAI 响应体。
        collected = collect_genai_response(messages, model, max_tokens)
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": collected["content"],
                        "reasoning_content": collected["reasoning_content"],
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(collected["content"]),
                "total_tokens": len(collected["content"])
            }
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/v1/responses', methods=['POST'])
def responses():
    """处理最小 OpenAI Responses 兼容请求。

    Returns:
        Response: Flask JSON 响应或 SSE 流式响应。
    """
    try:
        req_data = request.get_json()
        if not req_data or 'input' not in req_data:
            return jsonify({'error': 'Missing input field'}), 400

        model = req_data.get('model', 'gpt-4.1')
        stream = req_data.get('stream', False)
        max_output_tokens = req_data.get('max_output_tokens', req_data.get('max_tokens', 30000))
        messages = build_response_input_messages(req_data.get('input'))

        if not messages:
            return jsonify({'error': 'No input message found'}), 400

        if stream:
            return Response(
                stream_with_context(stream_responses_api(messages, model, max_output_tokens)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                }
            )

        # 非流式返回时，将 reasoning 和 message 组装到 output 数组中。
        collected = collect_genai_response(messages, model, max_output_tokens)
        response_id = f"resp_{uuid.uuid4().hex}"
        output = []
        if collected["reasoning_content"]:
            output.append({
                "id": f"rs_{uuid.uuid4().hex[:12]}",
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": collected["reasoning_content"],
                    }
                ]
            })
        output.append({
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": collected["content"],
                }
            ]
        })

        return jsonify({
            "id": response_id,
            "object": "response",
            "created_at": int(datetime.now().timestamp()),
            "status": "completed",
            "model": model,
            "output": output,
            "output_text": collected["content"],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """返回当前对外暴露的模型列表。

    Returns:
        Response: OpenAI `/v1/models` 兼容 JSON 响应。
    """
    models = []
    for spec in MODEL_SPECS:
        models.append({
            "id": spec["public_id"],
            "object": "model",
            "owned_by": "genai",
            "permission": []
        })
    
    return jsonify({"object": "list", "data": models})

@app.route('/health', methods=['GET'])
def health_check():
    """返回服务健康状态。

    Returns:
        tuple[Response, int]: 健康检查 JSON 响应与状态码。
    """
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=False)
