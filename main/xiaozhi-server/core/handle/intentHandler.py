import json
import asyncio
import uuid
from core.handle.sendAudioHandle import send_stt_message
from core.handle.helloHandle import checkWakeupWords
from core.utils.util import remove_punctuation_and_length
from core.providers.tts.dto.dto import ContentType
from core.utils.dialogue import Message
from core.handle.mcpHandle import call_mcp_tool
from plugins_func.register import Action, ActionResponse
from loguru import logger

TAG = __name__


async def handle_user_intent(conn, text):
    # 检查是否有明确的退出命令
    filtered_text = remove_punctuation_and_length(text)[1]
    if await check_direct_exit(conn, filtered_text):
        return True
    # 检查是否是唤醒词
    if await checkWakeupWords(conn, filtered_text):
        return True

    if conn.intent_type == "function_call":
        # 使用支持function calling的聊天方法,不再进行意图分析
        return False
    # 使用LLM进行意图分析
    intent_result = await analyze_intent_with_llm(conn, text)
    if not intent_result:
        return False
    # 处理各种意图
    return await process_intent_result(conn, intent_result, text)


async def check_direct_exit(conn, text):
    """检查是否有明确的退出命令"""
    _, text = remove_punctuation_and_length(text)
    cmd_exit = conn.cmd_exit
    for cmd in cmd_exit:
        if text == cmd:
            conn.logger.bind(tag=TAG).info(f"detect clear exit command: {text}")
            await send_stt_message(conn, text)
            await conn.close()
            return True
    return False


async def analyze_intent_with_llm(conn, text):
    """使用LLM分析用户意图"""
    if not hasattr(conn, "intent") or not conn.intent:
        conn.logger.bind(tag=TAG).warning("Intent recognition service not initialized")
        return None

    # 对话历史记录
    dialogue = conn.dialogue
    try:
        intent_result = await conn.intent.detect_intent(conn, dialogue.dialogue, text)
        return intent_result
    except Exception as e:
        conn.logger.bind(tag=TAG).error(f"Intent Recognition failed: {str(e)}")

    return None


async def process_intent_result(conn, intent_result, original_text):
    """处理意图识别结果"""
    try:
        # 尝试将结果解析为JSON
        intent_data = json.loads(intent_result)

        # 检查是否有function_call
        if "function_call" in intent_data:
            # 直接从意图识别获取了function_call
            conn.logger.bind(tag=TAG).debug(
                f"the intent result in function_call format is detected: {intent_data['function_call']['name']}"
            )
            function_name = intent_data["function_call"]["name"]
            if function_name == "continue_chat":
                return False

            if function_name == "play_music":
                funcItem = conn.func_handler.get_function(function_name)
                if not funcItem:
                    conn.func_handler.function_registry.register_function("play_music")

            function_args = {}
            if "arguments" in intent_data["function_call"]:
                function_args = intent_data["function_call"]["arguments"]
                if function_args is None:
                    function_args = {}
            # 确保参数是字符串格式的JSON
            if isinstance(function_args, dict):
                function_args = json.dumps(function_args)

            function_call_data = {
                "name": function_name,
                "id": str(uuid.uuid4().hex),
                "arguments": function_args,
            }

            await send_stt_message(conn, original_text)
            conn.client_abort = False

            # 使用executor执行函数调用和结果处理
            def process_function_call():
                conn.dialogue.put(Message(role="user", content=original_text))

                # 处理Server端MCP工具调用
                if conn.mcp_manager.is_mcp_tool(function_name):
                    result = conn._handle_mcp_tool_call(function_call_data)
                elif hasattr(conn, "mcp_client") and conn.mcp_client.has_tool(
                    function_name
                ):
                    # 如果是小智端MCP工具调用
                    conn.logger.bind(tag=TAG).debug(
                        f"call xiaozhi-side MCP tool: {function_name}, parameter: {function_args}"
                    )
                    try:
                        result = asyncio.run_coroutine_threadsafe(
                            call_mcp_tool(
                                conn, conn.mcp_client, function_name, function_args
                            ),
                            conn.loop,
                        ).result()
                        conn.logger.bind(tag=TAG).debug(f"MCP-tool call result: {result}")
                        result = ActionResponse(
                            action=Action.REQLLM, result=result, response=""
                        )
                    except Exception as e:
                        conn.logger.bind(tag=TAG).error(f"failed to call MCP tool: {e}")
                        result = ActionResponse(
                            action=Action.REQLLM, result="failed to call MCP tool", response=""
                        )
                else:
                    # 处理系统函数
                    result = conn.func_handler.handle_llm_function_call(
                        conn, function_call_data
                    )

                if result:
                    if result.action == Action.RESPONSE:  # 直接回复前端
                        text = result.response
                        if text is not None:
                            speak_txt(conn, text)
                    elif result.action == Action.REQLLM:  # 调用函数后再请求llm生成回复
                        text = result.result
                        conn.dialogue.put(Message(role="tool", content=text))
                        llm_result = conn.intent.replyResult(text, original_text)
                        if llm_result is None:
                            llm_result = text
                        speak_txt(conn, llm_result)
                    elif (
                        result.action == Action.NOTFOUND
                        or result.action == Action.ERROR
                    ):
                        text = result.result
                        if text is not None:
                            speak_txt(conn, text)
                    elif function_name != "play_music":
                        # For backward compatibility with original code
                        # 获取当前最新的文本索引
                        text = result.response
                        if text is None:
                            text = result.result
                        if text is not None:
                            speak_txt(conn, text)

            # 将函数执行放在线程池中
            conn.executor.submit(process_function_call)
            return True
        return False
    except json.JSONDecodeError as e:
        conn.logger.bind(tag=TAG).error(f"Error occurs when handling intent result: {e}")
        return False


def speak_txt(conn, text):
    conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=text)
    conn.dialogue.put(Message(role="assistant", content=text))
