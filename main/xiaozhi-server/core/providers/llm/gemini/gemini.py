"""
Uses the same file name as the original Gemini provider for easier migration. Does not use Proxy IP for anything, so need to change web interface
API key and model name still needed to be set from Manager-Web
Hard-coded project_id and location, should be set in either the config file or environment variables
Uses user credentials from `vertex-ai-credentials.json` file in the current directory, not pushed for obvious reasons; same with project_id
"""

import os, json, uuid
from types import SimpleNamespace
from typing import Any, Dict, List

import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration, Part, Content

from core.providers.llm.base import LLMProviderBase
from config.logger import setup_logging

log = setup_logging()
TAG = __name__


class LLMProvider(LLMProviderBase):
    def __init__(self, cfg: Dict[str, Any]):
        log.bind(tag=TAG).info(f"Initializing Gemini Vertex AI provider with config: {cfg}")
        try:
            self.model_name = cfg.get("model_name", "gemini-2.0-flash")
            self.project_id = "your-project-id"                                                 # Replace with your actual project ID or fetch from config
            self.location = "asia-east2"
        except KeyError as e:
            log.bind(tag=TAG).error(f"Configuration missing for Vertex AI. Required key not found: {e}. Please update your config to include project_id and location.")
            raise e

        # The service account key file path can be configured here
        credentials_path = "vertex-ai-credentials.json"
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        else:
            log.bind(tag=TAG).warning(f"Vertex AI credentials file not found at: {credentials_path}")


        vertexai.init(project=self.project_id)
        self.model = GenerativeModel(self.model_name)
        self.chats: Dict[str, Any] = {}

        self.gen_cfg = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

    @staticmethod
    def _build_tools(funcs: List[Dict[str, Any]] | None):
        if not funcs:
            return None
        return [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name=f["function"]["name"],
                        description=f["function"]["description"],
                        parameters=f["function"]["parameters"],
                    )
                    for f in funcs
                ]
            )
        ]

    # Gemini documentation mentions that there is no need to maintain session-id, directly spliced together with dialogue
    def response(self, session_id, dialogue, **kwargs):
        yield from self._generate(session_id, dialogue, None)

    def response_with_functions(self, session_id, dialogue, functions=None):
        yield from self._generate(session_id, dialogue, self._build_tools(functions))

    def _generate(self, session_id, dialogue, tools):
        chat = self.chats.get(session_id)
        
        # If no chat session exists, create a new one
        if not chat:
            role_map = {"assistant": "model", "user": "user"}
            history: list[Content] = []
            system_instruction = None

            if dialogue and dialogue[0].get("role") == "system":
                system_instruction = dialogue[0].get("content")
                dialogue = dialogue[1:]

            model = GenerativeModel(self.model_name, system_instruction=system_instruction)

            for m in dialogue:
                r = m["role"]
                
                if r == "assistant" and "tool_calls" in m:
                    tc = m["tool_calls"][0]
                    history.append(Content(role="model", parts=[Part.from_dict({
                        'function_call': {
                            'name': tc['function']['name'],
                            'args': json.loads(tc['function']['arguments'])
                        }
                    })]))
                    continue

                if r == "tool":
                    history.append(Content(role="function", parts=[Part.from_dict({
                        'function_response': {
                            'name': m['name'],
                            'response': {'content': m['content']}
                        }
                    })]))
                    continue

                mapped_role = role_map.get(r)
                if mapped_role:
                    content_text = str(m.get("content", ""))
                    if content_text:
                        if history and history[-1].role == mapped_role:
                            history[-1].parts.append(Part.from_text(content_text))
                        else:
                            history.append(Content(role=mapped_role, parts=[Part.from_text(content_text)]))

            chat = model.start_chat(history=history)
            self.chats[session_id] = chat

        # The last message in the dialogue is the one to send
        latest_message = dialogue[-1] if dialogue else None
        
        if not latest_message or not latest_message.get("content"):
            # If there's no content to send, we can't proceed
            return

        log.bind(tag=TAG).info(f"Sending request to Vertex AI for session {session_id}")
        try:
            stream = chat.send_message(
                content=latest_message["content"],
                generation_config=self.gen_cfg,
                tools=tools,
                stream=True,
            )
            log.bind(tag=TAG).info("Successfully sent request to Vertex AI, streaming response.")
        except Exception as e:
            log.bind(tag=TAG).error(f"Error calling Vertex AI: {e}", exc_info=True)
            raise

        try:
            for chunk in stream:
                log.bind(tag=TAG).debug(f"Received chunk from Vertex AI: {chunk}")
                if not chunk.candidates:
                    continue
                
                part = chunk.candidates[0].content.parts[0]
                # a) Function call - usually the last part is the function call
                if part.function_call:
                    fc = part.function_call
                    yield None, [
                        SimpleNamespace(
                            id=uuid.uuid4().hex,
                            type="function",
                            function=SimpleNamespace(
                                name=fc.name,
                                arguments=json.dumps(
                                    dict(fc.args), ensure_ascii=False
                                ),
                            ),
                        )
                    ]
                    return
                # b) Normal text
                if part.text:
                    yield part.text if tools is None else (part.text, None)
        except Exception as e:
            log.bind(tag=TAG).error(f"Error processing stream from Vertex AI: {e}", exc_info=True)
        finally:
            if tools is not None:
                yield None, None  # function‑mode ends, returns dummy package
