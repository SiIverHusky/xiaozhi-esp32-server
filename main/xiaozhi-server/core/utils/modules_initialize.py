from typing import Dict, Any
from config.logger import setup_logging
from core.utils import tts, llm, intent, memory, vad, asr

TAG = __name__
logger = setup_logging()


def initialize_modules(
    logger,
    config: Dict[str, Any],
    init_vad=False,
    init_asr=False,
    init_llm=False,
    init_tts=False,
    init_memory=False,
    init_intent=False,
) -> Dict[str, Any]:
    """
    初始化所有模块组件

    Args:
        config: 配置字典

    Returns:
        Dict[str, Any]: 包含所有初始化后的模块的字典
    """
    modules = {}

    # 初始化TTS模块
    if init_tts:
        select_tts_module = config["selected_module"]["TTS"]
        modules["tts"] = initialize_tts(config)
        logger.bind(tag=TAG).info(f"initialize module: tts success {select_tts_module}")

    # 初始化LLM模块
    if init_llm:
        select_llm_module = config["selected_module"]["LLM"]
        llm_type = (
            select_llm_module
            if "type" not in config["LLM"][select_llm_module]
            else config["LLM"][select_llm_module]["type"]
        )
        modules["llm"] = llm.create_instance(
            llm_type,
            config["LLM"][select_llm_module],
        )
        logger.bind(tag=TAG).info(f"initialize module: llm success {select_llm_module}")

    # 初始化Intent模块
    if init_intent:
        select_intent_module = config["selected_module"]["Intent"]
        intent_type = (
            select_intent_module
            if "type" not in config["Intent"][select_intent_module]
            else config["Intent"][select_intent_module]["type"]
        )
        modules["intent"] = intent.create_instance(
            intent_type,
            config["Intent"][select_intent_module],
        )
        logger.bind(tag=TAG).info(f"initialize module: intent success {select_intent_module}")

    # 初始化Memory模块
    if init_memory:
        select_memory_module = config["selected_module"]["Memory"]
        memory_type = (
            select_memory_module
            if "type" not in config["Memory"][select_memory_module]
            else config["Memory"][select_memory_module]["type"]
        )
        modules["memory"] = memory.create_instance(
            memory_type,
            config["Memory"][select_memory_module],
            config.get("summaryMemory", None),
        )
        logger.bind(tag=TAG).info(f"initialize module: memory success {select_memory_module}")

    # 初始化VAD模块
    if init_vad:
        select_vad_module = config["selected_module"]["VAD"]
        vad_type = (
            select_vad_module
            if "type" not in config["VAD"][select_vad_module]
            else config["VAD"][select_vad_module]["type"]
        )
        modules["vad"] = vad.create_instance(
            vad_type,
            config["VAD"][select_vad_module],
        )
        logger.bind(tag=TAG).info(f"initialize module: vad success {select_vad_module}")

    # 初始化ASR模块
    if init_asr:
        select_asr_module = config["selected_module"]["ASR"]
        modules["asr"] = initialize_asr(config)
        logger.bind(tag=TAG).info(f"initialize module: asr success {select_asr_module}")
    return modules


def initialize_tts(config):
    select_tts_module = config["selected_module"]["TTS"]
    tts_type = (
        select_tts_module
        if "type" not in config["TTS"][select_tts_module]
        else config["TTS"][select_tts_module]["type"]
    )
    new_tts = tts.create_instance(
        tts_type,
        config["TTS"][select_tts_module],
        str(config.get("delete_audio", True)).lower() in ("true", "1", "yes"),
    )
    return new_tts


def initialize_asr(config):
    select_asr_module = config["selected_module"]["ASR"]
    asr_type = (
        select_asr_module
        if "type" not in config["ASR"][select_asr_module]
        else config["ASR"][select_asr_module]["type"]
    )
    new_asr = asr.create_instance(
        asr_type,
        config["ASR"][select_asr_module],
        str(config.get("delete_audio", True)).lower() in ("true", "1", "yes"),
    )
    return new_asr
