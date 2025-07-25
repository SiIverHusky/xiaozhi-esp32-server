from config.logger import setup_logging
import json
from plugins_func.register import (
    FunctionRegistry,
    ActionResponse,
    Action,
    ToolType,
    DeviceTypeRegistry,
)
from plugins_func.functions.hass_init import append_devices_to_prompt

TAG = __name__


class FunctionHandler:
    def __init__(self, conn):
        self.conn = conn
        self.config = conn.config
        self.device_type_registry = DeviceTypeRegistry()
        self.function_registry = FunctionRegistry()
        self.register_nessary_functions()
        self.register_config_functions()
        self.functions_desc = self.function_registry.get_all_function_desc()
        self.finish_init = True

    def current_support_functions(self):
        func_names = []
        for func in self.functions_desc:
            func_names.append(func["function"]["name"])
        # 打印当前支持的函数列表
        self.conn.logger.bind(tag=TAG, session_id=self.conn.session_id).info(
            f"currently supported function list: {func_names}"
        )
        return func_names

    def get_functions(self):
        """获取功能调用配置"""
        return self.functions_desc

    def register_nessary_functions(self):
        """注册必要的函数"""
        self.function_registry.register_function("handle_exit_intent")
        self.function_registry.register_function("get_time")
        self.function_registry.register_function("get_lunar")

    def register_config_functions(self):
        """注册配置中的函数,可以不同客户端使用不同的配置"""
        for func in self.config["Intent"][self.config["selected_module"]["Intent"]].get(
            "functions", []
        ):
            self.function_registry.register_function(func)

        """home assistant需要初始化提示词"""
        append_devices_to_prompt(self.conn)

    def get_function(self, name):
        return self.function_registry.get_function(name)

    def handle_llm_function_call(self, conn, function_call_data):
        try:
            function_name = function_call_data["name"]
            funcItem = self.get_function(function_name)
            if not funcItem:
                return ActionResponse(
                    action=Action.NOTFOUND, result="did not find corresponding function", response=""
                )
            func = funcItem.func
            arguments = function_call_data["arguments"]
            arguments = json.loads(arguments) if arguments else {}
            self.conn.logger.bind(tag=TAG).debug(
                f"call function: {function_name}, parameter: {arguments}"
            )
            if (
                funcItem.type == ToolType.SYSTEM_CTL
                or funcItem.type == ToolType.IOT_CTL
            ):
                return func(conn, **arguments)
            elif funcItem.type == ToolType.WAIT:
                return func(**arguments)
            elif funcItem.type == ToolType.CHANGE_SYS_PROMPT:
                return func(conn, **arguments)
            else:
                return ActionResponse(
                    action=Action.NOTFOUND, result="did not find corresponding function", response=""
                )
        except Exception as e:
            self.conn.logger.bind(tag=TAG).error(f"handling function call error: {e}")

        return None
