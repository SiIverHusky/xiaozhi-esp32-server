import json
import asyncio
from plugins_func.register import (
    FunctionItem,
    register_device_function,
    ActionResponse,
    Action,
    ToolType,
)

TAG = __name__


def wrap_async_function(async_func):
    """包装异步函数为同步函数"""

    def wrapper(*args, **kwargs):
        try:
            # 获取连接对象（第一个参数）
            conn = args[0]
            if not hasattr(conn, "loop"):
                conn.logger.bind(tag=TAG).error("Connection object does not have loop attribute")
                return ActionResponse(
                    Action.ERROR,
                    "Connection object does not have loop attribute",
                    "Error when performing operation: Connection object does not have loop attribute",
                )

            # 使用conn对象中的事件循环
            loop = conn.loop
            # 在conn的事件循环中运行异步函数
            future = asyncio.run_coroutine_threadsafe(async_func(*args, **kwargs), loop)
            # 等待结果返回
            return future.result()
        except Exception as e:
            conn.logger.bind(tag=TAG).error(f"Error occurred while running asynchronous func: {e}")
            return ActionResponse(Action.ERROR, str(e), f"error occurs when executing operation: {e}")

    return wrapper


def create_iot_function(device_name, method_name, method_info):
    """
    根据IOT设备描述生成通用的控制函数
    """

    async def iot_control_function(
        conn, response_success=None, response_failure=None, **params
    ):
        try:
            # 设置默认响应消息
            if not response_success:
                response_success = "operation successed"
            if not response_failure:
                response_failure = "operation failed"

            # 打印响应参数
            conn.logger.bind(tag=TAG).debug(
                f"response parameter received by control function: success='{response_success}', failure='{response_failure}'"
            )

            # 发送控制命令
            await send_iot_conn(conn, device_name, method_name, params)
            # 等待一小段时间让状态更新
            await asyncio.sleep(0.1)

            # 生成结果信息
            result = f"{method_name} of {device_name} operated successfully"

            # 处理响应中可能的占位符
            response = response_success
            # 替换{value}占位符
            for param_name, param_value in params.items():
                # 先尝试直接替换参数值
                if "{" + param_name + "}" in response:
                    response = response.replace(
                        "{" + param_name + "}", str(param_value)
                    )

                # 如果有{value}占位符，用相关参数替换
                if "{value}" in response:
                    response = response.replace("{value}", str(param_value))
                    break

            return ActionResponse(Action.RESPONSE, result, response)
        except Exception as e:
            conn.logger.bind(tag=TAG).error(
                f"failed to operate {method_name} of {device_name} : {e}"
            )

            # 操作失败时使用大模型提供的失败响应
            response = response_failure

            return ActionResponse(Action.ERROR, str(e), response)

    return wrap_async_function(iot_control_function)


def create_iot_query_function(device_name, prop_name, prop_info):
    """
    根据IOT设备属性创建查询函数
    """

    async def iot_query_function(conn, response_success=None, response_failure=None):
        try:
            # 打印响应参数
            conn.logger.bind(tag=TAG).info(
                f"Response parameter received by query function: success='{response_success}', failure='{response_failure}'"
            )

            value = await get_iot_status(conn, device_name, prop_name)

            # 查询成功，生成结果
            if value is not None:
                # 使用大模型提供的成功响应，并替换其中的占位符
                response = response_success.replace("{value}", str(value))

                return ActionResponse(Action.RESPONSE, str(value), response)
            else:
                # 查询失败，使用大模型提供的失败响应
                response = response_failure

                return ActionResponse(Action.ERROR, f"Property {prop_name} does not exist", response)
        except Exception as e:
            conn.logger.bind(tag=TAG).error(
                f"Error occurs when search for {prop_name} of {device_name}: {e}"
            )

            # 查询出错时使用大模型提供的失败响应
            response = response_failure

            return ActionResponse(Action.ERROR, str(e), response)

    return wrap_async_function(iot_query_function)


class IotDescriptor:
    """
    A class to represent an IoT descriptor.
    """

    def __init__(self, name, description, properties, methods):
        self.name = name
        self.description = description
        self.properties = []
        self.methods = []

        # 根据描述创建属性
        if properties is not None:
            for key, value in properties.items():
                property_item = {}
                property_item["name"] = key
                property_item["description"] = value["description"]
                if value["type"] == "number":
                    property_item["value"] = 0
                elif value["type"] == "boolean":
                    property_item["value"] = False
                else:
                    property_item["value"] = ""
                self.properties.append(property_item)

        # 根据描述创建方法
        if methods is not None:
            for key, value in methods.items():
                method = {}
                method["description"] = value["description"]
                method["name"] = key
                # 检查方法是否有参数
                if "parameters" in value:
                    method["parameters"] = {}
                    for k, v in value["parameters"].items():
                        method["parameters"][k] = {
                            "description": v["description"],
                            "type": v["type"],
                        }
                self.methods.append(method)


def register_device_type(descriptor, device_type_registry):
    """注册设备类型及其功能"""
    device_name = descriptor["name"]
    type_id = device_type_registry.generate_device_type_id(descriptor)

    # 如果该类型已注册，直接返回类型ID
    if type_id in device_type_registry.type_functions:
        return type_id

    functions = {}

    # 为每个属性创建查询函数
    for prop_name, prop_info in descriptor["properties"].items():
        func_name = f"get_{device_name.lower()}_{prop_name.lower()}"
        func_desc = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": f"Search for {prop_info['description']} of {descriptor['description']}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response_success": {
                            "type": "string",
                            "description": f"Friendly reply when the query success, must use {{value}} as a placeholder to represent the retrieved value",
                        },
                        "response_failure": {
                            "type": "string",
                            "description": f"Friendly reply when the query success failed, e.g., 'cannot get {prop_info['description']} of {device_name}'",
                        },
                    },
                    "required": ["response_success", "response_failure"],
                },
            },
        }
        query_func = create_iot_query_function(device_name, prop_name, prop_info)
        decorated_func = register_device_function(
            func_name, func_desc, ToolType.IOT_CTL
        )(query_func)
        functions[func_name] = FunctionItem(
            func_name, func_desc, decorated_func, ToolType.IOT_CTL
        )

    # 为每个方法创建控制函数
    for method_name, method_info in descriptor["methods"].items():
        func_name = f"{device_name.lower()}_{method_name.lower()}"

        # 创建参数字典，添加原有参数
        parameters = {}
        required_params = []

        # 如果方法有参数，则添加参数信息
        if "parameters" in method_info:
            parameters = {
                param_name: {
                    "type": param_info["type"],
                    "description": param_info["description"],
                }
                for param_name, param_info in method_info["parameters"].items()
            }
            required_params = list(method_info["parameters"].keys())

        # 添加响应参数
        parameters.update(
            {
                "response_success": {
                    "type": "string",
                    "description": "Friendly reply when the operation successed. For the device name in the operation result, use the name from the description whenever possible",
                },
                "response_failure": {
                    "type": "string",
                    "description": "Friendly reply when the operation fails. For the device name in the operation result, use the name from the description whenever possible",
                },
            }
        )

        # 构建必须参数列表（原有参数 + 响应参数）
        required_params.extend(["response_success", "response_failure"])

        func_desc = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": f"{descriptor['description']} - {method_info['description']}",
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required_params,
                },
            },
        }
        control_func = create_iot_function(device_name, method_name, method_info)
        decorated_func = register_device_function(
            func_name, func_desc, ToolType.IOT_CTL
        )(control_func)
        functions[func_name] = FunctionItem(
            func_name, func_desc, decorated_func, ToolType.IOT_CTL
        )

    device_type_registry.register_device_type(type_id, functions)
    return type_id


# 用于接受前端设备推送的搜索iot描述
async def handleIotDescriptors(conn, descriptors):
    wait_max_time = 5
    while conn.func_handler is None or not conn.func_handler.finish_init:
        await asyncio.sleep(1)
        wait_max_time -= 1
        if wait_max_time <= 0:
            conn.logger.bind(tag=TAG).debug("connected object does not have func_handler")
            return
    """处理物联网描述"""
    functions_changed = False

    for descriptor in descriptors:
        # 如果descriptor没有properties和methods，则直接跳过
        if "properties" not in descriptor and "methods" not in descriptor:
            continue

        # 处理缺失properties的情况
        if "properties" not in descriptor:
            descriptor["properties"] = {}
            # 从methods中提取所有参数作为properties
            if "methods" in descriptor:
                for method_name, method_info in descriptor["methods"].items():
                    if "parameters" in method_info:
                        for param_name, param_info in method_info["parameters"].items():
                            # 将参数信息转换为属性信息
                            descriptor["properties"][param_name] = {
                                "description": param_info["description"],
                                "type": param_info["type"],
                            }

        # 创建IOT设备描述符
        iot_descriptor = IotDescriptor(
            descriptor["name"],
            descriptor["description"],
            descriptor["properties"],
            descriptor["methods"],
        )
        conn.iot_descriptors[descriptor["name"]] = iot_descriptor

        if conn.load_function_plugin:
            # 注册或获取设备类型
            device_type_registry = conn.func_handler.device_type_registry
            type_id = register_device_type(descriptor, device_type_registry)
            device_functions = device_type_registry.get_device_functions(type_id)

            # 在连接级注册设备函数
            if hasattr(conn, "func_handler"):
                for func_name, func_item in device_functions.items():
                    conn.func_handler.function_registry.register_function(
                        func_name, func_item
                    )
                    conn.logger.bind(tag=TAG).info(
                        f"register IOT function to function handler: {func_name}"
                    )
                    functions_changed = True

    # 如果注册了新函数，更新function描述列表
    if functions_changed and hasattr(conn, "func_handler"):
        func_names = conn.func_handler.current_support_functions()
        conn.logger.bind(tag=TAG).info(f"device type: {type_id}")
        conn.logger.bind(tag=TAG).info(
            f"Function description list updated. Currently supported functions: {func_names}"
        )


async def handleIotStatus(conn, states):
    """处理物联网状态"""
    for state in states:
        for key, value in conn.iot_descriptors.items():
            if key == state["name"]:
                for property_item in value.properties:
                    for k, v in state["state"].items():
                        if property_item["name"] == k:
                            if type(v) != type(property_item["value"]):
                                conn.logger.bind(tag=TAG).error(
                                    f"attribute {property_item['name']}value type unmatch"
                                )
                                break
                            else:
                                property_item["value"] = v
                                conn.logger.bind(tag=TAG).info(
                                    f"IOT status updated: {key} , {property_item['name']} = {v}"
                                )
                            break
                break


async def get_iot_status(conn, name, property_name):
    """获取物联网状态"""
    for key, value in conn.iot_descriptors.items():
        if key == name:
            for property_item in value.properties:
                if property_item["name"] == property_name:
                    return property_item["value"]
    conn.logger.bind(tag=TAG).warning(f"did not find property {property_name} of device {name}")
    return None


async def set_iot_status(conn, name, property_name, value):
    """设置物联网状态"""
    for key, iot_descriptor in conn.iot_descriptors.items():
        if key == name:
            for property_item in iot_descriptor.properties:
                if property_item["name"] == property_name:
                    if type(value) != type(property_item["value"]):
                        conn.logger.bind(tag=TAG).error(
                            f"property {property_item['name']} value type unmatch"
                        )
                        return
                    property_item["value"] = value
                    conn.logger.bind(tag=TAG).info(
                        f"IOT status updated: {name} , {property_name} = {value}"
                    )
                    return
    conn.logger.bind(tag=TAG).warning(f"did not find property {property_name} of device {name} ")


async def send_iot_conn(conn, name, method_name, parameters):
    """发送物联网指令"""
    for key, value in conn.iot_descriptors.items():
        if key == name:
            # 找到了设备
            for method in value.methods:
                # 找到了方法
                if method["name"] == method_name:
                    # 构建命令对象
                    command = {
                        "name": name,
                        "method": method_name,
                    }

                    # 只有当参数不为空时才添加parameters字段
                    if parameters:
                        command["parameters"] = parameters
                    send_message = json.dumps({"type": "iot", "commands": [command]})
                    await conn.websocket.send(send_message)
                    conn.logger.bind(tag=TAG).info(f"send IOT command: {send_message}")
                    return
    conn.logger.bind(tag=TAG).error(f"did not find method {method_name}")
