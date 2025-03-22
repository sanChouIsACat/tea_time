import yaml
import random
import string
from dataclasses import dataclass, asdict, fields, MISSING, is_dataclass
from typing import Type
from types import GenericAlias


# 自动生成 from_dict 方法的装饰器
def serialiable(cls):
    """
    装饰器，自动为 dataclass 类添加 from_dict 方法
    """

    def from_dict(cls, data):
        return cls(**data)

    # 将 from_dict 方法添加到 dataclass 类
    setattr(cls, "from_dict", classmethod(from_dict))
    return cls


# 序列化/反序列化库
class Serializer:

    # 随机生成器函数
    @staticmethod
    def __generate_random_value(
        field_type, set_value_ele_type=str, set_key_ele_type=str
    ):
        """
        根据字段类型生成随机值
        :param field_type: 字段类型
        :return: 随机值
        """
        if field_type == None:
            return None
        elif is_dataclass(field_type):
            # 如果是 dataclass 类型，递归生成值
            return Serializer.__generate_example_dict(field_type)
        elif field_type == str:
            return "".join(
                random.choices(string.ascii_letters, k=10)
            )  # 随机生成一个10个字符的字符串
        elif field_type == float:
            return round(
                random.uniform(1.0, 100.0), 2
            )  # 随机生成一个浮动值，范围在1到100之间
        elif field_type == int:
            return random.randint(1, 100)  # 随机生成一个整数，范围在1到100之间
        elif field_type == list:
            ret = []
            for _ in range(3):
                ret.append(Serializer.__generate_random_value(set_value_ele_type))
            return ret

        elif field_type == dict:
            ret = {}
            for _ in range(3):
                ret[Serializer.__generate_random_value(set_key_ele_type)] = (
                    Serializer.__generate_random_value(set_value_ele_type)
                )
            return ret
        elif isinstance(field_type, GenericAlias):
            tuple = Serializer.__extract_generic_type(field_type)
            if len(tuple) == 2:
                return Serializer.__generate_random_value(tuple[0], tuple[1])
            else:
                return Serializer.__generate_random_value(tuple[0], tuple[1], tuple[2])
        else:
            return None  # 如果遇到未知类型，返回None

    # 递归生成示例字典
    @staticmethod
    def __generate_example_dict(cls: Type) -> dict:
        """
        根据 dataclass 类型生成示例字典，递归处理嵌套的 dataclass 字段
        :param cls: dataclass 类型
        :return: 示例字典
        """
        example_dict = {}
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            random_value = Serializer.__generate_random_value(field_type)
            example_dict[field_name] = random_value
        return example_dict

    @staticmethod
    def __extract_generic_type(generic_type):
        # 检查泛型类型对象中的__args__属性
        if hasattr(generic_type, "__args__"):
            return generic_type.__origin__, *generic_type.__args__
        return generic_type.__origin__

    @staticmethod
    def serialize(obj) -> str:
        """
        将 dataclass 实例序列化为 YAML 格式的字符串
        :param obj: dataclass 实例
        :return: YAML 格式的字符串
        """
        if not hasattr(obj, "__dataclass_fields__"):
            raise TypeError(f"对象 {obj} 不是一个 dataclass 实例")
        obj_dict = asdict(obj)
        return yaml.dump(obj_dict)

    @staticmethod
    def deserialize(yaml_str: str, cls: Type) -> object:
        """
        将 YAML 字符串反序列化为指定类型的 dataclass 实例
        :param yaml_str: YAML 格式的字符串
        :param cls: 目标 dataclass 类型
        :return: 指定类型的 dataclass 实例
        """
        data = yaml.load(yaml_str, Loader=yaml.FullLoader)
        return cls.from_dict(data)

    @staticmethod
    def generate_example_yaml(cls: Type) -> str:
        """
        根据 dataclass 类型生成一个示例 YAML 文件，使用随机真实值
        :param cls: dataclass 类型
        :return: 示例 YAML 格式的字符串
        """
        if not hasattr(cls, "__dataclass_fields__"):
            raise TypeError(f"类 {cls} 不是一个 dataclass 类型")

        # 创建一个包含字段随机值的字典
        example_dict = Serializer.__generate_example_dict(cls)

        # 将字典转换为 YAML 格式的字符串
        return yaml.dump(example_dict)
