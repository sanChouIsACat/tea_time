from ..serializer import Serializer, serialiable
from dataclasses import dataclass
from typing import *


@serialiable
@dataclass
class InnerCtx:
    id: str


@serialiable
@dataclass
class testCtx:
    id: int
    name: str
    addresses: list[str]
    human: bool
    inner: InnerCtx
    inners: list[InnerCtx]
    some_tuple: Tuple[str, str]
    others: dict[str, str]


def test_example():
    yaml: str = Serializer.generate_example_yaml(testCtx)
    assert yaml != None
    assert yaml != ""
    print(yaml)


def test_deserialze():
    some_dict = dict()
    some_dict["1"] = str(123)
    origin = testCtx(
        1,
        "wang",
        ["stress 1", "stress 2"],
        True,
        InnerCtx(1),
        [InnerCtx(1), InnerCtx(2)],
        ("1", "2"),
        some_dict,
    )
    yaml = Serializer.serialize(origin)
    print(yaml)
    deserialized = Serializer.deserialize(yaml, testCtx)
    assert deserialized == origin
