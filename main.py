from util.serializer import Serializer, serialiable
from dataclasses import dataclass
from loader.myLoader import MyLoaderCtx
from model.basic_model import VaeCtx


@serialiable
@dataclass
class trainCtx:
    vae_ctx: VaeCtx
    my_loader_ctx: MyLoaderCtx


print(Serializer.generate_example_yaml(trainCtx))
