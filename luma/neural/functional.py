from typing import Any, Optional, Tuple
from dataclasses import dataclass, asdict

from luma.interface.typing import LayerLike
from luma.neural.layer import *
from luma.neural.autoprop import LayerNode, LayerGraph, MergeMode


def make_res_layers(
    in_channels: int,
    out_channels: int,
    block: LayerLike,
    n_blocks: int,
    layer_num: int,
    conv_base_args: dict,
    res_base_args_dc: dataclass,
    stride: int = 1,
    layer_label: str = "ResNetConv",
) -> tuple[Sequential, int]:
    downsampling: Optional[Sequential] = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsampling = Sequential(
            Conv2D(
                in_channels,
                out_channels * block.expansion,
                1,
                stride,
                **conv_base_args,
            ),
            BatchNorm2D(out_channels * block.expansion),
        )

    first_block = block(
        in_channels,
        out_channels,
        stride,
        downsampling,
        **asdict(res_base_args_dc),
    )
    layers: list = [(f"{layer_label}{layer_num}_1", first_block)]

    in_channels = out_channels * block.expansion
    for i in range(1, n_blocks):
        new_block = (
            f"{layer_label}{layer_num}_{i + 1}",
            block(
                in_channels,
                out_channels,
                **asdict(res_base_args_dc),
            ),
        )
        layers.append(new_block)

    return Sequential(*layers), in_channels


def attach_se_block(
    layer: type[LayerLike],
    se_block: type[LayerLike],
    layer_args: dict = {},
    se_args: dict = {},
    pre_build: bool = True,
) -> LayerGraph:
    layer_inst = layer(**layer_args)
    se_inst = se_block(**se_args)

    root_node = LayerNode(layer_inst, name="root")
    se_node = LayerNode(se_inst, name="se")
    scale_node = LayerNode(Identity(), MergeMode.HADAMARD, name="scale")

    graph = LayerGraph(
        graph={
            root_node: [se_node, scale_node],
            se_node: [scale_node],
        },
        root=root_node,
        term=scale_node,
    )
    if pre_build:
        graph.build()

    setattr(graph, "__str__", layer.__str__)
    setattr(graph, "out_shape", layer_inst.out_shape)

    return graph
