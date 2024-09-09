from typing import Optional, Tuple
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
    layer: LayerLike,
    se_block: LayerLike,
    graph_name: str | None = None,
    pre_build: bool = True,
    only_graph: bool = False,
) -> Tuple[str, LayerGraph] | LayerGraph:

    def _get_default_name(layer: LayerLike) -> str:
        return type(layer).__name__ + "_node"

    root_node = LayerNode(layer, name=_get_default_name(layer))
    se_node = LayerNode(se_block, name=_get_default_name(se_block))
    scale_node = LayerNode(Identity(), MergeMode.HADAMARD, name="scale_node")

    graph = LayerGraph(
        graph={root_node: [se_node, scale_node], se_node: [scale_node]},
        root=root_node,
        term=scale_node,
    )
    if pre_build:
        graph.build()

    if graph_name is None:
        graph_name = type(layer).__name__ + "_SE"

    return graph if only_graph else (graph_name, graph)
