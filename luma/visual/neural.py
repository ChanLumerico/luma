from typing import Any, List, Literal, Optional, Type
import matplotlib.pyplot as plt
import numpy as np

from luma.core.super import Visualizer
from luma.neural.base import NeuralModel
from luma.neural.model import get_model, load_model_registry


FormattedKey = str


class ModelScatterPlot(Visualizer):
    def __init__(
        self,
        models: List[Type[NeuralModel] | str],
        x_axis: FormattedKey,
        y_axis: FormattedKey,
        s_key: FormattedKey | None = None,
    ) -> None:
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.s_key = s_key

        self.models = []
        self.model_names = []

        for model in models:
            model_type = model if isinstance(model, type) else get_model(model)
            if model_type is None:
                raise ValueError(f"'{model}' is an invalid model!")

            self.models.append(model_type)
            self.model_names.append(model_type.__name__)

        self.x_data, self.y_data, self.s_data = [], [], []

        model_regs = [load_model_registry(m) for m in self.model_names]
        for reg in model_regs:
            x_val = self._get_key_value(reg, self.x_axis)
            y_val = self._get_key_value(reg, self.y_axis)

            s_val = None
            if self.s_key is not None:
                s_val = self._get_key_value(reg, self.s_key)

            if isinstance(x_val, dict) or isinstance(y_val, dict):
                raise ValueError(
                    f"Key pair '{self.x_axis}, {self.y_axis}' is"
                    + f" invalid for the model '{reg["name"]}'!"
                )

            self.x_data.append(x_val)
            self.y_data.append(y_val)

            if s_val is not None and isinstance(s_val, (int, float)):
                self.s_data.append(s_val)

    def _get_key_value(self, reg: dict, key: FormattedKey) -> Any:
        value = reg
        split_key = key.split(":")
        for k in split_key:
            value = value[k]
        return value

    def _scale(self, data: list[int | float]) -> list[int | float]:
        return [d / min(data) * 20 for d in data]

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        x_scale: str = "linear",
        y_scale: str = "linear",
        cmap: str = "viridis",
        scale_size: bool = False,
        grid: bool = True,
        title: Literal["auto"] | str = "auto",  # handle this further
        show: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        size_arr = self.s_data if self.s_data else None
        sc = ax.scatter(
            self.x_data, 
            self.y_data, 
            s=self._scale(size_arr) if scale_size else size_arr, 
            c=size_arr, 
            marker="o", 
            cmap=cmap, 
            alpha=0.7,
        )

        for x, y, name in zip(self.x_data, self.y_data, self.model_names):
            ax.text(
                x, 
                y, 
                name, 
                fontsize="x-small", 
                alpha=0.8, 
                horizontalalignment="center",
                verticalalignment="center",
            )

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        ax.set_xlabel(self.x_axis.split(":")[0])
        ax.set_ylabel(self.y_axis.split(":")[0])
        ax.set_title(title)

        if grid:
            ax.grid(alpha=0.2)
        
        cbar = ax.figure.colorbar(sc)
        cbar.set_label(self.s_key.split(":")[0])
        ax.figure.tight_layout()

        if show:
            plt.show()
            plt.savefig("test")  # Remove this later
        return ax
