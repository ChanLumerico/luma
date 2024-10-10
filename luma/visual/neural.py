from typing import Any, List, Optional, Type
import matplotlib.pyplot as plt

from luma.core.super import Visualizer
from luma.neural.base import NeuralModel
from luma.neural.model import get_model, load_model_registry, load_entire_registry


__all__ = ("ModelScatterPlot", "ModelFamilyPlot")


FormattedKey = str


def _get_key_value(reg: dict, key: FormattedKey) -> Any:
    value = reg
    split_key = key.split(":")
    for k in split_key:
        value = value[k]
    return value


def _scale(data: list[int | float], scale: int = 25) -> list[int | float]:
    return [d / min(data) * scale for d in data]


def _format_number(num: float, decimals: int = 1) -> str:
    suffixes = ["K", "M", "B", "T", "P", "E", "Z", "Y"]

    magnitude = 0
    while abs(num) >= 1000 and magnitude < len(suffixes):
        num /= 1000.0
        magnitude += 1

    formatted_num = f"{num:.{decimals}f}"
    if magnitude > 0:
        formatted_num += suffixes[magnitude - 1]

    return formatted_num


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

        self.model_regs = [load_model_registry(m) for m in self.model_names]
        for reg in self.model_regs:
            x_val = _get_key_value(reg, self.x_axis)
            y_val = _get_key_value(reg, self.y_axis)

            s_val = None
            if self.s_key is not None:
                s_val = _get_key_value(reg, self.s_key)

            if isinstance(x_val, dict) or isinstance(y_val, dict):
                raise ValueError(
                    f"Key pair '{self.x_axis}, {self.y_axis}' is"
                    + f" invalid for the model '{reg["name"]}'!"
                )

            self.x_data.append(x_val)
            self.y_data.append(y_val)

            if s_val is not None and isinstance(s_val, (int, float)):
                self.s_data.append(s_val)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        x_scale: str = "linear",
        y_scale: str = "linear",
        cmap: str = "viridis",
        scale_size: bool = False,
        scale_factor: int = 25,
        grid: bool = True,
        title: Optional[str] = None,
        show: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        size_arr = self.s_data if self.s_data else None
        sc = ax.scatter(
            self.x_data,
            self.y_data,
            s=_scale(size_arr, scale_factor) if scale_size else size_arr,
            c=size_arr if size_arr else self.y_data,
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
                ha="center",
                va="bottom",
            )
        for x, y, reg in zip(self.x_data, self.y_data, self.model_regs):
            ax.text(
                x,
                y,
                _format_number(reg["params"]),
                fontsize="x-small",
                alpha=0.5,
                ha="center",
                va="top",
            )

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        ax.set_xlabel(self.x_axis.split(":")[0])
        ax.set_ylabel(self.y_axis.split(":")[0])
        if title is not None:
            ax.set_title(title)

        if grid:
            ax.grid(alpha=0.2)

        cbar = ax.figure.colorbar(sc)
        if self.s_key is not None:
            cbar.set_label(self.s_key.split(":")[0])

        ax.figure.tight_layout()
        if show:
            plt.show()
            plt.savefig(title)

        return ax


class ModelFamilyPlot(Visualizer):
    def __init__(
        self,
        families: list[str],
        x_axis: FormattedKey,
        y_axis: FormattedKey,
    ) -> None:
        self.families = families
        self.x_axis = x_axis
        self.y_axis = y_axis

        self.model_families = []
        reg_json = load_entire_registry()

        for fam_name in families:
            model_reg_arr = []

            for reg_dict in reg_json:
                reg_fam_name: str = reg_dict["family"]
                if reg_fam_name is None:
                    continue

                alt_fam_name = reg_fam_name.lower().replace("_", "-")
                if reg_fam_name == fam_name or alt_fam_name == fam_name:
                    model_reg_arr.append(reg_dict)

            self.model_families.append(model_reg_arr)

        self.x_data_arr, self.y_data_arr = [], []
        for family_reg in self.model_families:
            x_data, y_data = [], []

            for model_reg in family_reg:
                x_val = _get_key_value(model_reg, self.x_axis)
                y_val = _get_key_value(model_reg, self.y_axis)

                x_data.append(x_val)
                y_data.append(y_val)

            self.x_data_arr.append(x_data)
            self.y_data_arr.append(y_data)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        x_scale: str = "linear",
        y_scale: str = "linear",
        grid: bool = True,
        title: Optional[str] = None,
        show: bool = False,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots()
            show = True

        markers = ["o", "d", "s", "x", "v", ">", "<"]
        for i, (x_data, y_data, reg_arr) in enumerate(
            zip(self.x_data_arr, self.y_data_arr, self.model_families)
        ):
            label = reg_arr[0]["family"] + " Family"
            ax.plot(
                x_data,
                y_data,
                marker=markers[i % len(markers)],
                alpha=0.8,
                label=label,
            )
            for x, y, reg in zip(x_data, y_data, reg_arr):
                ax.text(
                    x,
                    y,
                    reg["name"],
                    fontsize="x-small" if len(self.families) < 4 else "xx-small",
                    alpha=0.8,
                    ha="center",
                    va="center",
                )

        ax.legend(fontsize="x-small")
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        ax.set_xlabel(self.x_axis.split(":")[0])
        ax.set_ylabel(self.y_axis.split(":")[0])
        if title is not None:
            ax.set_title(title)

        if grid:
            ax.grid(alpha=0.2)

        ax.figure.tight_layout()
        if show:
            plt.show()

        return ax
