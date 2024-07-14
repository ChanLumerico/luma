import platform

arc = platform.machine()

MLX_TOGGLE: bool = False

try:
    if MLX_TOGGLE:
        import mlx as _  # type: ignore
except ModuleNotFoundError as err:
    if arc != "arm64":
        print(f"Your python environment is not running on 'arm64', but on '{arc}'")
        print(
            f"Run luma/arm.sh to setup an ARM supported Python environment",
            f"and install MLX via: 'pip install mlx'.",
        )
    else:
        print(err)
