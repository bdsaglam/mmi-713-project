from pathlib import Path
import numpy as np
import typer


def main(
    n: int = typer.Option(default=1000),
    dim: int = typer.Option(default=384),
    out: Path = typer.Option(...),
):
    points = np.random.rand(n, dim)
    with open(out, "w") as f:
        for point in points:
            f.write(" ".join(map(str, point)) + "\n")


if __name__ == "__main__":
    typer.run(main)
