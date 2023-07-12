from pathlib import Path
from typing import Optional

import typer
from mrc import imread, imsave

app = typer.Typer(no_args_is_help=True)


@app.command(no_args_is_help=True)
def recon(
    file: Path = typer.Argument(..., help="The file to process"),
    otf: Path = typer.Argument(..., help="The OTF file to use"),
    output: Optional[Path] = typer.Option(None, help="The output file to write to"),
):
    from pycudasirecon.sim_reconstructor import reconstruct_multi

    typer.secho(locals())
    # return
    array = imread(str(file))
    reconstructed = reconstruct_multi(array, otf=imread(otf))
    typer.secho(reconstructed.shape)
    if output:
        imsave(str(output), reconstructed)


@app.command()
def register(name: str):
    typer.echo(f"Hello {name}!")


if __name__ == "__main__":
    app()
