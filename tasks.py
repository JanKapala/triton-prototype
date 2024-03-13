# pylint: disable=missing-function-docstring

"""Project tasks that can be run via command line using inv command."""

from invoke import Context, task  # type: ignore[attr-defined]

from constants import MODEL_REPOSITORY_PATH, PROJECT_ROOT_PATH


@task
def mypy(c: Context) -> None:
    c.run(
        f"poetry run mypy --install-types --non-interactive {PROJECT_ROOT_PATH}"
    )


@task
def black(c: Context, only_check: bool = False) -> None:
    if only_check:
        command = f"poetry run black --check {PROJECT_ROOT_PATH}"
    else:
        command = f"poetry run black {PROJECT_ROOT_PATH}"
    c.run(command)


@task
def isort(c: Context, only_check: bool = False) -> None:
    if only_check:
        command = f"poetry run isort --check {PROJECT_ROOT_PATH}"
    else:
        command = f"poetry run isort {PROJECT_ROOT_PATH}"
    c.run(command)


@task
def lint(c: Context) -> None:
    c.run(f"poetry run pylint --jobs=0 --recursive=y {PROJECT_ROOT_PATH}")


@task
def triton_server(c: Context) -> None:
    c.run(
        f"docker run --gpus=1 --rm --net=host -v "
        f"{MODEL_REPOSITORY_PATH}:/models "
        f"nvcr.io/nvidia/tritonserver:24.02-py3 "
        f"tritonserver --model-repository=/models"
    )


@task
def infer(c: Context) -> None:
    c.run("python main.py")
