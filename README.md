# ResNet inference on the NVIDIA Triton
It is a minimal implementation (<100 LOC in the `main.py`) of the [ResNet](https://arxiv.org/abs/1512.03385) inference on the [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) with the **GPU**.
It is performed on the single `<repo-root-dir>/dog.jpg` image (will be downloaded automatically) from the [ImageNet](https://www.image-net.org/) dataset and can be easily scaled up.

### Setup
- Install [docker](https://docs.docker.com/)
- Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration)
- Install [pyenv](https://github.com/pyenv/pyenv)
- Install Python 3.11.5: `pyenv install 3.11.5`
- Clone this repo: `git clone git@github.com:JanKapala/triton-prototype.git`
- Go to the repo root dir: `cd <repo-root-dir>`
- Set python version: `pyenv local 3.11.5`
- Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
- Install repo dependencies: `poetry install`

### Run
- Run the NVIDIA Triton Server: `inv triton-server`
- When triton server is ready, run the inference: `inv infer`

You should see the following output:
```
TOP-5 Inference results for the dog.jpg image:
Samoyed: 0.8846
Arctic fox: 0.0458
white wolf: 0.0443
Pomeranian: 0.0056
Great Pyrenees: 0.0047
```

Feel free to modify the `main.py` and e.g.: perform inference on other models and datasets.