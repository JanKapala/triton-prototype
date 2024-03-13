"""The whole implementation of the inference on the Triton"""

import os
from urllib import request

import torch
from PIL import Image
from torch import Tensor, jit, randn
from torch.nn import Module
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
    InferResult,
)

from constants import MODEL_REPOSITORY_PATH, PROJECT_ROOT_PATH

MODEL_REPOSITORY = "pytorch/vision:v0.10.0"
MODEL_NAME = "resnet18"
MODEL_PATH = os.path.join(MODEL_REPOSITORY_PATH, MODEL_NAME, "1", "model.pt")
LABELS_FILE_NAME = "imagenet_classes.txt"
LABELS_FILE_URL = os.path.join(
    "https://raw.githubusercontent.com/pytorch/hub/master/", LABELS_FILE_NAME
)
IMAGENET_LABELS_FILE_PATH = os.path.join(PROJECT_ROOT_PATH, LABELS_FILE_NAME)
IMAGE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"


def get_model() -> Module:
    model = torch.hub.load(  # type: ignore
        repo_or_dir=MODEL_REPOSITORY,
        model=MODEL_NAME,
        weights=ResNet18_Weights.DEFAULT,
        verbose=False,
    )
    return model.eval().cuda()  # type: ignore


def export_model_for_triton(model: Module) -> None:
    jit.save(  # type: ignore
        jit.trace(model, randn(1, 3, 224, 224).cuda()),  # type: ignore
        MODEL_PATH,
    )


def get_image() -> Image:
    filename = IMAGE_URL.rsplit("/", maxsplit=1)[-1]
    request.urlretrieve(IMAGE_URL, filename)
    return Image.open(filename)


def preprocess(image: Image) -> Tensor:
    pipeline = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return pipeline(image).unsqueeze(0).cuda()  # type: ignore


def infer(client: InferenceServerClient, model_input: Tensor) -> InferResult:
    model_input = model_input.cpu().squeeze().numpy()
    inputs = InferInput("input__0", model_input.shape, "FP32")
    inputs.set_data_from_numpy(model_input, False)
    outputs = InferRequestedOutput("output__0", False, 1000)
    return client.infer(MODEL_NAME, inputs=[inputs], outputs=[outputs])


def postprocess(result: InferResult) -> Tensor:
    raw_results = result.as_numpy("output__0")
    model_output = torch.empty(raw_results.shape, dtype=torch.float32)
    for raw_result in raw_results:
        raw_logit, raw_idx = raw_result.split(":")
        model_output[int(raw_idx)] = float(raw_logit)
    return torch.nn.functional.softmax(model_output, dim=0)


def get_labels() -> list[str]:
    request.urlretrieve(LABELS_FILE_URL, LABELS_FILE_URL.split("/")[-1])
    with open("imagenet_classes.txt", "r", encoding="UTF-8") as file:
        return [line.strip() for line in file.readlines()]


if __name__ == "__main__":
    triton_client = InferenceServerClient("localhost:8000")
    export_model_for_triton(get_model())
    probs = postprocess(infer(triton_client, preprocess(get_image())))
    print("TOP-5 Inference results for the dog.jpg image:")
    labels = get_labels()
    for p, idx in zip(*torch.topk(probs, 5)):
        print(f"{labels[idx]}: {p.item():.4f}")
