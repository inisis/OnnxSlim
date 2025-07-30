# OnnxSlim

<p align="center">
    <a href="https://pypi.org/project/onnxslim">
        <img src="https://badgen.net/pypi/v/onnxslim?color=blue" />
    </a>
    <a href="https://pypi.org/project/onnxslim">
        <img src="https://static.pepy.tech/badge/onnxslim/week" />
    </a>
    <a href="https://pypi.org/project/onnxslim">
        <img src="https://static.pepy.tech/badge/onnxslim/month" />
    </a>    
    <a href="https://pypi.org/project/onnxslim">
        <img src="https://static.pepy.tech/badge/onnxslim" />
    </a>   
    <a href="https://github.com/inisis/onnxslim/actions/workflows/ci.yaml">
        <img src="https://github.com/inisis/onnxslim/actions/workflows/ci.yml/badge.svg" />
    </a>
    <a href="https://codecov.io/gh/inisis/onnxslim" > 
        <img src="https://codecov.io/gh/inisis/onnxslim/branch/main/graph/badge.svg?token=C69ZH6802N"/> 
    </a>    
    <a href="https://muhammadrizwanmunawar.medium.com/boost-onnx-load-speed-by-10-15-with-onnxslims-python-package-d401eb8c2e69">
        <img src="https://img.shields.io/badge/Blog-OnnxSlim?style=flat&label=OnnxSlim" />
    </a>
    <a href="https://deepwiki.com/inisis/OnnxSlim"><img src="https://img.shields.io/badge/DeepWiki-inisis%2FOnnxSlim-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==" alt="DeepWiki"></a> 
</p>

OnnxSlim can help you slim your onnx model, with less operators, but same accuracy, better inference speed.

- 🚀 2025/05/17: OnnxSlim is merged into [optimum](https://github.com/huggingface/optimum) 🤗🤗🤗
- 🚀 2025/04/30: Rank 1st in the [AICAS 2025 LLM inference optimization challenge](https://tianchi.aliyun.com/competition/entrance/532289/customize588)
- 🚀 2025/01/28: Achieved 1M downloads
- 🚀 2024/06/23: OnnxSlim is merged into [transformers.js](https://github.com/huggingface/transformers.js) 🤗🤗🤗
- 🚀 2024/06/02: OnnxSlim is merged into [ultralytics](https://github.com/ultralytics/ultralytics) ❤️❤️❤️
- 🚀 2024/04/30: Rank 1st in the [AICAS 2024 LLM inference optimization challenge](https://tianchi.aliyun.com/competition/entrance/532170/customize440) held by Arm and T-head
- 🚀 2024/01/25: OnnxSlim is merged to [mnn-llm](https://github.com/wangzhaode/mnn-llm), performance increased by 5%

# Benchmark

![Image](https://github.com/user-attachments/assets/fefc79f1-5d8d-486b-935a-a088846b3900)

# Installation

## Using Prebuilt

```bash
pip install onnxslim
```

## Install From Source

```bash
pip install git+https://github.com/inisis/OnnxSlim@main
```

## Install From Local

```bash
git clone https://github.com/inisis/OnnxSlim && cd OnnxSlim/
pip install .
```

# How to use

## Bash

```bash
onnxslim your_onnx_model slimmed_onnx_model
```

<div align=left><img src="https://raw.githubusercontent.com/inisis/onnxslim/main/images/onnxslim.gif"></div>

## Inscript

```inscript
import onnx
import onnxslim

model = onnx.load("model.onnx")
slimmed_model = onnxslim.slim(model)
onnx.save(slimmed_model, "slimmed_model.onnx")
```

For more usage, see onnxslim -h or refer to our [examples](./examples)

# Projects using OnnxSlim

- <img src="https://avatars.githubusercontent.com/u/131524?s=48&v=4" width="22" height="22"/>[Mozilla/smart_autofill](https://github.com/mozilla/smart_autofill)
- <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" width="22" height="22"/>[alibaba/MNN](https://github.com/alibaba/MNN)
- <img src="https://avatars.githubusercontent.com/u/23534030?s=48&v=4" width="22" height="22"/>[PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- <img src="https://avatars.githubusercontent.com/u/25720743?s=48&v=4" width="22" height="22"/>[huggingface/transformers.js](https://github.com/huggingface/transformers.js)
- <img src="https://avatars.githubusercontent.com/u/25720743?s=48&v=4" width="22" height="22"/>[huggingface/optimum](https://github.com/huggingface/optimum)
- <img src="https://avatars.githubusercontent.com/u/86091366?s=48&v=4" width="22" height="22"/>[THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- <img src="https://avatars.githubusercontent.com/u/26833451?s=48&v=4" width="22" height="22"/>[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- <img src="https://avatars.githubusercontent.com/u/109945100?s=48&v=4" width="22" height="22"/>[ModelScope/FunASR](https://github.com/modelscope/FunASR)
- <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" width="22" height="22"/>[alibaba/MNN-LLM](https://github.com/wangzhaode/mnn-llm)
- <img src="https://avatars.githubusercontent.com/u/126587470?s=48&v=4" width="22" height="22"/>[deepghs/imgutils](https://github.com/deepghs/imgutils)
- <img src="https://avatars.githubusercontent.com/u/48153283?s=48&v=4" width="22" height="22"/>[sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)
- <img src="https://avatars.githubusercontent.com/u/147458884?s=48&v=4" width="22" height="22"/>[nndeploy/nndeploy](https://github.com/nndeploy/nndeploy)

# References

> - [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
> - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy)
> - [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
> - [tabulate](https://github.com/astanin/python-tabulate)
> - [onnxruntime](https://github.com/microsoft/onnxruntime)

# Contact

Discord: https://discord.gg/nRw2Fd3VUS QQ Group: `873569894`
