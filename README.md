# OnnxSlim

<p align="center">
    <a href="https://pypi.org/project/onnxslim">
        <img src="https://img.shields.io/pypi/v/onnxslim?color=blue" />
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

- 🚀 2025/11/29: Top 1% on PyPI
- 🚀 2025/11/27: OnnxSlim is merged into [NVIDIA TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 🤗🤗🤗
- 🚀 2025/05/17: OnnxSlim is merged into [HuggingFace optimum](https://github.com/huggingface/optimum) 🤗🤗🤗
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

<table style="width:100%; border-collapse:separate; border-spacing:10px;">
  <tr>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/1728152?s=200&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/NVIDIA/TensorRT-Model-Optimizer" target="_blank">NVIDIA/TensorRT-Model-Optimizer</a>
    </td>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/alibaba/MNN" target="_blank">alibaba/MNN</a>
    </td>
  </tr>
  <tr>
  <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/26833451?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/ultralytics/ultralytics" target="_blank">ultralytics/ultralytics</a>
    </td>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/131524?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/mozilla/smart_autofill" target="_blank">Mozilla/smart_autofill</a>
    </td>
  </tr>
  <tr>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/wangzhaode/mnn-llm" target="_blank">alibaba/MNN-LLM</a>
    </td>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/25720743?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/huggingface/transformers.js" target="_blank">huggingface/transformers.js</a>
    </td>
  </tr>
  <tr>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/25720743?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/huggingface/optimum" target="_blank">huggingface/optimum</a>
    </td>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/23534030?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/PaddlePaddle/PaddleOCR" target="_blank">PaddlePaddle/PaddleOCR</a>
    </td>
  </tr>
  <tr>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/109945100?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/modelscope/FunASR" target="_blank">ModelScope/FunASR</a>
    </td>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/111754012?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/CVCUDA/CV-CUDA" target="_blank">CVCUDA/CV-CUDA</a>
    </td>
  </tr>
  <tr>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/86091366?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/THU-MIG/yolov10" target="_blank">THU-MIG/yolov10</a>
    </td>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/48153283?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/sunsmarterjie/yolov12" target="_blank">sunsmarterjie/yolov12</a>
    </td>
  </tr>
  <tr>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/147458884?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/nndeploy/nndeploy" target="_blank">nndeploy/nndeploy</a>
    </td>
    <td style="vertical-align:middle;">
      <img src="https://avatars.githubusercontent.com/u/126587470?s=48&v=4" width="22" height="22" style="vertical-align:middle; margin-right:8px;"/>
      <a href="https://github.com/deepghs/imgutils" target="_blank">deepghs/imgutils</a>
    </td>
  </tr>
</table>

# References

> - [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
> - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy)
> - [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
> - [tabulate](https://github.com/astanin/python-tabulate)
> - [onnxruntime](https://github.com/microsoft/onnxruntime)

# Contact

Discord: https://discord.gg/nRw2Fd3VUS QQ Group: `873569894`
