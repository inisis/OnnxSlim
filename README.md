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
</p>

OnnxSlim can help you slim your onnx model, with less operators, but same accuracy, better inference speed.

- 🚀 2025/04/30: Rank 1st in the [AICAS 2025 LLM inference optimization challenge](https://tianchi.aliyun.com/competition/entrance/532289/customize588)
- 🚀 2025/01/28: Achieved 1M downloads
- 🚀 2024/06/23: OnnxSlim is merged into [transformers.js](https://github.com/xenova/transformers.js) 🤗🤗🤗
- 🚀 2024/06/02: OnnxSlim is merged into [ultralytics](https://github.com/ultralytics/ultralytics) ❤️❤️❤️
- 🚀 2024/04/30: Rank 1st in the [AICAS 2024 LLM inference optimization challenge](https://tianchi.aliyun.com/competition/entrance/532170/customize440) held by Arm and T-head
- 🚀 2024/01/25: OnnxSlim is merged to [mnn-llm](https://github.com/wangzhaode/mnn-llm), performance increased by 5%

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

```
onnxslim your_onnx_model slimmed_onnx_model
```

<div align=left><img src="https://raw.githubusercontent.com/inisis/onnxslim/main/images/onnxslim.gif"></div>

For more usage, see onnxslim -h or refer to our [examples](./examples)

# Projects using OnnxSlim

- <img src="https://avatars.githubusercontent.com/u/131524?s=48&v=4" width="22" height="22"/>[Mozilla/smart_autofill](https://github.com/mozilla/smart_autofill)
- <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" width="22" height="22"/>[alibaba/MNN](https://github.com/alibaba/MNN)
- <img src="https://avatars.githubusercontent.com/u/23534030?s=48&v=4" width="22" height="22"/>[PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- <img src="https://avatars.githubusercontent.com/u/25720743?s=48&v=4" width="22" height="22"/>[huggingface/transformers.js](https://github.com/huggingface/transformers.js)
- <img src="https://avatars.githubusercontent.com/u/86091366?s=48&v=4" width="22" height="22"/>[THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)
- <img src="https://avatars.githubusercontent.com/u/26833451?s=48&v=4" width="22" height="22"/>[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- <img src="https://avatars.githubusercontent.com/u/109945100?s=48&v=4" width="22" height="22"/>[ModelScope/FunASR](https://github.com/modelscope/FunASR)
- <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" width="22" height="22"/>[alibaba/MNN-LLM](https://github.com/wangzhaode/mnn-llm)
- <img src="https://avatars.githubusercontent.com/u/126587470?s=48&v=4" width="22" height="22"/>[deepghs/imgutils](https://github.com/deepghs/imgutils)
- <img src="https://avatars.githubusercontent.com/u/48153283?s=48&v=4" width="22" height="22"/>[sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)

# References

> - [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
> - [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy)
> - [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
> - [tabulate](https://github.com/astanin/python-tabulate)
> - [onnxruntime](https://github.com/microsoft/onnxruntime)

# Contact

Discord: https://discord.gg/nRw2Fd3VUS QQ Group: 873569894
