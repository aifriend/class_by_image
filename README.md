# Class by Image

Computer vision toolkit for image classification, object detection, GAN-based generation, and data augmentation.

## Overview

A collection of deep learning models and utilities for image-based machine learning tasks, built with PyTorch and TensorFlow.

## Modules

| Module | Description |
|--------|-------------|
| `classification/` | Image classification models |
| `object_detection/` | Object detection pipelines |
| `gan/` | Generative Adversarial Networks (DCGAN) |
| `augmentation/` | Data augmentation utilities |
| `common/` | Shared utilities |

## Tech Stack

- **Language:** Python 3.6–3.8
- **Frameworks:** PyTorch 1.9+, TensorFlow 2.4+
- **GPU:** CUDA 11.0/11.1, cuDNN 8.0

## Requirements

- Python 3.6+
- CUDA 11.x compatible GPU (recommended)

## Installation

```bash
pip install -r requirements.txt
```

For GPU support (PyTorch + CUDA 11.1):
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Jose** — [@aifriend](https://github.com/aifriend)
