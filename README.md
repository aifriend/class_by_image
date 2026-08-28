# Class by Image

Computer vision toolkit for image classification, object detection, GAN-based generation, and data augmentation.

## Overview

A collection of deep learning models and utilities for image-based machine learning tasks, built with PyTorch.

## Modules

| Module | Description |
|--------|-------------|
| `classification/` | Image classification training (VGG11 transfer learning) |
| `object_detection/` | Object detection / recognition pipelines (VGG16) |
| `gan/` | Generative Adversarial Networks (DCGAN + context-encoder inpainting) |
| `augmentation/` | Data augmentation utilities (albumentations) |
| `common/` | Shared utilities |
| `tests/` | Unit tests (pytest) |

## Tech Stack

- **Language:** Python 3.9+
- **Frameworks:** PyTorch 2.x, torchvision
- **GPU:** CUDA optional — all scripts fall back to CPU

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Augment a dataset (writes copies into an "aug" subfolder)
python augmentation/aug_pipeline.py /path/to/images --aug-factor 10

# Train the document classifier (expects an ImageFolder dataset)
python classification/train_doc_class.py

# Run a VGG16 prediction on a sample image
python object_detection/predict_obj_detect.py

# Train the DCGAN
python gan/run_gan.py

# Train the context-encoder inpainting GAN
python gan/paint/run_inpaint.py --dataroot dataset/train --ngpu 0
```

## Tests

```bash
pip install pytest
python -m pytest tests/
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Jose** — [@aifriend](https://github.com/aifriend)
