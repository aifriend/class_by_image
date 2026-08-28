# Download an example image from the pytorch website and classify it
import os
import urllib.request

import torch
from PIL import Image
from torchvision import transforms, models

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights="IMAGENET1K_V1")
    model.to(device)
    model.eval()

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # The output has unnormalized scores. To get probabilities, run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Read the categories
    with open(os.path.join(SCRIPT_DIR, "predict", "imagenet_classes.txt"), "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == '__main__':
    main()
