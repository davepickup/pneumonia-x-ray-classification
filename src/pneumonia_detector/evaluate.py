"""Evaluate Pneumonia Detector models against test set"""
import argparse
import json
import os

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import transforms

from pneumonia_detector.model import PneumoniaClassifier
from pneumonia_detector.preprocess import XrayDataset


def evaluate_model(model_dir: str, model_filename: str, test_dir: str):
    """
    Function to evaluate a chosen model against a test set

    Parameters
    ----------
    model_dir : str
        path to model directory.
    model_filename : str
        model filename as appears in model directory.
    test_dir : str
        path to directory holding the test set to evaluate on.

    """

    with open(os.path.join(model_dir, f"{model_filename}_params.json"), "r") as f:
        params = json.load(f)

    device = params["device"]
    model = PneumoniaClassifier(image_size=params["image_size"])
    model.load_state_dict(
        torch.load(os.path.join(model_dir, model_filename), map_location=device)
    )
    model.to(device)

    test_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4823, 0.4823, 0.4823], std=[0.2363, 0.2363, 0.2363]
            ),
        ]
    )

    test_dataset = XrayDataset(root_dir=test_dir, transform=test_transforms)

    model.eval()
    with torch.no_grad():
        y_preds, y_true = zip(
            *[
                (model(item[0].unsqueeze(0)).argmax().item(), item[1])
                for item in test_dataset
            ]
        )

    print("\n******EVALUATION RESULTS******\n")
    print(f"Model accuracy: {accuracy_score(y_true, y_preds)}")
    print("\n\n")
    print(f"Confusion matrix:\n {confusion_matrix(y_true, y_preds)}")
    print("\n\n")
    print("Classification Report:\n")
    print(classification_report(y_true, y_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="evaluate_model",
        description="Evaluates a PneumoniaClassifier model against a given test set",
    )
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_filename", type=str)
    parser.add_argument("--test_dir", type=str)
    args = parser.parse_args()

    evaluate_model(**vars(args))
