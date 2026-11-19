"""Reproduce a controlled LeNet experiment on MNIST.

The test set is accessed for evaluation only after all three training runs
have finished. Best checkpoints are selected by validation cross entropy.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import sys
import time

# CUDA deterministic matrix multiplication must be configured before torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F
import torchvision
from torchvision.datasets import MNIST

from model import LeNet, integer_translate


MODEL_CONFIGS = {
    "baseline": {"activation": "sigmoid", "augmentation": False},
    "relu": {"activation": "relu", "augmentation": False},
    "shift": {"activation": "relu", "augmentation": True},
}
SHIFTS = {"up": (0, -2), "down": (0, 2), "left": (-2, 0), "right": (2, 0)}


class RunLog:
    def __init__(self, path):
        self.stream = path.open("w", encoding="utf-8")

    def write(self, message):
        print(message, flush=True)
        self.stream.write(message + "\n")
        self.stream.flush()


def get_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, threads, and learning rate must be positive")
    return args


def stratified_split(labels, seed):
    """Reserve exactly 500 samples of each class for validation."""
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for digit in range(10):
        indices = np.flatnonzero(labels.numpy() == digit)
        rng.shuffle(indices)
        validation.extend(indices[:500])
        training.extend(indices[500:])
    return np.asarray(training, dtype=np.int64), np.asarray(validation, dtype=np.int64)


@torch.inference_mode()
def evaluate(model, images, labels, batch_size, device, shift=None):
    model.eval()
    logits_parts = []
    total_loss = 0.0
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size].to(device)
        y = labels[start:start + batch_size].to(device)
        if shift is not None:
            x = integer_translate(x, *shift)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        logits_parts.append(logits.cpu())
    logits = torch.cat(logits_parts)
    probabilities = torch.softmax(logits, dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = int((prediction == labels).sum())
    return {
        "loss": total_loss / len(images),
        "accuracy": correct / len(images),
        "correct": correct,
        "total": len(images),
        "prediction": prediction.numpy(),
        "confidence": confidence.numpy(),
        "probabilities": probabilities.numpy(),
    }


def classification_metrics(evaluation, labels):
    prediction = evaluation["prediction"]
    cm = np.bincount(10 * labels.numpy() + prediction, minlength=100).reshape(10, 10)
    precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(10), where=cm.sum(axis=0) > 0)
    recall = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(10), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=(precision + recall) > 0)
    per_class = [
        {"label": digit, "precision": float(precision[digit]), "recall": float(recall[digit]),
         "f1": float(f1[digit]), "support": int(cm[digit].sum())}
        for digit in range(10)
    ]
    return {
        key: evaluation[key] for key in ("accuracy", "loss", "correct", "total")
    } | {"macro_f1": float(f1.mean()), "per_class": per_class, "confusion_matrix": cm.tolist()}


def train_one(name, config, initial_state, images, labels, validation_images,
              validation_labels, args, device, checkpoint_dir, history_writer, history_stream, log):
    model = LeNet(config["activation"]).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_epoch = None
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        # Separate generators prevent augmentation from changing sample order.
        order_rng = torch.Generator().manual_seed(args.seed + epoch)
        augmentation_rng = torch.Generator().manual_seed(args.seed + 10000 + epoch)
        indices = torch.randperm(len(images), generator=order_rng)
        total_loss, correct = 0.0, 0
        for start in range(0, len(indices), args.batch_size):
            index = indices[start:start + args.batch_size]
            x, y = images[index].to(device), labels[index].to(device)
            if config["augmentation"]:
                offsets = torch.randint(-2, 3, (len(index), 2), generator=augmentation_rng).to(device)
                x = integer_translate(x, offsets[:, 0], offsets[:, 1])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(index)
            correct += int((logits.argmax(dim=1) == y).sum())
        validation = evaluate(model, validation_images, validation_labels, args.batch_size, device)
        seconds = time.perf_counter() - epoch_start
        row = {"model": name, "epoch": epoch, "train_loss": total_loss / len(images),
               "train_accuracy": correct / len(images), "val_loss": validation["loss"],
               "val_accuracy": validation["accuracy"], "seconds": seconds}
        history_writer.writerow(row)
        history_stream.flush()
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save({"model_state_dict": model.state_dict(), "activation": config["activation"],
                        "epoch": epoch, "validation_loss": best_loss}, checkpoint_dir / f"{name}.pt")
        log.write(f"{name} epoch {epoch:02d}/{args.epochs} train_loss={row['train_loss']:.5f} "
                  f"train_acc={row['train_accuracy']:.5f} val_loss={row['val_loss']:.5f} "
                  f"val_acc={row['val_accuracy']:.5f} seconds={seconds:.2f}")
    return {"best_epoch": best_epoch, "validation_loss": best_loss,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_seconds": time.perf_counter() - training_start, **config}


def main():
    args = get_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    default_output = Path(__file__).resolve().parent / "results"
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = args.output.parent / "checkpoints" if args.output.resolve() == default_output else args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(args.output / "run.log")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    environment = {"python": platform.python_version(), "pytorch": torch.__version__,
                   "torchvision": torchvision.__version__, "numpy": np.__version__,
                   "device": str(device), "threads": args.threads, "platform": platform.platform()}
    log.write(json.dumps(environment, ensure_ascii=False))
    source = MNIST(args.data_dir, train=True, download=True)
    train_indices, val_indices = stratified_split(source.targets, args.seed)
    np.savez_compressed(args.output / "split_indices.npz", train=train_indices, validation=val_indices)
    split_hash = hashlib.sha256(train_indices.tobytes() + val_indices.tobytes()).hexdigest()
    train_images = source.data[train_indices].unsqueeze(1).float() / 255
    train_labels = source.targets[train_indices]
    val_images = source.data[val_indices].unsqueeze(1).float() / 255
    val_labels = source.targets[val_indices]
    # Reset after data preparation so every run starts with identical parameters.
    torch.manual_seed(args.seed)
    initial_state = copy.deepcopy(LeNet("sigmoid").state_dict())
    assert sum(t.numel() for t in initial_state.values()) == 61706
    summary = {"schema_version": 1, "environment": environment,
               "config": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                          "optimizer": "Adam", "learning_rate": args.learning_rate,
                          "selection": "minimum validation cross entropy", "normalization": "pixel / 255",
                          "training_translation": "independent integer dx,dy sampled uniformly from [-2,2]",
                          "test_translation": SHIFTS},
               "data_split": {"train": len(train_indices), "validation": len(val_indices), "test": 10000,
                              "validation_per_class": 500, "sha256": split_hash,
                              "train_per_class": torch.bincount(train_labels, minlength=10).tolist()},
               "models": {}}
    with (args.output / "histories.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "epoch", "train_loss", "train_accuracy",
                                                   "val_loss", "val_accuracy", "seconds"))
        writer.writeheader()
        for name, config in MODEL_CONFIGS.items():
            summary["models"][name] = train_one(name, config, initial_state, train_images, train_labels,
                                                 val_images, val_labels, args, device, checkpoint_dir,
                                                 writer, stream, log)
    log.write("All training and validation selection finished. Beginning locked test evaluation.")
    testing = MNIST(args.data_dir, train=False, download=True)
    test_images = testing.data.unsqueeze(1).float() / 255
    test_labels = testing.targets
    saved_arrays = {"test_images": testing.data.numpy(), "test_labels": test_labels.numpy()}
    predictions = {}
    for name, config in MODEL_CONFIGS.items():
        model = LeNet(config["activation"]).to(device)
        state = torch.load(checkpoint_dir / f"{name}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        clean = evaluate(model, test_images, test_labels, args.batch_size, device)
        summary["models"][name]["clean"] = classification_metrics(clean, test_labels)
        robust = {}
        predictions[name] = {"clean": clean}
        for direction, displacement in SHIFTS.items():
            shifted = evaluate(model, test_images, test_labels, args.batch_size, device, displacement)
            robust[direction] = {"accuracy": shifted["accuracy"], "loss": shifted["loss"],
                                 "drop": clean["accuracy"] - shifted["accuracy"]}
            predictions[name][direction] = shifted
        robust["mean_shift_accuracy"] = float(np.mean([robust[k]["accuracy"] for k in SHIFTS]))
        summary["models"][name]["robustness"] = robust
        error_indices = np.flatnonzero(clean["prediction"] != test_labels.numpy())
        error_order = error_indices[np.argsort(-clean["confidence"][error_indices], kind="stable")]
        summary["models"][name]["top_errors"] = [
            {"index": int(i), "true": int(test_labels[i]), "pred": int(clean["prediction"][i]),
             "confidence": float(clean["confidence"][i])} for i in error_order[:20]
        ]
        summary["models"][name]["feature_sample"] = {"index": 0, "true": int(test_labels[0]),
                                                       "pred": int(clean["prediction"][0])}
        with torch.inference_mode():
            for layer, values in model.features(test_images[:1].to(device)).items():
                saved_arrays[f"{name}_{layer}"] = values[0].cpu().numpy()
        saved_arrays[f"{name}_probabilities"] = clean["probabilities"]
        log.write(f"TEST {name} accuracy={clean['accuracy']:.5f} "
                  f"macro_f1={summary['models'][name]['clean']['macro_f1']:.5f} "
                  f"shift_mean={robust['mean_shift_accuracy']:.5f}")
    fields = ["index", "true"]
    for name in MODEL_CONFIGS:
        fields.extend([f"{name}_pred", f"{name}_confidence"] + [f"{name}_{k}_pred" for k in SHIFTS])
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(len(test_labels)):
            row = {"index": i, "true": int(test_labels[i])}
            for name, values in predictions.items():
                row[f"{name}_pred"] = int(values["clean"]["prediction"][i])
                row[f"{name}_confidence"] = float(values["clean"]["confidence"][i])
                for direction in SHIFTS:
                    row[f"{name}_{direction}_pred"] = int(values[direction]["prediction"][i])
            writer.writerow(row)
    np.savez_compressed(args.output / "analysis.npz", **saved_arrays)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.write("Saved summary.json, histories.csv, predictions.csv, split_indices.npz, analysis.npz, and checkpoints.")
    log.stream.close()


if __name__ == "__main__":
    main()
