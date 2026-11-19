"""Verify exported MNIST experiment results using only NumPy and the standard library.

Run ``python verify_results.py`` after training finishes, or supply another
results directory with ``--results``. This checks consistency between exported
files. The per-class validation count is checked as a summary declaration;
the verifier does not download MNIST or independently authenticate its labels.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np


MODELS = ("baseline", "relu", "shift")
DIRECTIONS = ("up", "down", "left", "right")
INPUT_FILES = ("summary.json", "histories.csv", "predictions.csv", "split_indices.npz")
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    actual, expected = float(actual), float(expected)
    require(math.isfinite(actual) and math.isfinite(expected), f"{name} must be finite")
    require(abs(actual - expected) <= TOLERANCE,
            f"{name} mismatch: recomputed {actual:.16g}, saved {expected:.16g}")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path, required_fields):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"{path.name} has no header")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)),
                f"{path.name} has duplicate column names")
        require(set(required_fields).issubset(reader.fieldnames),
                f"{path.name} is missing required columns")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values())
                for row in rows), f"{path.name} has incomplete or extra CSV fields")
    return rows


def check_split(results, summary):
    declared = summary["data_split"]
    require(declared["train"] == 55000, "Expected 55,000 training samples")
    require(declared["validation"] == 5000, "Expected 5,000 validation samples")
    require(declared["test"] == 10000, "Expected 10,000 test samples")
    require(declared["validation_per_class"] == 500,
            "Summary must declare 500 validation samples per class")
    train_per_class = np.asarray(declared["train_per_class"])
    require(train_per_class.shape == (10,) and train_per_class.dtype.kind in "iu",
            "train_per_class must contain ten integer counts")
    require(np.all(train_per_class >= 0) and int(train_per_class.sum()) == 55000,
            "Declared training class counts do not sum to 55,000")
    with np.load(results / "split_indices.npz", allow_pickle=False) as split:
        training = split["train"]
        validation = split["validation"]
    for name, values, expected in (("train", training, 55000),
                                   ("validation", validation, 5000)):
        require(values.ndim == 1 and values.dtype.kind in "iu",
                f"{name} indices must be a one-dimensional integer array")
        require(len(values) == expected, f"Unexpected {name} index count")
        require(len(np.unique(values)) == expected, f"Duplicate {name} indices")
        require(np.all((values >= 0) & (values < 60000)),
                f"{name} contains an index outside the training dataset")
    require(np.intersect1d(training, validation).size == 0,
            "Training and validation indices overlap")
    combined = np.concatenate((training, validation))
    require(np.array_equal(np.sort(combined), np.arange(60000)),
            "Training and validation do not cover all 60,000 source indices")
    digest = hashlib.sha256(training.tobytes() + validation.tobytes()).hexdigest()
    require(digest == declared["sha256"], "Split SHA-256 differs from summary")
    return {"train": len(training), "validation": len(validation),
            "test": declared["test"], "sha256": digest,
            "validation_per_class": 500,
            "class_balance_check": "summary declaration only; raw MNIST labels not read"}


def check_histories(results, summary):
    epochs = summary["config"]["epochs"]
    require(isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0,
            "config.epochs must be a positive integer")
    fields = ("model", "epoch", "train_loss", "train_accuracy", "val_loss",
              "val_accuracy", "seconds")
    rows = read_csv(results / "histories.csv", fields)
    require(len(rows) == len(MODELS) * epochs, "History row count is incomplete")
    grouped = {name: {} for name in MODELS}
    for row in rows:
        name = row["model"]
        require(name in grouped, f"Unknown history model: {name}")
        epoch = int(row["epoch"])
        require(epoch not in grouped[name], f"Duplicate history epoch for {name}")
        for field in fields[2:]:
            value = float(row[field])
            require(math.isfinite(value) and value >= 0,
                    f"{name} epoch {epoch}: invalid {field}")
            if field.endswith("accuracy"):
                require(value <= 1, f"{name} epoch {epoch}: {field} exceeds 1")
        grouped[name][epoch] = row
    checks = {}
    for name, records in grouped.items():
        require(set(records) == set(range(1, epochs + 1)),
                f"{name} does not contain every configured epoch")
        best_epoch = min(sorted(records), key=lambda epoch: float(records[epoch]["val_loss"]))
        model = summary["models"][name]
        require(model["best_epoch"] == best_epoch,
                f"{name}: checkpoint epoch differs from minimum validation loss")
        close(records[best_epoch]["val_loss"], model["validation_loss"],
              f"{name} checkpoint validation loss")
        checks[name] = {"epochs": len(records), "best_epoch": best_epoch}
    return checks


def check_predictions(results, summary):
    fields = ["index", "true"]
    for name in MODELS:
        fields.extend([f"{name}_pred", f"{name}_confidence"])
        fields.extend(f"{name}_{direction}_pred" for direction in DIRECTIONS)
    rows = read_csv(results / "predictions.csv", fields)
    expected_count = summary["data_split"]["test"]
    require(len(rows) == expected_count, "Prediction row count differs from test size")
    indices = np.asarray([int(row["index"]) for row in rows])
    require(np.array_equal(np.sort(indices), np.arange(expected_count)),
            "Test indices must contain 0 through 9999 exactly once")
    rows = [rows[i] for i in np.argsort(indices)]
    labels = np.asarray([int(row["true"]) for row in rows], dtype=np.int64)
    require(np.all((labels >= 0) & (labels <= 9)), "Invalid true class in prediction CSV")
    checks = {}
    for name in MODELS:
        prediction = np.asarray([int(row[f"{name}_pred"]) for row in rows])
        confidence = np.asarray([float(row[f"{name}_confidence"]) for row in rows])
        require(np.all((prediction >= 0) & (prediction <= 9)),
                f"{name}: invalid predicted class")
        require(np.all(np.isfinite(confidence) & (confidence >= 0) & (confidence <= 1)),
                f"{name}: invalid confidence")
        saved = summary["models"][name]
        clean = saved["clean"]
        count = len(labels)
        correct = int(np.count_nonzero(prediction == labels))
        error_count = int(np.count_nonzero(prediction != labels))
        require(clean["total"] == count, f"{name}: saved clean total differs from CSV")
        require(clean["correct"] == correct, f"{name}: clean correct count mismatch")
        require(clean["total"] - clean["correct"] == error_count,
                f"{name}: clean error count mismatch")
        accuracy = correct / count
        close(accuracy, clean["accuracy"], f"{name} clean accuracy")
        cm = np.bincount(10 * labels + prediction, minlength=100).reshape(10, 10)
        saved_cm = np.asarray(clean["confusion_matrix"])
        require(saved_cm.shape == (10, 10) and saved_cm.dtype.kind in "iu",
                f"{name}: invalid saved confusion matrix")
        require(np.array_equal(cm, saved_cm), f"{name}: confusion matrix mismatch")
        precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(10),
                              where=cm.sum(axis=0) > 0)
        recall = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(10),
                           where=cm.sum(axis=1) > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10),
                       where=precision + recall > 0)
        close(f1.mean(), clean["macro_f1"], f"{name} macro-F1")
        per_class = clean["per_class"]
        require(len(per_class) == 10, f"{name}: per-class results must contain ten rows")
        require({row["label"] for row in per_class} == set(range(10)),
                f"{name}: duplicate or missing per-class labels")
        for row in per_class:
            digit = row["label"]
            require(row["support"] == int(cm[digit].sum()), f"{name}: class support mismatch")
            for metric, values in (("precision", precision), ("recall", recall), ("f1", f1)):
                close(values[digit], row[metric], f"{name} class {digit} {metric}")
        error_indices = np.flatnonzero(prediction != labels)
        ranked_errors = error_indices[np.argsort(-confidence[error_indices], kind="stable")][:20]
        require([row["index"] for row in saved["top_errors"]] == ranked_errors.tolist(),
                f"{name}: top-error selection differs from prediction CSV")
        for row in saved["top_errors"]:
            i = row["index"]
            require(row["true"] == int(labels[i]) and row["pred"] == int(prediction[i]),
                    f"{name}: top-error labels differ from prediction CSV")
            close(confidence[i], row["confidence"], f"{name} top-error confidence")
        robustness = saved["robustness"]
        shift_accuracies = {}
        for direction in DIRECTIONS:
            shifted = np.asarray([int(row[f"{name}_{direction}_pred"]) for row in rows])
            require(np.all((shifted >= 0) & (shifted <= 9)),
                    f"{name} {direction}: invalid predicted class")
            shifted_accuracy = float(np.count_nonzero(shifted == labels) / count)
            close(shifted_accuracy, robustness[direction]["accuracy"],
                  f"{name} {direction} accuracy")
            close(accuracy - shifted_accuracy, robustness[direction]["drop"],
                  f"{name} {direction} accuracy drop")
            shift_accuracies[direction] = shifted_accuracy
        mean_shift = float(np.mean(list(shift_accuracies.values())))
        close(mean_shift, robustness["mean_shift_accuracy"], f"{name} mean shift accuracy")
        checks[name] = {"total": count, "correct": correct, "errors": error_count,
                        "accuracy": accuracy, "macro_f1": float(f1.mean()),
                        "shift_accuracy": shift_accuracies, "mean_shift_accuracy": mean_shift}
    return checks


def verify(results):
    for filename in INPUT_FILES:
        require((results / filename).is_file(), f"Missing required input: {filename}")
    hashes = {filename: file_hash(results / filename) for filename in INPUT_FILES}
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    require(summary["schema_version"] == 1, "Unsupported summary schema version")
    require(set(summary["models"]) == set(MODELS), "Expected baseline, relu, and shift models")
    split = check_split(results, summary)
    histories = check_histories(results, summary)
    predictions = check_predictions(results, summary)
    require(hashes == {filename: file_hash(results / filename) for filename in INPUT_FILES},
            "Input files changed during verification; run again after training finishes")
    return {"schema_version": 1, "status": "PASS", "absolute_tolerance": TOLERANCE,
            "scope": "export consistency; does not authenticate MNIST labels or rerun inference",
            "input_sha256": hashes, "split": split, "histories": histories,
            "predictions": predictions}


def write_report(path, report):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    try:
        report = verify(args.results)
    except (AssertionError, ValueError, KeyError, TypeError, OSError, csv.Error) as error:
        if args.results.is_dir():
            write_report(args.results / "verification.json", {
                "schema_version": 1, "status": "FAIL", "error_type": type(error).__name__,
                "message": "Result verification failed; inspect the console error and source inputs."})
        raise ValueError(f"Result verification failed: {error}") from error
    write_report(args.results / "verification.json", report)
    print("PASS: split, epoch histories, 10,000 prediction rows, confusion matrices, "
          "macro-F1, and four-direction accuracies agree. Saved verification.json.")


if __name__ == "__main__":
    main()
