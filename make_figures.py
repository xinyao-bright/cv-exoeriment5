"""Create report figures from the saved, completed LeNet experiment.

No model training or test-set model selection takes place in this script.
Run from the repository root with ``python make_figures.py``.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("baseline", "relu", "shift")
NAMES = {"baseline": "A 原始 Sigmoid", "relu": "B ReLU", "shift": "C ReLU + 平移增强"}
COLORS = {"baseline": "#3262a8", "relu": "#d47d25", "shift": "#218377"}
DIRECTIONS = ("up", "down", "left", "right")
DIRECTION_NAMES = {"up": "上移 2 像素", "down": "下移 2 像素", "left": "左移 2 像素", "right": "右移 2 像素"}


def save_figure(figure, destination):
    figure.savefig(destination, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def read_histories(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            rows.append({key: value if key == "model" else float(value) for key, value in source.items()})
    return rows


def draw_learning_curves(histories, summary, output):
    figure, axes = plt.subplots(2, 2, figsize=(12.3, 8), layout="constrained")
    panels = [("train_loss", "训练交叉熵", "损失"), ("val_loss", "验证交叉熵", "损失"),
              ("train_accuracy", "训练准确率", "准确率 / %"), ("val_accuracy", "验证准确率", "准确率 / %")]
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        for name in MODELS:
            rows = [row for row in histories if row["model"] == name]
            values = [row[key] * (100 if "accuracy" in key else 1) for row in rows]
            axis.plot([row["epoch"] for row in rows], values, label=NAMES[name], color=COLORS[name], linewidth=1.7)
            if key == "val_loss":
                chosen = summary["models"][name]["best_epoch"]
                chosen_row = next(row for row in rows if row["epoch"] == chosen)
                axis.scatter(chosen, chosen_row[key], color=COLORS[name], s=35, zorder=4)
        axis.set(title=title, xlabel="训练轮次", ylabel=ylabel)
        axis.grid(alpha=.22)
        axis.set_xlim(1, summary["config"]["epochs"])
        if "accuracy" in key:
            axis.set_ylim(0, 101)
    axes[0, 0].legend(fontsize=10, frameon=False)
    figure.suptitle("共享划分与训练预算的三组对照  ·  验证损失圆点标出选中轮次", fontsize=15)
    save_figure(figure, output / "learning_curves.png")


def draw_confusions(summary, output):
    figure, axes = plt.subplots(1, 3, figsize=(16.8, 5.3), layout="constrained")
    for axis, name in zip(axes, MODELS):
        matrix = np.asarray(summary["models"][name]["clean"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for (row, col), count in np.ndenumerate(matrix):
            if count:
                axis.text(col, row, str(count), ha="center", va="center", fontsize=8,
                          color="white" if count > matrix.max()*.48 else "#20344d")
        axis.set(title=NAMES[name], xlabel="预测类别", ylabel="真实类别", xticks=range(10), yticks=range(10))
    figure.suptitle("官方 10,000 张测试图像的混淆矩阵", fontsize=15)
    save_figure(figure, output / "confusion_matrices.png")

    figure, axis = plt.subplots(figsize=(7.2, 6.5), layout="constrained")
    matrix = np.asarray(summary["models"]["shift"]["clean"]["confusion_matrix"])
    image = axis.imshow(matrix, cmap="Blues")
    for (row, col), count in np.ndenumerate(matrix):
        if count:
            axis.text(col, row, str(count), ha="center", va="center", fontsize=10,
                      color="white" if count > matrix.max()*.48 else "#20344d")
    axis.set(title="C 组干净测试集混淆矩阵", xlabel="预测类别", ylabel="真实类别", xticks=range(10), yticks=range(10))
    figure.colorbar(image, ax=axis, shrink=.82, label="样本数")
    save_figure(figure, output / "confusion_shift.png")


def draw_class_f1(summary, output):
    figure, axis = plt.subplots(figsize=(11.6, 4.8), layout="constrained")
    for name in MODELS:
        values = [item["f1"]*100 for item in summary["models"][name]["clean"]["per_class"]]
        axis.plot(range(10), values, marker="o", markersize=5, color=COLORS[name], label=NAMES[name])
    axis.set(title="逐类 F1  ·  同时考虑 precision 与 recall", xlabel="数字类别", ylabel="F1 / %", xticks=range(10))
    axis.grid(alpha=.22)
    all_f1 = [item["f1"]*100 for name in MODELS for item in summary["models"][name]["clean"]["per_class"]]
    axis.set_ylim(max(0, np.floor(min(all_f1)-1)), 100.15)
    axis.legend(loc="lower center", frameon=False, ncol=3)
    save_figure(figure, output / "class_f1.png")


def draw_robustness(summary, output):
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), layout="constrained")
    x = np.arange(5)
    width = .24
    for i, name in enumerate(MODELS):
        model = summary["models"][name]
        scores = [model["clean"]["accuracy"]] + [model["robustness"][direction]["accuracy"] for direction in DIRECTIONS]
        bars = axes[0].bar(x+(i-1)*width, np.asarray(scores)*100, width, color=COLORS[name], label=NAMES[name])
        axes[0].bar_label(bars, labels=[f"{value*100:.1f}" for value in scores], fontsize=8, padding=2)
        drops = [(model["clean"]["accuracy"]-model["robustness"][direction]["accuracy"])*100 for direction in DIRECTIONS]
        axes[1].plot(range(4), drops, marker="o", color=COLORS[name], label=NAMES[name])
    axes[0].set(title="原图与固定平移测试准确率", ylabel="准确率 / %", xticks=x,
                xticklabels=["原图", "上移", "下移", "左移", "右移"], ylim=(0, 108))
    axes[1].set(title="相对原图的准确率下降", ylabel="下降 / 百分点", xticks=range(4), xticklabels=["上移", "下移", "左移", "右移"])
    axes[1].axhline(0, color="#888888", linewidth=.8)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("四方向各平移 2 像素  ·  零填充  ·  测试方案在训练前固定", fontsize=15)
    save_figure(figure, output / "translation_robustness.png")


def draw_errors(summary, arrays, output):
    figure, axes = plt.subplots(3, 5, figsize=(11.2, 7.9), layout="constrained")
    for row, name in enumerate(MODELS):
        errors = summary["models"][name]["top_errors"][:5]
        for col, axis in enumerate(axes[row]):
            axis.axis("off")
            if col >= len(errors):
                continue
            error = errors[col]
            axis.imshow(arrays["test_images"][error["index"]], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axis.set_title(f"真实 {error['true']} → 预测 {error['pred']}\n置信度 {error['confidence']*100:.1f}%  ·  #{error['index']}", fontsize=10)
        axes[row, 0].text(-.14, .5, NAMES[name], rotation=90, va="center", ha="right", transform=axes[row, 0].transAxes, fontsize=12)
    figure.suptitle("高置信度误判画廊  ·  各组按错误预测置信度降序取前 5 张", fontsize=15)
    save_figure(figure, output / "high_confidence_errors.png")


def draw_features(summary, arrays, output):
    name = "shift"
    example = summary["models"][name]["feature_sample"]
    for suffix, layers, columns, rows, size in [
        ("early", ("conv1", "pool1"), 6, 2, (12.0, 4.3)),
        ("deep", ("conv2", "pool2"), 8, 4, (12.5, 7.2)),
    ]:
        figure, axes = plt.subplots(rows, columns, figsize=size, layout="constrained")
        axes = np.asarray(axes)
        for layer_index, layer in enumerate(layers):
            values = arrays[f"{name}_{layer}"]
            low, high = float(values.min()), float(values.max())
            for channel, values_one in enumerate(values):
                row = layer_index*(rows//2)+channel//columns
                col = channel % columns
                axis = axes[row, col]
                axis.imshow(values_one, cmap="viridis", vmin=low, vmax=high, interpolation="nearest")
                axis.set_title(f"{layer}  通道 {channel+1}", fontsize=9)
                axis.axis("off")
        figure.suptitle(f"C 组中间激活  ·  测试索引 0  ·  真值 {example['true']} / 预测 {example['pred']}\n每层内部共用色标  ·  各层独立缩放", fontsize=13)
        save_figure(figure, output / f"feature_maps_{suffix}.png")


def translate_numpy(image, dx, dy):
    result = np.zeros_like(image)
    height, width = image.shape
    y0, y1 = max(dy, 0), min(height+dy, height)
    x0, x1 = max(dx, 0), min(width+dx, width)
    result[y0:y1, x0:x1] = image[y0-dy:y1-dy, x0-dx:x1-dx]
    return result


def draw_input_translations(arrays, predictions, output):
    figure, axes = plt.subplots(3, 5, figsize=(10.5, 6.5), layout="constrained")
    transforms = [("原图", None, 0, 0), ("上移", "up", 0, -2), ("下移", "down", 0, 2),
                  ("左移", "left", -2, 0), ("右移", "right", 2, 0)]
    for row, index in enumerate((0, 1, 2)):
        for col, (label, direction, dx, dy) in enumerate(transforms):
            axis = axes[row, col]
            axis.imshow(translate_numpy(arrays["test_images"][index], dx, dy), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            key = "shift_pred" if direction is None else f"shift_{direction}_pred"
            pred = predictions[index][key]
            axis.set_title(f"{label}  ·  预测 {pred}\n#{index}  真值 {arrays['test_labels'][index]}", fontsize=10)
            axis.axis("off")
    figure.suptitle("固定测试索引 0、1、2 的平移示意  ·  标注使用 C 组实际预测", fontsize=14)
    save_figure(figure, output / "input_translations.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    result_dir = args.results
    output = result_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    arrays = np.load(result_dir / "analysis.npz")
    histories = read_histories(result_dir / "histories.csv")
    with (result_dir / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["SimSun", "Songti SC", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11, "axes.titlepad": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    draw_learning_curves(histories, summary, output)
    draw_confusions(summary, output)
    draw_class_f1(summary, output)
    draw_robustness(summary, output)
    draw_errors(summary, arrays, output)
    draw_features(summary, arrays, output)
    draw_input_translations(arrays, predictions, output)
    print(f"Saved {len(list(output.glob('*.png')))} figures in {output}")


if __name__ == "__main__":
    main()
