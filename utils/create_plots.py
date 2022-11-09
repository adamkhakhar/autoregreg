from matplotlib import pyplot as plt
from typing import List
import os


def single_y_plot(
    x: List,
    y: List,
    save_title: str,
    save_path=os.path.dirname(os.path.realpath(__file__)),
    labels=None,
    colors=None,
    line_style=None,
    x_axis_label=None,
    y_axis_label=None,
    x_axis_exp=False,
    y_axis_exp=False,
    title=None,
    legend=False,
):
    assert len(x) == len(y)
    _, ax = plt.subplots()
    ax.spines["top"].set_color((0.8, 0.8, 0.8))
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title, fontsize=20)

    for i in range(len(x)):
        kwargs = {}
        if labels is not None:
            kwargs["label"] = labels[i]
        if colors is not None:
            kwargs["color"] = colors[i]
        if line_style is not None:
            kwargs["linestyle"] = line_style[i]
        plt.plot(x[i], y[i], **kwargs)

    if legend:
        plt.legend()
    if y_axis_exp:
        plt.yscale("symlog")
    if x_axis_exp:
        plt.xscale("symlog")

    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style("italic")
    xlab.set_size(10)
    ylab.set_style("italic")
    ylab.set_size(10)

    # tweak the title
    ttl = ax.title
    ttl.set_weight("bold")

    plt.tight_layout()
    plt.grid(linestyle="--", alpha=0.25)
    plt.savefig(f"{save_path}/{save_title}", bbox_inches="tight")
