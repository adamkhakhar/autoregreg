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
    markers=None,
    linewidth=None,
    x_axis_label=None,
    y_axis_label=None,
    x_axis_exp=False,
    y_axis_exp=False,
    title=None,
    legend=False,
    tick_size=12,
    label_size=15,
    num_yaxis_ticks=6,
    save_every_xth_ytick=2,
):
    assert len(x) == len(y)
    _, ax = plt.subplots()
    ax.spines["top"].set_color((0.8, 0.8, 0.8))
    ax.spines["right"].set_color((0.8, 0.8, 0.8))
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
        if markers is not None:
            kwargs["marker"] = markers[i]
        if linewidth is not None:
            kwargs["linewidth"] = linewidth[i]
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
    xlab.set_size(label_size)
    ylab.set_style("italic")
    ylab.set_size(label_size)

    # tweak the title
    ttl = ax.title
    ttl.set_weight("bold")

    # tweak ticks
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    ax.set_yticks(
        [
            ax.get_yticks()[i]
            for i in range(len(ax.get_yticks()))
            if i % save_every_xth_ytick == 0
        ]
    )

    plt.tight_layout()
    plt.grid(linestyle="--", alpha=0.25)
    plt.savefig(f"{save_path}/{save_title}.png", bbox_inches="tight")
