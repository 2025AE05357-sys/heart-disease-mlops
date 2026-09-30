"""House plotting style: the plan requires 150 dpi figures and no text below 11 pt."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from PIL import Image  # noqa: E402

from heartrisk import viz  # noqa: E402


def test_save_fig_writes_a_150_dpi_png_named_after_the_figure(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    path = viz.save_fig(fig, "eda_example", tmp_path / "figures")

    assert path == tmp_path / "figures" / "eda_example.png"
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert [round(d) for d in image.info["dpi"]] == [150, 150]


def test_every_text_element_is_at_least_11_pt_after_apply_style():
    viz.apply_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="series")
    ax.set_title("Title")
    ax.set_xlabel("x")
    ax.legend()
    fig.canvas.draw()

    texts = [ax.title, ax.xaxis.label, *ax.get_xticklabels(), *ax.get_legend().get_texts()]
    sizes = [FontProperties(size=t.get_fontsize()).get_size_in_points() for t in texts]
    assert min(sizes) >= 11
    plt.close(fig)
