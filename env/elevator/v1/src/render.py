import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches


def render_building(
    num_floors,
    num_elevators,
    floor_states,
    people_on_each_floor,
    elevators,
    current_step,
):
    """Render building state to an (H, W, 3) uint8 RGB numpy array."""
    col_w = 2.8
    row_h = 0.7
    num_cols = 1 + num_elevators  # floor info + one per elevator
    total_w = num_cols * col_w
    total_h = num_floors * row_h

    fig = Figure(figsize=(total_w + 0.6, total_h + 0.8), facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.90])
    ax.set_xlim(-0.05, total_w + 0.05)
    ax.set_ylim(-0.05, total_h + 0.05)
    ax.axis("off")

    for floor in range(num_floors):
        # row 0 at bottom of image = floor 0, so y maps directly
        y = floor * row_h
        up_count = int(people_on_each_floor[floor][0])
        down_count = int(people_on_each_floor[floor][1])
        up_on = bool(floor_states[floor][0])
        down_on = bool(floor_states[floor][1])

        # floor info cell
        ax.add_patch(
            patches.Rectangle(
                (0, y),
                col_w,
                row_h,
                facecolor="#f0f0ea",
                edgecolor="#444",
                lw=1,
            )
        )
        up_str = f"▲{up_count}" if up_on else f"  {up_count}"
        dn_str = f"▼{down_count}" if down_on else f"  {down_count}"
        ax.text(
            col_w / 2,
            y + row_h * 0.62,
            f"Floor {floor + 1}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            family="monospace",
        )
        ax.text(
            col_w / 2,
            y + row_h * 0.28,
            f"{up_str}  {dn_str}",
            ha="center",
            va="center",
            fontsize=7,
            family="monospace",
            color="#555",
        )

        # elevator shaft cells
        for e_idx, elev in enumerate(elevators):
            ex = (1 + e_idx) * col_w
            is_here = elev.current_floor == floor
            bg = "#4A90D9" if is_here else "#e0e0e0"
            ax.add_patch(
                patches.Rectangle(
                    (ex, y),
                    col_w,
                    row_h,
                    facecolor=bg,
                    edgecolor="#444",
                    lw=1,
                )
            )
            if is_here:
                txt_c = "white"
                ax.text(
                    ex + col_w / 2,
                    y + row_h * 0.62,
                    f"Elev {e_idx + 1}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=txt_c,
                    family="monospace",
                )
                ax.text(
                    ex + col_w / 2,
                    y + row_h * 0.28,
                    f"carrying: {len(elev.carrying_people)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_c,
                    family="monospace",
                )

    ax.set_title(f"Step {current_step}", fontsize=10, fontweight="bold", pad=6)

    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    # H.264 requires even dimensions for YUV420 chroma subsampling
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]
