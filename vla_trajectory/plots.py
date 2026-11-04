"""Optional Matplotlib XYZ and action-time figures, with no gap interpolation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def create_episode_figure(analysis):
    """Create a 3-D XYZ plot plus X/Y/Z over Action Steps; caller closes figure.

    Each line is one observed transition. Event markers use the transition's
    next observation, consistent with post-action annotation frame semantics.
    Coordinates retain native simulator units; physical units are not inferred.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as exc:
        raise ImportError("Plots require the optional analysis dependency: matplotlib") from exc
    installed = {item.name for item in font_manager.fontManager.ttflist}
    candidates = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS") if name in installed]
    with plt.rc_context({"font.family": candidates + ["DejaVu Sans"], "axes.unicode_minus": False}):
        fig = plt.figure(figsize=(14, 9), layout="constrained")
        grid = fig.add_gridspec(4, 2, width_ratios=(1.15, 1), height_ratios=(1, 1, 1, .65))
        spatial = fig.add_subplot(grid[:3, 0], projection="3d")
        temporal = [fig.add_subplot(grid[axis, 1]) for axis in range(3)]
        event_list = fig.add_subplot(grid[3, :])
        event_list.set_axis_off()
        fig.get_layout_engine().set(w_pad=.12, h_pad=.10, wspace=.12)
        all_points = []
        indexed = {edge["frame"]: edge for edge in analysis["trajectory"]}
        for edge in analysis["trajectory"]:
            before, after = edge["obs_xyz"], edge["next_xyz"]
            for point in (before, after):
                if point is not None:
                    all_points.append(point)
            if before is None or after is None:
                continue
            points = np.asarray([before, after])
            spatial.plot(*points.T, color="#336699", alpha=0.8, linewidth=1.2)
            step = edge["action_step"]
            if step is not None:
                for axis, chart in enumerate(temporal):
                    chart.plot([step - 1, step], points[:, axis], color="#336699", linewidth=1.2)
        labels_seen, captions, descriptions = set(), defaultdict(list), []
        for number, attempt in enumerate(analysis["subtask_events"], 1):
            for kind, marker, color in (("grasp", "^", "#c46c12"), ("place", "s", "#34814d")):
                edge = indexed.get(attempt[kind + "_frame"])
                if edge is None or edge["next_xyz"] is None:
                    continue
                point, step = edge["next_xyz"], edge["action_step"]
                label = kind if kind not in labels_seen else None
                spatial.scatter(*point, marker=marker, color=color, s=55, label=label)
                labels_seen.add(kind)
                code = kind[0].upper() + str(number)
                captions[tuple(point)].append(code)
                descriptions.append(f"{code}: {attempt['attempt_id']} | {attempt['object'] or 'unknown'} | {kind} | Step {step if step is not None else 'unknown'}")
                if step is not None:
                    for axis, chart in enumerate(temporal):
                        chart.scatter(step, point[axis], marker=marker, color=color, s=24, zorder=5)
        for point, codes in captions.items():
            spatial.text(*point, "/".join(codes), fontsize=8)
        shown = descriptions[:16]
        rows = max((len(shown) + 1) // 2, 1)
        for index, description in enumerate(shown):
            event_list.text(.01 + .5 * (index % 2), .95 - (index // 2) / rows,
                            description, va="top", fontsize=8)
        if len(descriptions) > len(shown):
            event_list.text(.01, -.08, f"{len(descriptions) - len(shown)} additional events: see subtask_events.csv", fontsize=8)
        for axis, chart in enumerate(temporal):
            chart.set_ylabel("XYZ"[axis] + " (sim units)")
            chart.grid(alpha=0.25)
        temporal[-1].set_xlabel("Action Step (policy env.step, 1-based)")
        spatial.set(xlabel="X (sim units)", ylabel="Y (sim units)", zlabel="Z (sim units)")
        spatial.set_box_aspect((1, 1, 1))
        if all_points:
            points = np.asarray(all_points)
            low, high = points.min(axis=0), points.max(axis=0)
            center = (low + high) / 2
            radius = max(float((high - low).max()) / 2, 1e-6) * 1.08
            spatial.set_xlim(center[0] - radius, center[0] + radius)
            spatial.set_ylim(center[1] - radius, center[1] + radius)
            spatial.set_zlim(center[2] - radius, center[2] + radius)
        if labels_seen:
            spatial.legend(loc="upper left")
        source = "SYNTHETIC DEMO | " if analysis["synthetic"] else ""
        coverage = "unknown" if analysis["eef_coverage"] is None else f"{analysis['eef_coverage']:.1%}"
        fig.suptitle(f"{source}{analysis['episode_id']} | {analysis['task_id']} | {analysis['stage']}\n"
                     f"EEF coverage: {coverage} | gaps are not connected")
    return fig


def export_episode_plots(analysis, output):
    """Save PNG and SVG and release the Matplotlib figure."""
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig = create_episode_figure(analysis)
    paths = [output / (analysis["episode_id"] + "_eef" + suffix) for suffix in (".png", ".svg")]
    try:
        for path in paths:
            fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return paths
