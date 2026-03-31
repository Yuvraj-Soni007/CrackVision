"""
Visualization Module
=====================
Comprehensive plotting and annotation utilities for all pipeline stages.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Dict, Tuple

from ..utils import load_config, ensure_dir, project_root, logger


class Visualizer:
    """Rich visualization of crack detection results."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        vis = cfg["visualization"]
        self.fmt = vis["save_format"]
        self.dpi = vis["dpi"]
        self.cmap = vis["colormap"]
        self.alpha = vis["overlay_alpha"]
        self.figsize = tuple(vis["figure_size"])
        self.out = str(project_root() / cfg["paths"]["results_dir"])
        ensure_dir(self.out)

    # ------------------------------------------------------------------
    # Overlay crack mask on original image
    # ------------------------------------------------------------------
    def overlay_mask(self, img: np.ndarray, mask: np.ndarray,
                     color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Overlay a binary mask on the image in a given colour."""
        bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay = bgr.copy()
        overlay[mask > 0] = color
        blended = cv2.addWeighted(bgr, 1 - self.alpha, overlay, self.alpha, 0)
        return blended

    # ------------------------------------------------------------------
    # Draw bounding boxes for crack regions
    # ------------------------------------------------------------------
    @staticmethod
    def draw_regions(img: np.ndarray, regions: List[Dict],
                     severity_colors: Optional[Dict[str, Tuple]] = None) -> np.ndarray:
        """Draw annotated bounding boxes on the image."""
        bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = bgr.copy()

        default_colors = {
            "none": (180, 180, 180),
            "minor": (0, 255, 0),
            "moderate": (0, 255, 255),
            "severe": (0, 128, 255),
            "critical": (0, 0, 255),
        }
        colors = severity_colors or default_colors

        for r in regions:
            # The structure might be a region dict directly, or a severity dict containing a "region" key
            reg_dict = r.get("region", r)
            x, y, w, h = reg_dict.get("x", 0), reg_dict.get("y", 0), reg_dict.get("w", 0), reg_dict.get("h", 0)
            sev = r.get("severity_name", "moderate")
            color = colors.get(sev, (0, 255, 255))
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

            label = f"{sev.upper()} [{r.get('composite_score', 0):.2f}]"
            cv2.putText(out, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Draw Hough lines
    # ------------------------------------------------------------------
    @staticmethod
    def draw_hough_lines(img: np.ndarray, lines: List,
                         color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw detected Hough lines on the image."""
        bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = bgr.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(out, (x1, y1), (x2, y2), color, 2)
        return out

    # ------------------------------------------------------------------
    # Multi-panel result visualization
    # ------------------------------------------------------------------
    def plot_pipeline_results(self, stages: Dict[str, np.ndarray],
                              title: str = "CrackVision Pipeline Results",
                              save_name: Optional[str] = None):
        """Plot multiple pipeline stages in a grid."""
        n = len(stages)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, (name, img) in zip(axes, stages.items()):
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.axis("off")

        # Hide unused axes
        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_name:
            path = f"{self.out}/{save_name}.{self.fmt}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved → {path}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Severity report visualisation
    # ------------------------------------------------------------------
    def plot_severity_report(self, severity_results: List[Dict],
                             save_name: Optional[str] = None):
        """Bar chart of severity scores for all detected regions."""
        if not severity_results:
            logger.warning("No severity results to plot")
            return

        names = [f"R{i + 1}" for i in range(len(severity_results))]
        scores = [r["composite_score"] for r in severity_results]
        levels = [r["severity_name"] for r in severity_results]

        color_map = {
            "none": "#9E9E9E", "minor": "#4CAF50",
            "moderate": "#FF9800", "severe": "#FF5722", "critical": "#D32F2F",
        }
        colors = [color_map.get(l, "#999") for l in levels]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, scores, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Composite Severity Score", fontsize=12)
        ax.set_xlabel("Crack Region", fontsize=12)
        ax.set_title("Crack Severity Analysis", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.5, label="Minor threshold")
        ax.axhline(y=0.55, color="orange", linestyle="--", alpha=0.5, label="Moderate threshold")
        ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="Severe threshold")
        ax.legend(fontsize=9)

        # Annotate bars
        for bar, score, level in zip(bars, scores, levels):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{score:.2f}\n({level})", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        if save_name:
            path = f"{self.out}/{save_name}.{self.fmt}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Saved → {path}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Depth map visualisation
    # ------------------------------------------------------------------
    def plot_depth_map(self, depth: np.ndarray, title: str = "Depth Map",
                       save_name: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(depth, cmap=self.cmap)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_name:
            path = f"{self.out}/{save_name}.{self.fmt}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Normal map visualisation
    # ------------------------------------------------------------------
    def plot_normal_map(self, normals_rgb: np.ndarray, save_name: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(normals_rgb)
        ax.set_title("Surface Normal Map", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        if save_name:
            path = f"{self.out}/{save_name}.{self.fmt}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Save single image
    # ------------------------------------------------------------------
    def save_image(self, img: np.ndarray, name: str):
        """Save an OpenCV image to results directory."""
        path = f"{self.out}/{name}.{self.fmt}"
        cv2.imwrite(path, img)
        logger.info(f"Image saved → {path}")
