"""Trajectory renderer for visualization of trajectory curves.

Visualizes paths r(t) through embedding space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import List, Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D

from ...physics.types import Trajectory


class TrajectoryRenderer:
    """Unified renderer for trajectory visualization in 2D and 3D."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        alpha: float = 0.7,
        **kwargs
    ):
        """Initialize trajectory renderer.
        
        Args:
            figsize: Figure size
            cmap: Colormap for coloring by time/index
            alpha: Transparency
            **kwargs: Additional matplotlib arguments
        """
        self.figsize = figsize
        self.cmap = cmap
        self.alpha = alpha
        self.kwargs = kwargs
    
    def render_2d(
        self,
        trajectories: List[Trajectory],
        title: str = "Trajectories r(t)",
        save_path: Optional[Path] = None,
        color_by_time: bool = True,
        show_start: bool = True,
        show_end: bool = True
    ) -> Figure:
        """Render 2D trajectories.
        
        Args:
            trajectories: List of trajectories, each shape (n_steps, 2)
            title: Plot title
            save_path: Optional path to save
            color_by_time: Color trajectories by time progression
            show_start: Mark start points
            show_end: Mark end points
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        cmap = plt.get_cmap(self.cmap)
        
        for i, r_t in enumerate(trajectories):
            if color_by_time:
                # Color by time within trajectory
                t = np.linspace(0, 1, len(r_t))
                for j in range(len(r_t) - 1):
                    ax.plot(
                        r_t[j:j+2, 0], r_t[j:j+2, 1],
                        color=cmap(t[j]),
                        alpha=self.alpha,
                        **self.kwargs
                    )
            else:
                # Single color per trajectory
                color = cmap(i / len(trajectories))
                ax.plot(
                    r_t[:, 0], r_t[:, 1],
                    color=color,
                    alpha=self.alpha,
                    **self.kwargs
                )
            
            # Mark start and end
            if show_start:
                ax.scatter(r_t[0, 0], r_t[0, 1], c='green', s=50, marker='o', zorder=5, label='Start' if i == 0 else '')
            if show_end:
                ax.scatter(r_t[-1, 0], r_t[-1, 1], c='red', s=50, marker='x', zorder=5, label='End' if i == 0 else '')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        if show_start or show_end:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved trajectories to {save_path}")
        
        return fig
    
    def render_3d(
        self,
        trajectories: List[Trajectory],
        title: str = "3D Trajectories r(t)",
        save_path: Optional[Path] = None,
        color_by_time: bool = True,
        show_start: bool = True,
        show_end: bool = True
    ) -> Figure:
        """Render 3D trajectories.
        
        Args:
            trajectories: List of trajectories, each shape (n_steps, 3)
            title: Plot title
            save_path: Optional path to save
            color_by_time: Color trajectories by time progression
            show_start: Mark start points
            show_end: Mark end points
            
        Returns:
            Figure object
        """
        if Axes3D is None:
            raise ImportError("mpl_toolkits.mplot3d required for 3D rendering")
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        cmap = plt.get_cmap(self.cmap)
        
        for i, r_t in enumerate(trajectories):
            if color_by_time:
                # Color by time within trajectory
                t = np.linspace(0, 1, len(r_t))
                for j in range(len(r_t) - 1):
                    ax.plot(
                        r_t[j:j+2, 0], r_t[j:j+2, 1], r_t[j:j+2, 2],
                        color=cmap(t[j]),
                        alpha=self.alpha,
                        **self.kwargs
                    )
            else:
                # Single color per trajectory
                color = cmap(i / len(trajectories))
                ax.plot(
                    r_t[:, 0], r_t[:, 1], r_t[:, 2],
                    color=color,
                    alpha=self.alpha,
                    **self.kwargs
                )
            
            # Mark start and end
            if show_start:
                ax.scatter(r_t[0, 0], r_t[0, 1], r_t[0, 2], c='green', s=50, marker='o', label='Start' if i == 0 else '')
            if show_end:
                ax.scatter(r_t[-1, 0], r_t[-1, 1], r_t[-1, 2], c='red', s=50, marker='x', label='End' if i == 0 else '')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        
        if show_start or show_end:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D trajectories to {save_path}")
        
        return fig
    
    def render(
        self,
        trajectories: List[Trajectory],
        **kwargs
    ) -> Figure:
        """Auto-detect dimensionality and render.
        
        Args:
            trajectories: List of trajectories
            **kwargs: Additional arguments
            
        Returns:
            Figure object
        """
        dim = trajectories[0].shape[1]
        
        if dim == 2:
            return self.render_2d(trajectories, **kwargs)
        elif dim == 3:
            return self.render_3d(trajectories, **kwargs)
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
