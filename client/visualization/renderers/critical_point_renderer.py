"""Critical point renderer for attractors, repellers, and saddles.

Visualizes critical points of vector fields based on divergence analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional, Tuple, Dict
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    Axes3D = None

from ...physics.types import ScalarField


class CriticalPointRenderer:
    """Unified renderer for critical points in 2D and 3D.
    
    Renders:
    - Attractors (sinks): div(V) < 0
    - Repellers (sources): div(V) > 0
    - Saddle points: div(V) ~= 0
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        attractor_color: str = 'blue',
        repeller_color: str = 'red',
        saddle_color: str = 'gray',
        **kwargs
    ):
        """Initialize critical point renderer.
        
        Args:
            figsize: Figure size
            attractor_color: Color for attractors
            repeller_color: Color for repellers
            saddle_color: Color for saddles
            **kwargs: Additional matplotlib arguments
        """
        self.figsize = figsize
        self.attractor_color = attractor_color
        self.repeller_color = repeller_color
        self.saddle_color = saddle_color
        self.kwargs = kwargs
    
    def render_2d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        critical_points: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "Critical Points",
        save_path: Optional[Path] = None,
        show_strength: bool = True,
        background_field: Optional[ScalarField] = None
    ) -> Figure:
        """Render 2D critical points.
        
        Args:
            X, Y: Meshgrid coordinates
            critical_points: Dict with keys 'attractors', 'repellers', 'saddles'
                            Values are (positions, strengths)
            title: Plot title
            save_path: Optional path to save
            show_strength: Size markers by strength
            background_field: Optional scalar field to show as background
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Show background field if provided
        if background_field is not None:
            im = ax.imshow(
                background_field,
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                origin='lower',
                cmap='RdBu_r',
                alpha=0.3,
                aspect='auto'
            )
            plt.colorbar(im, ax=ax, label='div(V)')
        
        # Plot attractors (sinks)
        if 'attractors' in critical_points:
            r_attractors, strengths = critical_points['attractors']
            if len(r_attractors) > 0:
                # Convert grid indices to coordinates
                y_idx, x_idx = r_attractors[:, 0], r_attractors[:, 1]
                x_coords = X[0, x_idx]
                y_coords = Y[y_idx, 0]
                
                if show_strength and strengths is not None:
                    sizes = 100 * np.abs(strengths) / np.abs(strengths).max()
                else:
                    sizes = 100
                
                ax.scatter(
                    x_coords, y_coords,
                    c=self.attractor_color,
                    s=sizes,
                    marker='o',
                    label=f'Attractors ({len(r_attractors)})',
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.8,
                    **self.kwargs
                )
        
        # Plot repellers (sources)
        if 'repellers' in critical_points:
            r_repellers, strengths = critical_points['repellers']
            if len(r_repellers) > 0:
                y_idx, x_idx = r_repellers[:, 0], r_repellers[:, 1]
                x_coords = X[0, x_idx]
                y_coords = Y[y_idx, 0]
                
                if show_strength and strengths is not None:
                    sizes = 100 * np.abs(strengths) / np.abs(strengths).max()
                else:
                    sizes = 100
                
                ax.scatter(
                    x_coords, y_coords,
                    c=self.repeller_color,
                    s=sizes,
                    marker='^',
                    label=f'Repellers ({len(r_repellers)})',
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.8,
                    **self.kwargs
                )
        
        # Plot saddles
        if 'saddles' in critical_points:
            r_saddles, _ = critical_points['saddles']
            if len(r_saddles) > 0:
                y_idx, x_idx = r_saddles[:, 0], r_saddles[:, 1]
                x_coords = X[0, x_idx]
                y_coords = Y[y_idx, 0]
                
                ax.scatter(
                    x_coords, y_coords,
                    c=self.saddle_color,
                    s=50,
                    marker='s',
                    label=f'Saddles ({len(r_saddles)})',
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.6,
                    **self.kwargs
                )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved critical points to {save_path}")
        
        return fig
    
    def render_3d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        critical_points: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "3D Critical Points",
        save_path: Optional[Path] = None,
        show_strength: bool = True
    ) -> Figure:
        """Render 3D critical points.
        
        Args:
            X, Y, Z: Meshgrid coordinates
            critical_points: Dict with critical point data
            title: Plot title
            save_path: Optional path to save
            show_strength: Size markers by strength
            
        Returns:
            Figure object
        """
        if Axes3D is None:
            raise ImportError("mpl_toolkits.mplot3d required for 3D rendering")
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot attractors
        if 'attractors' in critical_points:
            r_attractors, strengths = critical_points['attractors']
            if len(r_attractors) > 0:
                z_idx, y_idx, x_idx = r_attractors[:, 0], r_attractors[:, 1], r_attractors[:, 2]
                x_coords = X[z_idx, y_idx, x_idx]
                y_coords = Y[z_idx, y_idx, x_idx]
                z_coords = Z[z_idx, y_idx, x_idx]
                
                if show_strength and strengths is not None:
                    sizes = 100 * np.abs(strengths) / np.abs(strengths).max()
                else:
                    sizes = 100
                
                ax.scatter(
                    x_coords, y_coords, z_coords,
                    c=self.attractor_color,
                    s=sizes,
                    marker='o',
                    label=f'Attractors ({len(r_attractors)})',
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.8
                )
        
        # Plot repellers
        if 'repellers' in critical_points:
            r_repellers, strengths = critical_points['repellers']
            if len(r_repellers) > 0:
                z_idx, y_idx, x_idx = r_repellers[:, 0], r_repellers[:, 1], r_repellers[:, 2]
                x_coords = X[z_idx, y_idx, x_idx]
                y_coords = Y[z_idx, y_idx, x_idx]
                z_coords = Z[z_idx, y_idx, x_idx]
                
                if show_strength and strengths is not None:
                    sizes = 100 * np.abs(strengths) / np.abs(strengths).max()
                else:
                    sizes = 100
                
                ax.scatter(
                    x_coords, y_coords, z_coords,
                    c=self.repeller_color,
                    s=sizes,
                    marker='^',
                    label=f'Repellers ({len(r_repellers)})',
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.8
                )
        
        # Plot saddles
        if 'saddles' in critical_points:
            r_saddles, _ = critical_points['saddles']
            if len(r_saddles) > 0:
                z_idx, y_idx, x_idx = r_saddles[:, 0], r_saddles[:, 1], r_saddles[:, 2]
                x_coords = X[z_idx, y_idx, x_idx]
                y_coords = Y[z_idx, y_idx, x_idx]
                z_coords = Z[z_idx, y_idx, x_idx]
                
                ax.scatter(
                    x_coords, y_coords, z_coords,
                    c=self.saddle_color,
                    s=50,
                    marker='s',
                    label=f'Saddles ({len(r_saddles)})',
                    edgecolors='black',
                    linewidths=1,
                    alpha=0.6
                )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D critical points to {save_path}")
        
        return fig
