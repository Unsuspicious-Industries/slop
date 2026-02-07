"""Unified heatmap renderer for scalar fields.

Provides consistent visualization for scalar fields phi(r).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional, Tuple
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    Axes3D = None

from ...physics.types import ScalarField


class HeatmapRenderer:
    """Unified renderer for scalar field heatmaps in 2D and 3D.
    
    Supports:
    - 2D heatmaps (imshow/contourf)
    - 3D isosurfaces
    - 3D slice views
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'RdBu_r',
        style: str = 'filled',
        **kwargs
    ):
        """Initialize heatmap renderer.
        
        Args:
            figsize: Figure size (width, height)
            cmap: Colormap name
            style: 'filled' (imshow), 'contour', or 'both'
            **kwargs: Additional matplotlib arguments
        """
        self.figsize = figsize
        self.cmap = cmap
        self.style = style
        self.kwargs = kwargs
    
    def render_2d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        phi: ScalarField,
        title: str = "Scalar Field phi(r)",
        save_path: Optional[Path] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        levels: int = 20
    ) -> Figure:
        """Render 2D scalar field as heatmap.
        
        Args:
            X, Y: Meshgrid coordinates
            phi: Scalar field, shape (ny, nx)
            title: Plot title
            save_path: Optional path to save figure
            vmin, vmax: Value limits for colormap
            levels: Number of contour levels
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if vmin is None:
            vmin = phi.min()
        if vmax is None:
            vmax = phi.max()
        
        if self.style in ['filled', 'both']:
            im = ax.imshow(
                phi,
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                origin='lower',
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
                **self.kwargs
            )
            plt.colorbar(im, ax=ax, label='phi')
        
        if self.style in ['contour', 'both']:
            contours = ax.contour(
                X, Y, phi,
                levels=levels,
                colors='black' if self.style == 'both' else None,
                cmap=None if self.style == 'both' else self.cmap,
                linewidths=0.5 if self.style == 'both' else 1.0
            )
            if self.style == 'contour':
                plt.colorbar(contours, ax=ax, label='phi')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        
        return fig
    
    def render_3d_slices(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        phi: ScalarField,
        title: str = "3D Scalar Field phi(r) - Slices",
        save_path: Optional[Path] = None,
        slice_positions: Optional[Tuple[float, float, float]] = None
    ) -> Figure:
        """Render 3D scalar field as orthogonal slices.
        
        Args:
            X, Y, Z: Meshgrid coordinates
            phi: Scalar field, shape (nz, ny, nx)
            title: Plot title
            save_path: Optional path to save figure
            slice_positions: (x, y, z) positions for slices (fractions 0-1)
            
        Returns:
            Figure object
        """
        if Axes3D is None:
            raise ImportError("mpl_toolkits.mplot3d required for 3D rendering")
        
        if slice_positions is None:
            slice_positions = (0.5, 0.5, 0.5)
        
        nz, ny, nx = phi.shape
        ix = int(slice_positions[0] * nx)
        iy = int(slice_positions[1] * ny)
        iz = int(slice_positions[2] * nz)
        
        fig = plt.figure(figsize=(15, 5))
        
        vmin, vmax = phi.min(), phi.max()
        
        # YZ plane (constant x)
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(
            phi[:, :, ix].T,
            extent=[Z.min(), Z.max(), Y.min(), Y.max()],
            origin='lower',
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        ax1.set_xlabel('z')
        ax1.set_ylabel('y')
        ax1.set_title(f'YZ plane (x={X[0, 0, ix]:.2f})')
        plt.colorbar(im1, ax=ax1)
        
        # XZ plane (constant y)
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(
            phi[:, iy, :].T,
            extent=[Z.min(), Z.max(), X.min(), X.max()],
            origin='lower',
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        ax2.set_xlabel('z')
        ax2.set_ylabel('x')
        ax2.set_title(f'XZ plane (y={Y[0, iy, 0]:.2f})')
        plt.colorbar(im2, ax=ax2)
        
        # XY plane (constant z)
        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(
            phi[iz, :, :],
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower',
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title(f'XY plane (z={Z[iz, 0, 0]:.2f})')
        plt.colorbar(im3, ax=ax3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D slices to {save_path}")
        
        return fig
    
    def render_divergence(
        self,
        coords: Tuple,
        div_V: ScalarField,
        title: str = "Divergence div(V)",
        save_path: Optional[Path] = None,
        symmetric: bool = True
    ) -> Figure:
        """Render divergence field with appropriate colormap.
        
        Uses diverging colormap centered at zero.
        
        Args:
            coords: Grid coordinates
            div_V: Divergence field
            title: Plot title
            save_path: Optional path to save
            symmetric: Whether to make colormap symmetric around zero
            
        Returns:
            Figure object
        """
        # Use diverging colormap for divergence
        original_cmap = self.cmap
        self.cmap = 'RdBu_r'  # Red=positive (sources), Blue=negative (sinks)
        
        if symmetric:
            vmax = np.abs(div_V).max()
            vmin = -vmax
        else:
            vmin = div_V.min()
            vmax = div_V.max()
        
        if len(coords) == 2:
            fig = self.render_2d(*coords, div_V, title, save_path, vmin, vmax)
        else:
            fig = self.render_3d_slices(*coords, div_V, title, save_path)
        
        self.cmap = original_cmap
        return fig
