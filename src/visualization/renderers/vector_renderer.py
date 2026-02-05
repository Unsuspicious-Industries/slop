"""Unified vector field renderer with 2D and 3D support.

Provides consistent visualization for vector fields V(r).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import Optional, Tuple, Union
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    Axes3D = None

from ...physics.types import VectorField, GridCoords2D, GridCoords3D


class VectorRenderer:
    """Unified renderer for vector fields in 2D and 3D.
    
    Supports:
    - 2D quiver plots
    - 2D streamlines
    - 3D quiver plots
    - 3D streamlines (stream tubes)
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 8),
        style: str = 'quiver',
        density: float = 1.0,
        **kwargs
    ):
        """Initialize vector renderer.
        
        Args:
            figsize: Figure size (width, height)
            style: 'quiver' or 'stream'
            density: Density for quiver/streamplot (1.0 = default)
            **kwargs: Additional matplotlib arguments
        """
        self.figsize = figsize
        self.style = style
        self.density = density
        self.kwargs = kwargs
    
    def render_2d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        V: VectorField,
        title: str = "Vector Field V(r)",
        save_path: Optional[Path] = None,
        show_magnitude: bool = True
    ) -> Figure:
        """Render 2D vector field.
        
        Args:
            X, Y: Meshgrid coordinates
            V: Vector field, shape (ny, nx, 2)
            title: Plot title
            save_path: Optional path to save figure
            show_magnitude: Whether to color by magnitude
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract components
        Vx = V[..., 0]
        Vy = V[..., 1]
        
        if self.style == 'quiver':
            # Compute magnitude for coloring
            magnitude = np.sqrt(Vx**2 + Vy**2)
            
            # Subsample for clarity
            step = max(1, int(1.0 / self.density))
            
            if show_magnitude:
                im = ax.quiver(
                    X[::step, ::step], Y[::step, ::step],
                    Vx[::step, ::step], Vy[::step, ::step],
                    magnitude[::step, ::step],
                    cmap='viridis',
                    **self.kwargs
                )
                plt.colorbar(im, ax=ax, label='|V|')
            else:
                ax.quiver(
                    X[::step, ::step], Y[::step, ::step],
                    Vx[::step, ::step], Vy[::step, ::step],
                    **self.kwargs
                )
        
        elif self.style == 'stream':
            # Streamline plot
            magnitude = np.sqrt(Vx**2 + Vy**2)
            
            if show_magnitude:
                strm = ax.streamplot(
                    X[0, :], Y[:, 0],
                    Vx, Vy,
                    color=magnitude,
                    density=self.density,
                    cmap='viridis',
                    **self.kwargs
                )
                plt.colorbar(strm.lines, ax=ax, label='|V|')
            else:
                ax.streamplot(
                    X[0, :], Y[:, 0],
                    Vx, Vy,
                    density=self.density,
                    **self.kwargs
                )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved vector field to {save_path}")
        
        return fig
    
    def render_3d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        V: VectorField,
        title: str = "3D Vector Field V(r)",
        save_path: Optional[Path] = None,
        show_magnitude: bool = True,
        subsample: int = 2
    ) -> Figure:
        """Render 3D vector field.
        
        Args:
            X, Y, Z: Meshgrid coordinates
            V: Vector field, shape (nz, ny, nx, 3)
            title: Plot title
            save_path: Optional path to save figure
            show_magnitude: Whether to color by magnitude
            subsample: Subsampling factor for clarity
            
        Returns:
            Figure object
        """
        if Axes3D is None:
            raise ImportError("mpl_toolkits.mplot3d required for 3D rendering")
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract components
        Vx = V[..., 0]
        Vy = V[..., 1]
        Vz = V[..., 2]
        
        # Subsample for visualization
        X_sub = X[::subsample, ::subsample, ::subsample]
        Y_sub = Y[::subsample, ::subsample, ::subsample]
        Z_sub = Z[::subsample, ::subsample, ::subsample]
        Vx_sub = Vx[::subsample, ::subsample, ::subsample]
        Vy_sub = Vy[::subsample, ::subsample, ::subsample]
        Vz_sub = Vz[::subsample, ::subsample, ::subsample]
        
        if show_magnitude:
            magnitude = np.sqrt(Vx_sub**2 + Vy_sub**2 + Vz_sub**2)
            
            # Flatten for quiver3D
            x_flat = X_sub.flatten()
            y_flat = Y_sub.flatten()
            z_flat = Z_sub.flatten()
            u_flat = Vx_sub.flatten()
            v_flat = Vy_sub.flatten()
            w_flat = Vz_sub.flatten()
            mag_flat = magnitude.flatten()
            
            # Color by magnitude
            colors = plt.cm.viridis(mag_flat / mag_flat.max())
            
            ax.quiver(
                x_flat, y_flat, z_flat,
                u_flat, v_flat, w_flat,
                colors=colors,
                length=0.1,
                normalize=True,
                **self.kwargs
            )
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis')
            sm.set_array(mag_flat)
            plt.colorbar(sm, ax=ax, label='|V|', shrink=0.5)
        else:
            ax.quiver(
                X_sub, Y_sub, Z_sub,
                Vx_sub, Vy_sub, Vz_sub,
                length=0.1,
                normalize=True,
                **self.kwargs
            )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D vector field to {save_path}")
        
        return fig
    
    def render(
        self,
        coords: Union[GridCoords2D, GridCoords3D],
        V: VectorField,
        **kwargs
    ) -> Figure:
        """Auto-detect dimensionality and render.
        
        Args:
            coords: Grid coordinates (X, Y) or (X, Y, Z)
            V: Vector field
            **kwargs: Additional arguments for render_2d or render_3d
            
        Returns:
            Figure object
        """
        if len(coords) == 2:
            return self.render_2d(*coords, V, **kwargs)
        elif len(coords) == 3:
            return self.render_3d(*coords, V, **kwargs)
        else:
            raise ValueError(f"Invalid coordinate dimension: {len(coords)}")
