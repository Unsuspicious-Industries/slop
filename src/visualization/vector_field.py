"""Unified vector field visualizer supporting 2D and 3D fields.

This module provides a consistent interface for visualizing vector fields,
flow fields, gradients, and other vector data in both 2D and 3D spaces.
"""
# mypy: ignore-errors

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


class VectorFieldVisualizer:
    """Unified visualizer for vector fields in 2D and 3D."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
    
    def visualize_2d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        title: str = "2D Vector Field",
        poles: Optional[np.ndarray] = None,
        trajectories: Optional[list] = None,
        magnitude_colormap: str = "viridis",
        stream_density: float = 2.0,
        quiver_mode: bool = False,
        save_path: Optional[Path] = None,
        show_divergence: bool = False,
        divergence: Optional[np.ndarray] = None,
        **kwargs
    ) -> plt.Figure:
        """Visualize 2D vector field.
        
        Args:
            X, Y: Meshgrid coordinates
            U, V: Vector field components
            title: Plot title
            poles: Attractor/repeller points to highlight, shape (n, 2)
            trajectories: List of trajectory arrays to overlay
            magnitude_colormap: Colormap for vector magnitude
            stream_density: Density of streamlines (if quiver_mode=False)
            quiver_mode: Use quiver plot instead of streamplot
            save_path: Path to save figure
            show_divergence: Overlay divergence as background color
            divergence: Pre-computed divergence field
            **kwargs: Additional plotting parameters
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate magnitude
        magnitude = np.sqrt(U**2 + V**2)
        
        # Show divergence as background if requested
        if show_divergence and divergence is not None:
            im = ax.imshow(
                divergence,
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                origin='lower',
                cmap='RdBu_r',
                alpha=0.3,
                aspect='auto'
            )
            plt.colorbar(im, ax=ax, label='Divergence')
        
        # Plot vector field
        if quiver_mode:
            # Quiver plot
            skip = kwargs.get('quiver_skip', 1)
            Q = ax.quiver(
                X[::skip, ::skip],
                Y[::skip, ::skip],
                U[::skip, ::skip],
                V[::skip, ::skip],
                magnitude[::skip, ::skip],
                cmap=magnitude_colormap,
                alpha=0.7
            )
            plt.colorbar(Q, ax=ax, label='Magnitude')
        else:
            # Streamplot
            stream = ax.streamplot(
                X, Y, U, V,
                color=magnitude,
                cmap=magnitude_colormap,
                density=stream_density,
                linewidth=kwargs.get('linewidth', 2),
                arrowsize=kwargs.get('arrowsize', 1.5)
            )
            plt.colorbar(stream.lines, ax=ax, label='Magnitude')
        
        # Overlay trajectories
        if trajectories:
            for traj in trajectories:
                if traj.shape[1] >= 2:
                    ax.plot(traj[:, 0], traj[:, 1], 
                           alpha=0.5, linewidth=1.5, 
                           color=kwargs.get('traj_color', 'green'))
        
        # Mark poles/attractors
        if poles is not None and len(poles) > 0:
            ax.scatter(
                poles[:, 0], poles[:, 1],
                s=kwargs.get('pole_size', 300),
                c=kwargs.get('pole_color', 'red'),
                marker=kwargs.get('pole_marker', 'X'),
                edgecolors='white',
                linewidths=2,
                zorder=10,
                label='Attractors/Poles'
            )
            ax.legend()
        
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def visualize_3d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        W: np.ndarray,
        title: str = "3D Vector Field",
        poles: Optional[np.ndarray] = None,
        trajectories: Optional[list] = None,
        magnitude_colormap: str = "viridis",
        arrow_length: float = 0.1,
        save_path: Optional[Path] = None,
        elevation: float = 20,
        azimuth: float = 45,
        **kwargs
    ) -> plt.Figure:
        """Visualize 3D vector field.
        
        Args:
            X, Y, Z: Meshgrid coordinates
            U, V, W: Vector field components
            title: Plot title
            poles: Attractor/repeller points to highlight, shape (n, 3)
            trajectories: List of trajectory arrays to overlay (each shape (n, 3))
            magnitude_colormap: Colormap for vector magnitude
            arrow_length: Length scale for arrows
            save_path: Path to save figure
            elevation: Viewing elevation angle
            azimuth: Viewing azimuth angle
            **kwargs: Additional plotting parameters
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate magnitude
        magnitude = np.sqrt(U**2 + V**2 + W**2)
        
        # Normalize for visualization (optional)
        if kwargs.get('normalize', True):
            norm_factor = magnitude.max()
            if norm_factor > 0:
                U_norm = U / norm_factor * arrow_length
                V_norm = V / norm_factor * arrow_length
                W_norm = W / norm_factor * arrow_length
            else:
                U_norm, V_norm, W_norm = U, V, W
        else:
            U_norm, V_norm, W_norm = U * arrow_length, V * arrow_length, W * arrow_length
        
        # Subsample for clarity
        skip = kwargs.get('skip', 2)
        X_sub = X[::skip, ::skip, ::skip]
        Y_sub = Y[::skip, ::skip, ::skip]
        Z_sub = Z[::skip, ::skip, ::skip]
        U_sub = U_norm[::skip, ::skip, ::skip]
        V_sub = V_norm[::skip, ::skip, ::skip]
        W_sub = W_norm[::skip, ::skip, ::skip]
        mag_sub = magnitude[::skip, ::skip, ::skip]
        
        # Flatten for quiver
        X_flat = X_sub.flatten()
        Y_flat = Y_sub.flatten()
        Z_flat = Z_sub.flatten()
        U_flat = U_sub.flatten()
        V_flat = V_sub.flatten()
        W_flat = W_sub.flatten()
        mag_flat = mag_sub.flatten()
        
        # Plot quiver
        quiver = ax.quiver(
            X_flat, Y_flat, Z_flat,
            U_flat, V_flat, W_flat,
            mag_flat,
            cmap=magnitude_colormap,
            alpha=0.6,
            length=arrow_length,
            normalize=False
        )
        
        # Add colorbar
        cbar = plt.colorbar(quiver, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Magnitude')
        
        # Overlay trajectories
        if trajectories:
            for traj in trajectories:
                if traj.shape[1] >= 3:
                    ax.plot(
                        traj[:, 0], traj[:, 1], traj[:, 2],
                        alpha=0.6,
                        linewidth=2,
                        color=kwargs.get('traj_color', 'green')
                    )
        
        # Mark poles/attractors
        if poles is not None and len(poles) > 0:
            ax.scatter(
                poles[:, 0], poles[:, 1], poles[:, 2],
                s=kwargs.get('pole_size', 300),
                c=kwargs.get('pole_color', 'red'),
                marker=kwargs.get('pole_marker', 'o'),
                edgecolors='white',
                linewidths=2,
                label='Attractors/Poles'
            )
            ax.legend()
        
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_zlabel(kwargs.get('zlabel', 'Z'))
        ax.set_title(title)
        ax.view_init(elev=elevation, azim=azimuth)
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def visualize_auto(
        self,
        coords: Tuple[np.ndarray, ...],
        vectors: Tuple[np.ndarray, ...],
        **kwargs
    ) -> plt.Figure:
        """Automatically detect dimensionality and visualize appropriately.
        
        Args:
            coords: Tuple of coordinate arrays (X, Y) or (X, Y, Z)
            vectors: Tuple of vector component arrays (U, V) or (U, V, W)
            **kwargs: Passed to appropriate visualization method
        """
        if len(coords) == 2 and len(vectors) == 2:
            return self.visualize_2d(*coords, *vectors, **kwargs)
        elif len(coords) == 3 and len(vectors) == 3:
            return self.visualize_3d(*coords, *vectors, **kwargs)
        else:
            raise ValueError(f"Unsupported dimensionality: {len(coords)}D coordinates, {len(vectors)}D vectors")
    
    def save_vector_data(
        self,
        save_dir: Path,
        coords: Tuple[np.ndarray, ...],
        vectors: Tuple[np.ndarray, ...],
        prefix: str = "vector_field",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save vector field data in .npy format with metadata.
        
        Args:
            save_dir: Directory to save data
            coords: Tuple of coordinate arrays
            vectors: Tuple of vector component arrays
            prefix: Filename prefix
            metadata: Optional metadata dictionary to save as JSON
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save coordinates
        for i, coord in enumerate(coords):
            coord_name = ['X', 'Y', 'Z'][i]
            np.save(save_dir / f"{prefix}_coord_{coord_name}.npy", coord)
        
        # Save vectors
        vector_names = ['U', 'V', 'W']
        for i, vec in enumerate(vectors):
            vec_name = vector_names[i]
            np.save(save_dir / f"{prefix}_vector_{vec_name}.npy", vec)
        
        # Save magnitude
        magnitude = np.sqrt(sum(v**2 for v in vectors))
        np.save(save_dir / f"{prefix}_magnitude.npy", magnitude)
        
        # Save metadata
        if metadata:
            import json
            with open(save_dir / f"{prefix}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)


def create_3d_grid(
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 3D meshgrid for vector field computation.
    
    Args:
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        resolution: Number of points per dimension
    
    Returns:
        X, Y, Z meshgrid arrays
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z
