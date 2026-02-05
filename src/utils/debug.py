"""Debug utilities for saving analysis data and visualizations at every step.

This module provides utilities to save intermediate results, debug visualizations,
and analysis artifacts for comprehensive inspection of the analysis pipeline.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt


class DebugLogger:
    """Centralized debug logger for analysis pipeline."""
    
    def __init__(self, base_dir: Path, experiment_name: Optional[str] = None):
        """Initialize debug logger.
        
        Args:
            base_dir: Base directory for debug outputs
            experiment_name: Optional experiment name (defaults to timestamp)
        """
        self.base_dir = Path(base_dir)
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.debug_dir = self.base_dir / "debug" / experiment_name
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.subdirs = {
            'arrays': self.debug_dir / 'arrays',
            'figures': self.debug_dir / 'figures',
            'metadata': self.debug_dir / 'metadata',
            'trajectories': self.debug_dir / 'trajectories',
            'flow_fields': self.debug_dir / 'flow_fields',
            'embeddings': self.debug_dir / 'embeddings',
            'divergence': self.debug_dir / 'divergence',
            'attractors': self.debug_dir / 'attractors',
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        self.log_file = self.debug_dir / "debug_log.txt"
        self.log(f"Debug logger initialized: {self.experiment_name}")
    
    def log(self, message: str):
        """Append message to debug log file with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        print(log_entry.strip())
    
    def save_array(
        self,
        array: np.ndarray,
        name: str,
        subdir: str = 'arrays',
        metadata: Optional[Dict] = None
    ):
        """Save numpy array with optional metadata.
        
        Args:
            array: Numpy array to save
            name: Name for the array (without extension)
            subdir: Subdirectory key from self.subdirs
            metadata: Optional metadata to save alongside
        """
        save_dir = self.subdirs.get(subdir, self.debug_dir / subdir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        array_path = save_dir / f"{name}.npy"
        np.save(array_path, array)
        self.log(f"Saved array: {array_path} (shape: {array.shape})")
        
        if metadata:
            meta_path = save_dir / f"{name}_metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def save_figure(
        self,
        fig: plt.Figure,
        name: str,
        subdir: str = 'figures',
        dpi: int = 200
    ):
        """Save matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            name: Name for the figure (without extension)
            subdir: Subdirectory key from self.subdirs
            dpi: DPI for saved figure
        """
        save_dir = self.subdirs.get(subdir, self.debug_dir / subdir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        fig_path = save_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        self.log(f"Saved figure: {fig_path}")
        plt.close(fig)
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        name: str,
        subdir: str = 'metadata'
    ):
        """Save metadata dictionary as JSON.
        
        Args:
            metadata: Metadata dictionary
            name: Name for the metadata file
            subdir: Subdirectory key from self.subdirs
        """
        save_dir = self.subdirs.get(subdir, self.debug_dir / subdir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        meta_path = save_dir / f"{name}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(x) for x in obj]
            return obj
        
        metadata_converted = convert_types(metadata)
        
        with open(meta_path, 'w') as f:
            json.dump(metadata_converted, f, indent=2)
        self.log(f"Saved metadata: {meta_path}")
    
    def save_trajectories(
        self,
        trajectories: List[np.ndarray],
        name: str,
        metadata: Optional[Dict] = None
    ):
        """Save trajectory data.
        
        Args:
            trajectories: List of trajectory arrays
            name: Base name for trajectory files
            metadata: Optional metadata
        """
        save_dir = self.subdirs['trajectories']
        
        for i, traj in enumerate(trajectories):
            traj_path = save_dir / f"{name}_traj_{i:03d}.npy"
            np.save(traj_path, traj)
        
        self.log(f"Saved {len(trajectories)} trajectories: {name}_traj_*.npy")
        
        if metadata:
            self.save_metadata(metadata, f"{name}_trajectories", subdir='trajectories')
    
    def save_flow_field(
        self,
        grid: np.ndarray,
        flow_vectors: np.ndarray,
        name: str,
        divergence: Optional[np.ndarray] = None,
        curl: Optional[Union[np.ndarray, tuple]] = None,
        metadata: Optional[Dict] = None
    ):
        """Save flow field data including divergence and curl.
        
        Args:
            grid: Grid coordinates
            flow_vectors: Flow vector field
            name: Base name for flow field files
            divergence: Optional divergence field
            curl: Optional curl field (single array or tuple for 3D)
            metadata: Optional metadata
        """
        save_dir = self.subdirs['flow_fields']
        
        np.save(save_dir / f"{name}_grid.npy", grid)
        np.save(save_dir / f"{name}_vectors.npy", flow_vectors)
        self.log(f"Saved flow field: {name}")
        
        if divergence is not None:
            div_path = save_dir / f"{name}_divergence.npy"
            np.save(div_path, divergence)
            self.log(f"Saved divergence: {div_path}")
        
        if curl is not None:
            if isinstance(curl, tuple):
                # 3D curl
                for i, comp in enumerate(curl):
                    curl_path = save_dir / f"{name}_curl_{['x', 'y', 'z'][i]}.npy"
                    np.save(curl_path, comp)
                self.log(f"Saved curl components: {name}_curl_*.npy")
            else:
                # 2D curl (scalar)
                curl_path = save_dir / f"{name}_curl.npy"
                np.save(curl_path, curl)
                self.log(f"Saved curl: {curl_path}")
        
        if metadata:
            self.save_metadata(metadata, f"{name}_flow_field", subdir='flow_fields')
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        name: str,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        """Save embedding data.
        
        Args:
            embeddings: Embedding array
            name: Name for embedding file
            labels: Optional labels for embeddings
            metadata: Optional metadata
        """
        save_dir = self.subdirs['embeddings']
        
        np.save(save_dir / f"{name}.npy", embeddings)
        self.log(f"Saved embeddings: {name}.npy (shape: {embeddings.shape})")
        
        if labels is not None:
            np.save(save_dir / f"{name}_labels.npy", labels)
        
        if metadata:
            self.save_metadata(metadata, f"{name}_embeddings", subdir='embeddings')
    
    def save_divergence_analysis(
        self,
        divergence: np.ndarray,
        name: str,
        attractors: Optional[np.ndarray] = None,
        repellers: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ):
        """Save divergence analysis results.
        
        Args:
            divergence: Divergence field
            name: Base name for divergence files
            attractors: Optional attractor locations
            repellers: Optional repeller locations
            metadata: Optional metadata
        """
        save_dir = self.subdirs['divergence']
        
        np.save(save_dir / f"{name}_divergence.npy", divergence)
        self.log(f"Saved divergence: {name}_divergence.npy")
        
        if attractors is not None:
            np.save(save_dir / f"{name}_attractors.npy", attractors)
            self.log(f"Saved attractors: {name}_attractors.npy")
        
        if repellers is not None:
            np.save(save_dir / f"{name}_repellers.npy", repellers)
            self.log(f"Saved repellers: {name}_repellers.npy")
        
        if metadata:
            self.save_metadata(metadata, f"{name}_divergence", subdir='divergence')
    
    def create_summary(self, metrics: Dict[str, Any]):
        """Create summary file with all metrics and analysis results.
        
        Args:
            metrics: Dictionary of metrics and results
        """
        summary_path = self.debug_dir / "analysis_summary.json"
        
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'debug_directory': str(self.debug_dir)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.log(f"Created analysis summary: {summary_path}")
    
    def get_subdir(self, key: str) -> Path:
        """Get path to subdirectory.
        
        Args:
            key: Subdirectory key
        
        Returns:
            Path to subdirectory
        """
        return self.subdirs.get(key, self.debug_dir / key)


def save_step_debug(
    debug_logger: DebugLogger,
    step_name: str,
    arrays: Optional[Dict[str, np.ndarray]] = None,
    figures: Optional[Dict[str, plt.Figure]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save debug information for a single analysis step.
    
    Args:
        debug_logger: DebugLogger instance
        step_name: Name of the analysis step
        arrays: Dictionary of arrays to save
        figures: Dictionary of figures to save
        metadata: Dictionary of metadata to save
    """
    debug_logger.log(f"=== Starting debug save for step: {step_name} ===")
    
    if arrays:
        for name, array in arrays.items():
            debug_logger.save_array(array, f"{step_name}_{name}")
    
    if figures:
        for name, fig in figures.items():
            debug_logger.save_figure(fig, f"{step_name}_{name}")
    
    if metadata:
        debug_logger.save_metadata(metadata, f"{step_name}_step")
    
    debug_logger.log(f"=== Completed debug save for step: {step_name} ===")
