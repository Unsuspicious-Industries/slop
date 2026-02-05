from typing import Any, Dict, List


def compress_trajectory(trajectory: List[Dict[str, Any]], sample_rate: int = 1) -> List[Dict[str, Any]]:
    if sample_rate <= 1:
        return trajectory
    return [step for idx, step in enumerate(trajectory) if idx % sample_rate == 0]
