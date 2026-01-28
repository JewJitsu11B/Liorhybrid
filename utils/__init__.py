"""
Utilities for Bayesian Cognitive Field

Visualization, metrics, and diagnostics.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from .metrics import (
    compute_norm_conservation,
    compute_entropy,
    compute_local_correlation,
    compute_correlation_length,
    compute_effective_dimension
)

from .visualization import (
    plot_field_magnitude,
    plot_evolution_history,
    plot_correlation_structure,
    plot_eigenspectrum,
    animate_field_evolution
)

from .cpu_tasks import (
    read_lines,
    merge_json_files,
    count_tokens,
    list_files,
)

__all__ = [
    # Metrics
    'compute_norm_conservation',
    'compute_entropy',
    'compute_local_correlation',
    'compute_correlation_length',
    'compute_effective_dimension',

    # Visualization
    'plot_field_magnitude',
    'plot_evolution_history',
    'plot_correlation_structure',
    'plot_eigenspectrum',
    'animate_field_evolution',

    # CPU helpers
    'read_lines',
    'merge_json_files',
    'count_tokens',
    'list_files',
]
