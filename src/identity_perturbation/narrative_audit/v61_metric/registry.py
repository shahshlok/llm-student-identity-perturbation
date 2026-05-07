"""Registry of all V6.1 metrics grouped by type.

This module is the single source of truth for which metrics are active in the
current evaluator and how they are grouped for reporting.
"""

from .predictive import (
    change_presence,
    event_count_buckets,
    first_active_event_type,
    first_change_line,
    first_event_family,
    first_event_type,
    idle_gap_presence,
    run_presence,
)
from .trajectory import (
    episode_motif,
    event_family_dtw,
    event_family_edit_similarity,
    event_family_lcss,
    event_type_edit_similarity,
    event_type_lcss,
    event_type_overlap,
    event_type_prefix,
)

PREDICTIVE_METRICS = (
    first_event_type,
    first_event_family,
    first_active_event_type,
    change_presence,
    run_presence,
    idle_gap_presence,
    first_change_line,
    event_count_buckets,
)

TRAJECTORY_METRICS = (
    event_type_prefix,
    event_type_overlap,
    event_type_edit_similarity,
    event_family_edit_similarity,
    event_type_lcss,
    event_family_lcss,
    event_family_dtw,
    episode_motif,
)

ALL_METRICS = PREDICTIVE_METRICS + TRAJECTORY_METRICS
