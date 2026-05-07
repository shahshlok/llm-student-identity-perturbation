"""Score sequence similarity with longest common subsequence on event types.

This metric is tolerant to gaps, so it gives credit when the model predicts the
right kinds of actions in roughly the right order even if it misses some extras.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.event_type_lcss import *  # noqa: F403
