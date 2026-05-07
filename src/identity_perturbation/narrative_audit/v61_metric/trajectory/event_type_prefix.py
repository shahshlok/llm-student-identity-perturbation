"""Score how well the predicted episode starts like the observed episode.

This metric looks at the early part of the sequence, including exact prefix
match and shared-prefix overlap, to capture whether the model gets the start of
the repair episode right.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.event_type_prefix import *  # noqa: F403
