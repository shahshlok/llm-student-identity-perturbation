"""Score sequence similarity using event-type edit distance.

This metric rewards predicted event-type sequences that can be transformed into
the observed sequence with relatively few insertions, deletions, or
substitutions.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.event_type_edit_similarity import *  # noqa: F403
