"""Score sequence similarity using broader event-family edit distance.

This is a less brittle version of event-type edit similarity, since it compares
edit, run, pause, navigate, and submit families rather than raw event labels.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.event_family_edit_similarity import *  # noqa: F403
