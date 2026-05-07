"""Predict whether the next repair episode contains any code edits.

This captures the broad edit-vs-no-edit distinction, which is useful for
testing whether the model understands that the student is still in active
repair mode.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.change_presence import *  # noqa: F403
