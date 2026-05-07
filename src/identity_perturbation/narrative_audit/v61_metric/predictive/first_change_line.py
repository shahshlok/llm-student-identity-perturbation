"""Predict where the first code edit will happen.

This metric scores the first edited line exactly and with nearby-line
tolerances, so we can tell whether the model is pointing at roughly the right
region even when exact line recovery is too strict.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.first_change_line import *  # noqa: F403
