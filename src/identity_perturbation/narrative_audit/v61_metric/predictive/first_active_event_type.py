"""Predict the first non-pause event in the next repair episode.

This metric ignores leading idle gaps and asks whether the model gets the first
meaningful action right, which is often a better proxy for student intent than
the raw first event alone.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.first_active_event_type import *  # noqa: F403
