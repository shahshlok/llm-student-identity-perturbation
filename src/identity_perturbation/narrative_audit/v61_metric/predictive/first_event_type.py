"""Predict the literal first event type in the next repair episode.

This is a strict metric: it rewards getting the very first observed action
exactly right, including cases where the student starts by running or
navigating instead of editing.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.first_event_type import *  # noqa: F403
