"""Predict the high-level repair motif of the next episode.

This metric compresses the episode into a short family pattern such as
`edit->run->submit`, which is easier to interpret pedagogically than a full raw
sequence.
"""

from identity_perturbation.narrative_audit.v61_eval.metrics.episode_motif import *  # noqa: F403
