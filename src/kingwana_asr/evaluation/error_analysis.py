"""Reserved extension points for qualitative and entity-level error analysis.

Entity scoring is intentionally deferred until validated entity annotations exist.
"""


class EntityLabelsUnavailableError(RuntimeError):
    """Raised when entity scoring is requested without validated labels."""
