from consts import CANDIDATE_MAX_SEQUENCE_LENGTHS


class DatasetExceedsMaxSequenceLengthError(Exception):
    """The exception raised when some sequences in the dataset are too long."""

    def __init__(self) -> None:
        """Initialize DatasetExceedsMaxSequenceLengthError."""
        super().__init__(
            f"Error during max sequence length calculation: "
            f"the dataset contains some sequences that are longer than the max possible max sequence length "
            f"we can accommodate - {max(CANDIDATE_MAX_SEQUENCE_LENGTHS)}"
        )
