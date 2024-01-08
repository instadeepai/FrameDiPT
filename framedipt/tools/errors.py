"""Error class for handled errors."""


class DataError(Exception):
    """Data exception."""


class MmcifParsingError(DataError):
    """Raised when mmcif parsing fails."""


class ResolutionError(DataError):
    """Raised when resolution isn't acceptable."""


class LengthError(DataError):
    """Raised when length isn't acceptable."""


class NoModeledChainError(DataError):
    """Raised when no chain is modeled in a structure."""


class NumberOfChainsError(DataError):
    """Raised when the number of chains isn't acceptable."""


class ChainNotFoundError(Exception):
    """Exception raised when a chain id is not found among all chains in Model."""


class ProteinMPNNError(Exception):
    """Raised when ProteinMPNN run fails."""


class PickleLoadError(Exception):
    """Raised when pickle load fails."""


class CPUUnpicklerError(Exception):
    """Raised when CPUUnpickler class load fails."""
