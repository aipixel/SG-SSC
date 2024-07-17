# importing enum for enumerations
import enum
 

class VerbosityLevel(enum.Enum):
    ERROR = 1
    WARNING = 2
    INFO = 3


VERBOSITY = VerbosityLevel.WARNING


def set_level(level):
    global VERBOSITY
    VERBOSITY = level


def info(msg):
    if VERBOSITY == VerbosityLevel.INFO:
        print(msg)


def error(msg):
    if VERBOSITY == VerbosityLevel.ERROR:
        raise Exception(msg)
