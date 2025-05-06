"""ANSI color codes for terminal output."""

# Basic colors
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
GRAY = "\033[90m"

# Text styles
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"

# Reset
END = "\033[0m"

# Common combinations for specific types of data
SUCCESS = GREEN
ERROR = RED
WARNING = YELLOW
INFO = BLUE
COST = MAGENTA
MODEL = CYAN
METRIC = BLUE
HEADER = BOLD
DIM = GRAY
