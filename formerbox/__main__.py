import logging
import os
import sys

from formerbox.cli import main

if os.environ.get("DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
sys.path.insert(0, PROJECT_DIR)

LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"
logging.basicConfig(format=LOGGING_FORMAT, level=LEVEL)
logging.getLogger("filelock").setLevel(logging.WARNING)


def run_main() -> None:
    main(prog="formerbox")


if __name__ == "__main__":
    run_main()
