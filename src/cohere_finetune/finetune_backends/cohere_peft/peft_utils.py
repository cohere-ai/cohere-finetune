import logging
import os
from accelerate.logging import get_logger
from consts import ENVIRONMENT_MODE_KEY


# Create the logger for cohere_peft, where the accelerate logger will log only on the main process (main_process_only=True by default)
logging.basicConfig(
    level=logging.INFO if os.environ.get(ENVIRONMENT_MODE_KEY, "PROD") == "DEV" else logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = get_logger("cohere_peft")
