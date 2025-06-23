# fact_checker_agent/logger.py

import logging
import sys

# Define color codes for console output
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_logger(name: str):
    """Creates and configures a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# --- START: THE FIX ---
# Modified all log helpers to accept **kwargs to pass through extra arguments
# like exc_info=True to the underlying logger.
def log_info(logger, message, **kwargs):
    logger.info(f"{BColors.OKBLUE}{message}{BColors.ENDC}", **kwargs)

def log_success(logger, message, **kwargs):
    logger.info(f"{BColors.OKGREEN}{message}{BColors.ENDC}", **kwargs)

def log_warning(logger, message, **kwargs):
    logger.warning(f"{BColors.WARNING}{message}{BColors.ENDC}", **kwargs)

def log_error(logger, message, **kwargs):
    logger.error(f"{BColors.FAIL}{message}{BColors.ENDC}", **kwargs)
# --- END: THE FIX ---

def log_agent_start(logger, agent_name):
    logger.info(f"{BColors.OKCYAN}🚀 --- [AGENT START]: {agent_name} --- 🚀{BColors.ENDC}")

def log_agent_end(logger, agent_name):
    logger.info(f"{BColors.OKCYAN}✅ --- [AGENT END]: {agent_name} --- ✅{BColors.ENDC}")

def log_tool_call(logger, tool_name, params):
    logger.info(f"{BColors.WARNING}🔧 --- [TOOL CALL]: {tool_name} with params: {params} --- 🔧{BColors.ENDC}")

def log_api_request(logger, message):
    logger.info(f"{BColors.HEADER}📥 --- [API REQUEST]: {message} --- 📥{BColors.ENDC}")

def log_api_response(logger, message):
    logger.info(f"{BColors.HEADER}📤 --- [API RESPONSE]: {message} --- 📤{BColors.ENDC}")