from datetime import datetime


def get_timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def get_run_id() -> str:
    return datetime.now().strftime("run_%Y_%m_%d_%H%M%S")