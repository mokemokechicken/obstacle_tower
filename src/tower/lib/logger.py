from logging import StreamHandler, DEBUG, getLogger, Formatter, INFO, WARNING, ERROR, FileHandler
from pathlib import Path


def setup_logger(log_filename, level):
    Path(log_filename).absolute().parent.mkdir(exist_ok=True, parents=True)

    level = get_log_level(level)
    formatter = Formatter('%(asctime)s@%(name)s %(levelname)s # %(message)s')

    root = getLogger()
    for h in root.handlers:
        root.removeHandler(h)

    getLogger('tower').setLevel(level)

    file_handler = FileHandler(log_filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)


def get_log_level(level: str):
    level = level.lower()
    if level == 'debug':
        return DEBUG
    elif level == 'warning':
        return WARNING
    elif level == 'error':
        return ERROR
    else:
        return INFO
