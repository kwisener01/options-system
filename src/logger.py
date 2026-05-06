import logging
import logging.handlers
import os


def setup_logging(level: int = logging.INFO) -> None:
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "trader.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=7,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
