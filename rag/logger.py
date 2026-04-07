import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        from datetime import datetime
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def get_logger():
    logger = logging.getLogger("rag_logger")

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / "app.log"

        formatter = MillisecondFormatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        # File
        fh = RotatingFileHandler(
            log_file,
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        # 🔥 Capture ALL prints/errors into log file
        logging.captureWarnings(True)

    return logger