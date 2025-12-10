from dotenv import load_dotenv

from utils.llm_usage_tracking import enable_global_usage_tracking

from .inconsistency import (
    InconsistencyAgent,
    InconsistencyReport,
)


load_dotenv()
enable_global_usage_tracking()


__all__ = [
    "InconsistencyAgent",
    "InconsistencyReport",
]
