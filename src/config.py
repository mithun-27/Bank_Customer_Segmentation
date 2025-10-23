import os
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Paths:
    data_dir: str = os.path.join(PROJECT_ROOT, "data")
    reports_dir: str = os.path.join(PROJECT_ROOT, "reports")

@dataclass
class Settings:
    # If you want to force specific features, set e.g. ["Age","AnnualIncome","CreditScore", ...]
    forced_features: list[str] | None = None
    # K range to test
    k_min: int = 2
    k_max: int = 10
    random_state: int = 42
    n_init: int = 20

PATHS = Paths()
SETTINGS = Settings()
