"""Configuration shared by the Perspektywy experiments."""

from pathlib import Path

FOCAL_UNIVERSITY = "Politechnika Poznanska"
MAIN_YEAR = 2025
MAIN_PAIR = (2025, 2026)
ALL_PAIRS = [(2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026)]
CRITERIA = [
    "Prestiz",
    "Absolwenci",
    "PotencjalNaukowy",
    "Innowacyjnosc",
    "EfektywnoscNaukowa",
    "WarunkiKsztalcenia",
    "Umiedzynarodowienie",
]
DISPLAY_NAMES = {
    "Prestiz": "Reputation",
    "Absolwenci": "Employability",
    "PotencjalNaukowy": "Research Potential",
    "Innowacyjnosc": "Innovation",
    "EfektywnoscNaukowa": "Research Effectiveness",
    "WarunkiKsztalcenia": "Teaching",
    "Umiedzynarodowienie": "Internationalization",
}
GROUP_WEIGHTS = {
    2022: [12, 12, 15, 8, 28, 10, 15],
    2023: [12, 12, 15, 8, 28, 10, 15],
    2024: [12, 12, 13, 8, 30, 10, 15],
    2025: [12, 12, 13, 8, 30, 10, 15],
    2026: [12, 12, 13, 8, 30, 10, 15],
}
OBJECTIVES = ["max"] * len(CRITERIA)
EXPERT_RANGE = [[0, 100]] * len(CRITERIA)
EPSILON = 1e-6
SEED = 23
METHODS = ["RTOPSIS", "SAW", "VIKOR", "COPRAS"]
METHODS_EXTENDED = ["RTOPSIS", "SAW", "VIKOR", "COPRAS", "ARAS", "WASPAS"]
MARGIN_ALPHAS = [0.0, 0.25, 0.5]
E4_REPETITIONS = 200
E4_REPETITIONS_EVOLUTIONARY = 30
E4_CRITERIA_SUBSETS = {
    2: ["EfektywnoscNaukowa", "Umiedzynarodowienie"],
    4: ["EfektywnoscNaukowa", "Umiedzynarodowienie", "Innowacyjnosc", "WarunkiKsztalcenia"],
    7: CRITERIA,
}
E4_DIRECT_CRITERION = "EfektywnoscNaukowa"
PFA_ALGORITHMS = ["direct", "lexicographic", "nlp", "evolutionary", "wm"]
SLOW_CRITERIA = ["Prestiz", "Absolwenci"]

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATA_PATH = REPO_ROOT / "data" / "perspektywy" / "perspektywy_criteria_technical_2022_2026.csv"
DEFAULT_OUTDIR = PACKAGE_DIR
