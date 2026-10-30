"""Run model comparison after the final XGBoost bundle has been trained.

For the complete protocol, prefer ``python -m experiments.run_all``.
"""

from .run_all import run


if __name__ == "__main__":
    run()
