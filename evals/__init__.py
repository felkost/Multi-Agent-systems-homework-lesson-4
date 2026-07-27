"""Evaluation suite: the dataset, the graders and the experiment runner.

Kept out of the agent's own modules on purpose. Nothing here is imported at
runtime by `main.py`, so a broken evaluator can never break a research run.
"""
