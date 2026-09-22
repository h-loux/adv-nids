"""Attacks package — FGSM, PGD, and shared projection."""
from nids.attacks.projection import project
from nids.attacks.fgsm import fgsm
from nids.attacks.pgd import pgd

__all__ = ["project", "fgsm", "pgd"]
