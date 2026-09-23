"""Attacks package — FGSM, PGD, CW, zero-order, and shared projection."""
from nids.attacks.result import AttackResult
from nids.attacks.projection import project
from nids.attacks.fgsm import fgsm
from nids.attacks.pgd import pgd
from nids.attacks.carlini_wagner import carlini_wagner as cw
from nids.attacks.zeroth_order import nes_attack, transfer_attack
from nids.attacks.deepfool import deepfool
from nids.attacks.jsma import jsma

__all__ = ["AttackResult", "project", "fgsm", "pgd", "cw",
           "nes_attack", "transfer_attack", "deepfool", "jsma"]
