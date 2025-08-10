"""
Core prompting techniques for GSM8K benchmark.
"""

from .zero_shot import ZeroShotModule
from .few_shot import FewShotModule
from .cot import CoTModule
from .self_consistency import SelfConsistencyModule
from .prolog_style import PrologModule

__all__ = [
    'ZeroShotModule',
    'FewShotModule', 
    'CoTModule',
    'SelfConsistencyModule',
    'PrologModule'
]
