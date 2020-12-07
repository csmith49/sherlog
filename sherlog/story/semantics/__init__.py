from . import torch, dice, pyro, storchastic
from .semantics import convert_arguments

def evaluate(value, context, algebra):
    return convert_arguments([value], context, algebra)[0]