from .nn import MNISTNetwork, add

from sherlog.tooling.evaluation.model import Model
import sherlog

# load the source on import
with open("source.sl", 'r') as f:
    SOURCE = f.read()