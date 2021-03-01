from .problem import Register
from . import problem
from . import engine
from . import inference
from .interface import console
from .logs import enable_verbose_output
import itertools

def batch(data, batch_size=1, magnification=1):
    args = [itertools.chain(*itertools.repeat(data, magnification))] * batch_size
    for batch in itertools.zip_longest(*args):
        yield [b for b in batch if b is not None]