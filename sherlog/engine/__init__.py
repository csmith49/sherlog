# making an engine requires three things:
# 1. what to do with constants
# 2. what to do with built-ins
# 3. how to wrap/unwrap external functions

from .model import Statement, Model
from .store import Store
from . import value
from .algebra import Algebra, run, evaluate