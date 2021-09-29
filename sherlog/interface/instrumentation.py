from minotaur import Minotaur

minotaur = Minotaur()

def instrument(filepath : str):
    """Write instrumentation messages to the indicated file."""

    minotaur.add_filepath_handler(filepath)