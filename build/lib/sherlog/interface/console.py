from rich.console import Console

console = Console(markup=False)

def print(msg):
    """Alias for console printing."""
    
    console.print(msg)