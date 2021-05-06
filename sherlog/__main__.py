import click
from .interface import console
from . import logs
from .tooling import instrumentation
from .program import load
from .inference import Optimizer, minibatch, Batch
from torch.optim import SGD, Adam
from rich.progress import track

@click.group()
@click.option("-l", "--log", multiple=True,
    type=click.Choice(logs.logged_modules(), case_sensitive=False),
    help="Enable/disable verbose output via logging")
def main(log): 
    """Simple wrapper around common manipulations of Sherlog programs."""
    if log: logs.enable(*log)

@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-e", "--epochs", default=300, help="Number of training epochs.")
@click.option("-o", "--optimizer", default="sgd",
    type=click.Choice(["sgd", "adam"], case_sensitive=False), show_default=True,
    help="Optimization strategy for training parameters.")
@click.option("-l", "--learning-rate", default=0.01, show_default=True,
    help="Optimizer learning rate.")
@click.option("-s", "--samples", default=100, show_default=True,
    help="Samples-per-explanation in gradient estimation.")
@click.option("-i", "--instrument", type=click.Path(),
    help="Output file for instrumentation logs.")
@click.option("-r", "--resolution", default=50, help="Instrumentation resolution (in epochs).")
@click.option("-b", "--batch-size", default=10, help="Batch size.")
def train(filename, epochs, optimizer, learning_rate, samples, instrument, resolution, batch_size):
    """Train FILENAME with the provided parameters."""
    
    # load the problem and build the optimizer
    program, evidence = load(filename)

    optimizer = Optimizer(program, optimizer=optimizer, learning_rate=learning_rate)

    # build the instrumenter
    instrumenter = instrumentation.Instrumenter(instrument, context={
        "seed" : instrumentation.seed(),
        "benchmark" : filename,
        "epochs" : epochs,
        "optimizer" : optimizer,
        "learning rate" : learning_rate,
        "samples" : samples
    })

    for epoch in range(epochs):
        for batch in minibatch(evidence, batch_size):
            with optimizer as o:
                batch = Batch(batch)
                o.maximize(batch.objective(program, samples=samples))

        if epoch % resolution == 0:
            log = {"epoch" : epoch}
            for k, v in program.parameter_map.items():
                log[k] = v
                log[f"{k} grad"] = v.grad
            instrumenter.emit(**log)

    if instrument:
        instrumenter.flush()

    # print the values of the learned parameters
    console.print("MLE Results")
    for name, parameter in program.parameter_map.items():
        console.print(f"{name}: {parameter:f}")

@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-f", "--format", default="matplotlib",
    type=click.Choice(["dot", "matplotlib"], case_sensitive=False),
    help="Output format to render graph in")
@click.option("-o", "--output", type=click.Path(exists=False),
    help="File to write graph to")
def render(filename, format, output):
    """Render the computation graph generated by a single execution of FILENAME."""

    # import the relevant drawing tools
    from .explanation.semantics.graph import draw_matplotlib, to_dot

    # load the problem
    program, evidence = load(filename)

    # get a single piece of evidence and a single story
    explanation = next(program.explanations(evidence[0], quantity=1))

    # build the graph
    graph = explanation.graph()

    # and draw
    if output:
        if format == "matplotlib":
            draw_matplotlib(graph, output=output)
        elif format == "dot":
            to_dot(graph, output)
        else:
            print(f"Unsupported format {format}.")
    else:
        if format == "matplotlib":
            draw_matplotlib(graph)
        elif format == "dot":
            print(f"Format {format} requires output filepath be provided.")
        else:
            print(f"Unsupported format {format}.")

@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-s", "--show/--no-show", default=True, show_default=True,
    help="Show plots after generation.")
@click.option("-o", "--output", type=click.Path(), help="Save plots to HTML file.")
@click.option("-p", "--plot", multiple=True, nargs=2, type=str, metavar="X Y",
    help="Plot X vs Y, where X and Y are columns in FILENAME.")
def analyze(filename, show, output, plot):
    """Analyze instrumentation logs."""

    # import the relevant analysis tools
    from .tooling.visualization import load, confidence_interval, dump

    # load the data
    data = load(filename)

    # build a simple chart, if any plots are provided
    chart = None
    for x, y in plot:
        if chart:
            chart &= confidence_interval(data, x, y)
        else:
            chart = confidence_interval(data, x, y)

    # show the charts
    if chart and show:
        chart.show()

    # save the resulting chart (if indicated)
    if output and chart: dump(chart, output)

@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-b", "--burn-in", type=int, default=100)
@click.option("-s", "--samples", type=int, default=100)
def explain(filename, burn_in, samples):
    """Sample an explanation for each provided piece of evidence."""

    program, evidence = load(filename)

    for ev in evidence:
        console.print(f"Explaining {ev}:")
        explanation = program.sample_explanation(ev, burn_in=burn_in, samples=samples)
        console.print(f"Model: {explanation.model}")
        console.print(f"Meet:  {explanation.meet}")
        console.print(f"Avoid: {explanation.avoid}\n")

if __name__ == "__main__":
    main()