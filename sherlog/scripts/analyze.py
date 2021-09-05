import click
import altair as alt
import pandas as pd
import altair_viewer

from .cli import cli
from ..interface.logs import get

logger = get("scripts.analyze")

def load(filename : str) -> pd.DataFrame:
    """Load a file emitted by an Instrumenter."""

    logger.info(f"Loading data from {filename}.")
    return pd.read_json(filename, lines=True)

def confidence_interval(data : pd.DataFrame, x : str, y : str) -> alt.Chart:
    """Construct a chart plotting the 95% confidence interval of x vs. y across seeds."""

    logger.info(f"Building CI graph of {x} vs {y}.")
    line = alt.Chart(data).mark_line().encode(x=x, y=f"mean({y})")
    band = alt.Chart(data).mark_errorband(extent="ci").encode(x=x, y=y)
    return line + band

def dump(chart : alt.Chart, filename : str):
    """Save the chart to file."""

    logger.info(f"Writing chart to {filename}.")
    chart.save(filename)

def show(chart : alt.Chart):
    """Renders the given chart in a browser window."""

    altair_viewer.show(chart)

@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-s", "--show/--no-show", default=True, show_default=True,
    help="Show plots after generation.")
@click.option("-o", "--output", type=click.Path(), help="Save plots to HTML file.")
@click.option("-p", "--plot", multiple=True, nargs=2, type=str, metavar="X Y",
    help="Plot X vs Y, where X and Y are columns in FILENAME.")
def analyze(filename, show, output, plot):
    """Analyze instrumentation logs."""

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