from .logs import get

import altair as alt
import pandas as pd

import altair_viewer

logger = get("visualization")

def load(filename):
    """Load a file emitted by an Instrumenter.

    Parameters
    ----------
    filename : str

    Returns
    -------
    pandas.DataFrame
    """
    logger.info(f"Loading data from {filename}.")
    return pd.read_json(filename, lines=True)

def confidence_interval(data, x, y):
    """Construct a chart plotting the 95% confidence interval of x vs. y across seeds.

    Parameters
    ----------
    data : pandas.DataFrame

    x : str

    y : str

    Returns
    -------
    altair.Chart
    """
    logger.info(f"Building CI graph of {x} vs {y}.")
    line = alt.Chart(data).mark_line().encode(x=x, y=f"mean({y})")
    band = alt.Chart(data).mark_errorband(extent="ci").encode(x=x, y=y)
    return line + band

def dump(chart, filename):
    """
    Parameters
    ----------
    chart : altair.Chart

    filename : str
    """
    logger.info(f"Writing chart to {filename}.")
    chart.save(filename)

def show(chart):
    """Renders the given chart in a browser window.

    Parameters
    ----------
    chart : altair.Chart
    """
    altair_viewer.show(chart)