import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def set_renderer(renderer="plotly_mimetype+notebook"):
    """
    Set the default renderer for Plotly plots.

    Parameters:
    renderer (str): The name of the renderer to set as the default. Defaults to "notebook_connected".

    Returns:
    None
    """
    pio.renderers.default = renderer

def plot_data(x=None, y=None, z=None, size=None, colour=None, title="", colour_title="", x_label="", y_label="", name="", mode="markers", text="", fill=None, **traces):
    """
    General purpose function for plotting scatter plots in plotly.

    Parameters:
    - x: list or array-like, optional. x-coordinates of the data points.
    - y: list or array-like, optional. y-coordinates of the data points.
    - z: list or array-like, optional. z-coordinates of the data points for 3D plots.
    - size: list or array-like, optional. Sizes of the markers.
    - colour: list or array-like, optional. Colors of the markers.
    - title: str, optional. Title of the plot.
    - colour_title: str, optional. Title of the colorbar.
    - x_label: str, optional. Label for the x-axis.
    - y_label: str, optional. Label for the y-axis.
    - name: str, optional. Name of the trace.
    - mode: str, optional. Mode of the scatter plot.
    - text: list or array-like, optional. Text labels for the data points.
    - fill: str, optional. Fill type for the markers.
    - **traces: dict, optional. Additional traces to be added to the plot.

    Returns:
    - fig: plotly.graph_objects.Figure. The scatter plot figure.
    """
    fig = go.Figure(layout={
        "title": title,
        "xaxis": {"title": x_label},
        "yaxis": {"title": y_label}
    })

    marker = dict()
    
    if size is not None:
        marker["size"] = size
        marker["sizeref"] = 0.01
    if colour is not None:
        marker["color"] = colour
        marker["showscale"] = True
        marker["colorbar"] = dict(title=colour_title)
    
    if z is None:
        data = go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name,
            text=text,
            fill=fill,
            marker=marker,
        )
    else:
        data = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode=mode,
            name=name,
            text=text,
            marker=marker,
        )

    if x is not None and y is not None:
        fig.add_trace(data)
    
    for t in traces:
        fig.add_trace(traces[t])
    
    return fig

def create_trace(x=None, y=None, z=None, size=None, colour=None, colour_title="", name="", mode="lines", text="", fill=None):
    """
    Create a trace for a plot.

    Parameters:
    - x (list): x-coordinates of the trace.
    - y (list): y-coordinates of the trace.
    - z (list): z-coordinates of the trace (for 3D plots).
    - size (int): size of the markers.
    - colour (list): colour of the markers.
    - colour_title (str): title for the colour bar.
    - name (str): name of the trace.
    - mode (str): mode of the trace (e.g., 'lines', 'markers', 'lines+markers').
    - text (list): text associated with each marker.
    - fill (str): fill type for the trace (for area plots).

    Returns:
    - trace (go.Scatter or go.Scatter3d): the created trace object.
    """
    marker = dict()
    
    if size is not None:
        marker["size"] = size
        marker["sizeref"] = 0.01
    if colour is not None:
        marker["color"] = colour
        marker["showscale"] = True
        marker["colorbar"] = dict(title=colour_title)
    
    if z is None:
        trace = go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name,
            text=text,
            fill=fill,
            marker=marker
        )
    else:
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode=mode,
            name=name,
            text=text,
            marker=marker
        )
    
    return trace

def create_histogram(x=None, y=None, histnorm="", name=""):
    """
    Create a histogram plot.

    Parameters:
    - x: List or array-like object containing the data for the x-axis.
    - y: Not used in this function.
    - histnorm: Specifies the type of normalization used for the histogram. Default is "" (no normalization).
    - name: Name of the histogram plot.

    Returns:
    - A histogram plot object.

    """
    return go.Histogram(
        x=x,
        y=None,
        histnorm=histnorm,
        name=name,
    )

def create_candle_stick(x=None, open=None, close=None, high=None, low=None, name=""):
    """
    Create a candlestick chart object.

    Args:
        x (list): List of x-axis values.
        open (list): List of opening prices.
        close (list): List of closing prices.
        high (list): List of high prices.
        low (list): List of low prices.
        name (str): Name of the candlestick chart.

    Returns:
        go.Candlestick: Candlestick chart object.
    """
    return go.Candlestick(
        x=x,
        open=open,
        close=close,
        high=high,
        low=low,
        name=name,
    )

def plot_collection(plots, rows=1, cols=1, title="", subplot_titles=[], x_labels={}, y_labels={}, height=1000):
    """
    Create a collection of plots arranged in a grid layout.

    Args:
        plots (dict): A dictionary containing the plots to be added to the collection. The keys represent the position of the plot in the grid (row, col), and the values are instances of the `Plot` class.
        rows (int): The number of rows in the grid layout. Default is 1.
        cols (int): The number of columns in the grid layout. Default is 1.
        title (str): The title of the plot collection. Default is an empty string.
        subplot_titles (list): A list of titles for each subplot. The length of the list should be equal to the number of subplots. Default is an empty list.
        x_labels (dict): A dictionary mapping the position of each subplot to its x-axis label. Default is an empty dictionary.
        y_labels (dict): A dictionary mapping the position of each subplot to its y-axis label. Default is an empty dictionary.
        height (int): The height of the plot collection in pixels. Default is 1000.

    Returns:
        fig (plotly.graph_objects.Figure): The plotly figure object representing the plot collection.
    """
    specs = [
        [{"type": "xy"} for c in range(cols)] 
        for r in range(rows)
    ]
    
    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=subplot_titles,
        specs=specs,
    )
    
    fig.update_layout({
        "title": title,
        "height": height,
    })

    # Add traces
    for k in plots:
        for i in range(len(plots[k].data)):
            fig.add_trace(plots[k].data[i], row=k[0], col=k[1])

    # Update axes
    for k in plots:
        fig.update_xaxes(title_text=x_labels.get(k, ""), row=k[0], col=k[1])
        fig.update_yaxes(title_text=y_labels.get(k, ""), row=k[0], col=k[1])

    return fig