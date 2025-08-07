
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

def plot_frontier(efficient_frontier_risk_data=None, efficient_frontier_return_data=None, asset_risk_data=None, asset_return_data=None, asset_labels=None, point_risk_data=None, point_return_data=None, point_labels=None, annualize_data=False):
    """
    Function to plot the results of the optimization.

    Options to display:
    - Full efficient frontier
    - Selected optimal point, with label
    - Original assets

    Arguments:
        efficient_frontier_risk_data: list of floats representing the risk (standard deviation) for the efficient frontier
        efficient_frontier_return_data: list of floats representing the return for the efficient frontier
        asset_risk_data: list of floats representing the risk (standard deviation) of the original assets
        asset_return_data: list of floats representing the returns of the orignal asset
        point_risk_data: list of floats representing the risk (standard deviation) of select points
        point_return_data: list of floats representing the returns of select points
        point_labels: list of strings for labels of select points
        annualize_data: boolean representing whether the data should be annualized. Not yet implemented.
    """
    
    def plot_points(risk_data, return_data, style, linewidth=2, labels=None, series_label=None):
        if risk_data is not None and return_data is not None:
            if len(risk_data) != len(return_data):
                raise ValueError("Risk and return data must be of the same length.")
            plt.plot(risk_data, return_data, style, linewidth=linewidth, label=series_label)
            offset = (max(risk_data) - min(risk_data)) * 0.02
            if labels is not None:
                for x, y, label in zip(risk_data, return_data, labels):
                    plt.text(x + offset, y, label, fontsize=9, ha='left', va='center')
        return

    fig, ax = plt.subplots()
    
    if annualize_data:
        periods_per_year = 252
        if efficient_frontier_risk_data is not None:
            efficient_frontier_risk_data = np.array(efficient_frontier_risk_data) * np.sqrt(periods_per_year)
        if efficient_frontier_return_data is not None:
            efficient_frontier_return_data = np.array(efficient_frontier_return_data) * periods_per_year
        if asset_risk_data is not None:
            asset_risk_data = np.array(asset_risk_data) * np.sqrt(periods_per_year)
        if asset_return_data is not None:
            asset_return_data = np.array(asset_return_data) * periods_per_year
        if point_risk_data is not None:
            if isinstance(point_risk_data, (list, np.ndarray)):
                point_risk_data = np.array(point_risk_data) * np.sqrt(periods_per_year)
            else:
                point_risk_data = point_risk_data * np.sqrt(periods_per_year)
        if point_return_data is not None:
            if isinstance(point_return_data, (list, np.ndarray)):
                point_return_data = np.array(point_return_data) * periods_per_year
            else:
                point_return_data = point_return_data * periods_per_year

    plot_points(efficient_frontier_risk_data, efficient_frontier_return_data, "g-", linewidth=3, series_label="Efficient Frontier")
    plot_points(asset_risk_data, asset_return_data, "ro", labels=asset_labels, series_label="Assets")

    # Define markers and colors for notable points
    markers = cycle(['o', 's', '^', 'D', 'P', '*', 'X', 'v', 'h', 'H'])
    colors = cycle(['b', 'm', 'c', 'y', 'k', '#ff7f0e', '#2ca02c'])

    # plot_points(point_risk_data, point_return_data, "bs", labels=point_labels, series_label="Notable points")
    # Plot individually labeled notable points
    if point_risk_data and point_return_data and point_labels:
        for x, y, label in zip(point_risk_data, point_return_data, point_labels):
            marker = next(markers)
            color = next(colors)
            ax.plot(x, y, marker=marker, color=color, label=label)


    plt.xlabel("Annualized Standard deviation" if annualize_data else "Standard deviation")
    plt.ylabel("Annualized Return" if annualize_data else "Return")
    plt.title("Efficient Frontier")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    return