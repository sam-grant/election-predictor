import os
# import awkward as ak
import matplotlib.pyplot as plt
# import math
# import numpy as np
# from scipy import stats
from matplotlib.ticker import ScalarFormatter
# import matplotlib.colors as colors

class Plot:
    def __init__(self):
        """  init """
        style_path = os.path.join(os.path.dirname(__file__), 'election.mplstyle')
        plt.style.use(style_path)

    def sci_not(self, ax, cbar=None): 
        """  
            Set scientific notation on axes
            Condition: log scale is not used and the absolute limit is >= 1e4 or <= 1e-4 
        """ 
        # Access the max values of the axes
        xmax, ymax = ax.get_xlim()[1], ax.get_ylim()[1]
        if ax.get_xscale() != 'log' and (abs(xmax) >= 1e4 or abs(xmax) <= 1e-4): # x-axis 
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True)) # Use math formatting 
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # Set scientific notation
        if ax.get_yscale() != 'log' and (abs(ymax) >= 1e4 or abs(ymax) <= 1e-4): # y-axis
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if cbar is not None: # Colour bar 
            # Access the max value of the cbar range
            cmax = cbar.norm.vmax
            if abs(cmax) >= 1e4 or abs(cmax) <= 1e-4:
                cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # Use math formatting
                cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Set scientific notation

    def plot_graph(
        self, x, y, xerr=None, yerr=None,
        title=None, xlabel=None, ylabel=None,
        xmin=None, xmax=None, ymin=None, ymax=None,
        col='black', linestyle='None', fout='graph.png', 
        log_x=False, log_y=False, NDPI=300,
        show=True, save=True, ax=None
        ):
        """  
        Plot a scatter graph with error bars (if included)
        """  
        # Use provided axis or create new figure and axis
        new_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            new_fig = True

        if xerr is None: # If only using yerr
            xerr = [0] * len(x) 
        if yerr is None: # If only using xerr 
            yerr = [0] * len(y) 

        # Convert string labels to numeric positions
        # if isinstance(x[0], str):  # Check if x contains strings
        #     x_labels = x
        #     x = range(len(x))  # Convert strings to numeric indices for plotting
        # else:
        #     x_labels = None

        # Create graph
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=col, markersize=4, ecolor=col, capsize=2, elinewidth=1, linestyle=linestyle, linewidth=1)

        # Set axis limits
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

        # Log scale
        if log_x: 
            ax.set_xscale("log")
        if log_y: 
            ax.set_yscale("log")
            
        # Set title, xlabel, and ylabel
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Set the x-axis labels if x contains strings
        # if x_labels is not None:
        #     ax.set_xticks(range(len(x_labels)))  # Set the positions for the labels
        #     ax.set_xticklabels(x_labels, rotation=45, ha='right')  # Set the labels with rotation

        # Scientific notation
        self.sci_not(ax) 

        if new_fig:
            plt.tight_layout()

            if save:
                plt.savefig(fout, dpi=300, bbox_inches="tight")
                print(f"\n---> Wrote:\n\t{fout}")
            
            if show:
                plt.show()

            plt.close()

    def plot_graph_overlay(
        self, graphs_,
        title=None, xlabel=None, ylabel=None,
        xmin=None, xmax=None, ymin=None, ymax=None,
        leg_pos='best', linestyle='None', fout='graph.png',
        y_lines=None, x_lines=None,
        log_x=False, log_y=False, NDPI=300, 
        show=True, save=True
        ):
        """  
        Overlay many scatter graphs
        """  
        # Create figure and axes
        fig, ax = plt.subplots()

        # Loop through graphs and plot
        for i, (label, graph_) in enumerate(graphs_.items()):

            # Just to be explicit
            x = graph_[0]
            y = graph_[1]
            xerr = graph_[2] #FIXME: do you really have to write None, None every time? Can we improve this?
            yerr = graph_[3] 

            # Error bars
            if xerr is None: # If only using yerr
                xerr = [0] * len(x) 
            if yerr is None: # If only using xerr 
                yerr = [0] * len(y) 

            # Create this graph
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o',  label=label, markersize=4, capsize=2, elinewidth=1, linestyle=linestyle, linewidth=1)

        # Set axis limits
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

        # Log scale
        if log_x: 
            ax.set_xscale("log")
        if log_y: 
            ax.set_yscale("log")

        # Set title, xlabel, and ylabel
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Scientific notation 
        self.sci_not(ax) 

        # Legend
        ax.legend(loc=leg_pos)

        # Lines
        if y_lines:
            for y_line in y_lines: 
                ax.axhline(y=y_line, color='gray', linestyle='--')
        if x_lines:
            for x_line in x_lines: 
                ax.axvline(x=x_line, color='gray', linestyle='--')

        # Draw
        plt.tight_layout()

        # Save 
        if save:
            plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
            print("\n---> Wrote:\n\t", fout)

        # Save
        if show:
            plt.show()

        # Clear memory
        plt.close(fig)