"""The general plottings with scientific format are described."""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


class Plots():
    """_General class for multipurpose plotting
    """
    def __init__(self) -> None:

        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rc('font', size=7)
        mpl.rc('axes', titlesize=20)
        style_path, _ =os.path.split(__file__)
        # plt.style.use(['nature', 'science', 'no-latex'])
        plt.style.use([os.path.join(style_path, 'styles', 'nature.mplstyle'),
                            os.path.join(style_path, 'styles', 'science.mplstyle'),
                            os.path.join(style_path, 'styles', 'no-latex.mplstyle')])
        plt.rcParams['text.usetex'] = False
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'


    def plot_scatter(self, ax, x_values, y_values, color, legend_name=None, size=3):
        """_General function for plotting scatter plots
        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes
            x_values (np.array): x values
            y_values (np.array): y values
            legend_name (str): legend name. Defaults to None.
            color (str): color
            size (int): size. Defaults to 3.
        """
        ax.scatter(x_values, y_values, label=legend_name, color=color, s=size)


    def plot_line(self, ax, x_values, y_values, color, legend_name=None):
        """_General function for plotting line plots
        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes
            x_values (np.array): x values
            y_values (np.array): y values
            legend_name (str): legend name. Defaults to None.
            color (str): color
        """
        ax.plot(x_values, y_values, label=legend_name, color=color)


    def set_labels(self, ax, x_label, y_label):
        """_General function for setting labels
        Args:
            ax (matplotlib.axes._subplots.AxesSubplot): matplotlib axes
            x_label (str): x label
            y_label (str): y label
        """
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
