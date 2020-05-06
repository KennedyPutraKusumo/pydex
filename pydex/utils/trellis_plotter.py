from matplotlib import pyplot as plt
import numpy as np


class TrellisPlotter:
    def __init__(self):
        self.data = None
        self.data_sets = None
        self.intervals = None
        # options
        self.figsize = None
        self.constrained_layout = False
        self.marker = "o"
        self.markersize = 50
        self.n_xticks = 3
        self.n_yticks = 3
        self.xspace = 0.1
        self.yspace = 0.1
        self.xlabel = ""
        self.ylabel = ""
        self.xticks = None
        self.yticks = None
        self.xticklabels = None
        self.yticklabels = None
        self.oaxis_size = 0.20
        self.oaxis_n_xticks = 3
        self.oaxis_n_yticks = 3
        self.oaxis_xticks = None
        self.oaxis_yticks = None
        self.oaxis_xticklabels = None
        self.oaxis_yticklabels = None
        self.oaxis_bar_transparency = 0.6
        self.oaxis_xlabel = ""
        self.oaxis_ylabel = ""
        # computed
        self.bounds = None
        self.bins = None
        self.group_bins = None
        self.n_groups = None
        self.grouped_data = None
        # private
        self._multiple_data_sets = None

    def initialize(self):
        if isinstance(self.data, list):
            self._multiple_data_sets = True
        else:
            self._multiple_data_sets = False

        # check data's validity
        if self._multiple_data_sets:
            if not np.all([isinstance(datum, np.ndarray) for datum in self.data]):
                raise SyntaxError("All data sets must be a numpy array.")
            if not np.all([datum.ndim == 2 for datum in self.data]):
                raise SyntaxError("All data sets must be a 2D-array.")
            if not np.all([
                datum.shape[1] == self.data[0].shape[1]
                for datum in self.data
            ]):
                raise SyntaxError(
                    "Dimensions of points in the different data sets are inconsistent"
                )
        else:
            if not isinstance(self.data, np.ndarray):
                raise SyntaxError("Data must be a numpy array.")
            if self.data.ndim != 2:
                raise SyntaxError("Data must be a 2D-array.")

        # check if all data sets have the same dimension

        # check interval's validity
        if not isinstance(self.intervals, np.ndarray):
            raise SyntaxError("Intervals must be a numpy array.")
        if self.intervals.ndim != 1:
            raise SyntaxError("Intervals must be a 1D-array.")

        # check if interval agrees with given data
        if self._multiple_data_sets:
            if self.intervals.shape[0] != (self.data[0].shape[1] - 2):
                raise SyntaxError("Dimensions in given interval and data does not agree.")
        else:
            if self.intervals.shape[0] != (self.data.shape[1] - 2):
                raise SyntaxError("Dimensions in given interval and data does not agree.")

        self.n_groups = np.prod(self.intervals)

        if self._multiple_data_sets:
            self.data_sets = self.data

        return None

    def plot(self):
        self.initialize()

        if not self._multiple_data_sets:
            self.classify_data()

            width_ratios = np.ones(self.intervals[1] + 1)
            width_ratios[-1] = self.oaxis_size
            height_ratios = np.ones(self.intervals[0] + 1)
            height_ratios[0] = self.oaxis_size

            fig, axes = plt.subplots(
                nrows=self.intervals[0]+1,
                ncols=self.intervals[1]+1,
                gridspec_kw={
                    "wspace": self.xspace,
                    "hspace": self.yspace,
                    "width_ratios": width_ratios,
                    "height_ratios": height_ratios,
                },
                figsize=self.figsize,
                constrained_layout=self.constrained_layout
            )

            for pos, axis in np.ndenumerate(axes):
                r, c = pos
                if r == 0 and c == self.intervals[1]:
                    fig.delaxes(axis)
                # horizontal outer axis
                elif r == 0 and c != self.intervals[1]:
                    # handle limits
                    axis.set_xlim([self.bounds[3, 0], self.bounds[3, 1]])
                    axis.set_ylim([0, 1])
                    # handle ticks
                    axis.set_yticks([])
                    axis.xaxis.tick_top()
                    if c % 2 == 0:
                        self.oaxis_xticks = np.linspace(
                            self.bounds[3, 0],
                            self.bounds[3, 1],
                            self.oaxis_n_xticks
                        )
                        axis.set_xticks(self.oaxis_xticks)
                        if self.oaxis_xticklabels is None:
                            self.oaxis_xticklabels = [
                                f"{tick:.2f}" for tick in self.oaxis_xticks
                            ]
                        axis.xaxis.set_ticklabels(self.oaxis_xticklabels)
                    else:
                        axis.set_xticks([])
                    # draw bar
                    axis.fill_between(
                        x=[
                            self.group_bins[0, c, 1, 0],
                            self.group_bins[0, c, 1, 1],
                        ],
                        y1=[1, 1],
                        y2=[0, 0],
                        facecolor="gray",
                        alpha=1 - self.oaxis_bar_transparency
                    )
                    # add label
                    if c % 2 == 1:
                        axis.annotate(
                            s=self.oaxis_xlabel,
                            xy=(np.mean(self.bounds[3, :]), 0.5),
                            ha="center",
                            va="center",
                        )
                # vertical outer axis
                elif r != 0 and c == self.intervals[1]:
                    # draw vertical outer axes
                    axis.set_xlim([0, 1])
                    axis.set_ylim([self.bounds[2, 0], self.bounds[2, 1]])
                    # handle ticks
                    axis.set_xticks([])
                    axis.yaxis.tick_right()
                    if r % 2 == 0:
                        if self.oaxis_yticks is None:
                            self.oaxis_yticks = np.linspace(
                                self.bounds[2, 0], self.bounds[2, 1], self.oaxis_n_yticks
                            )
                        axis.set_yticks(self.oaxis_yticks)
                        if self.oaxis_yticklabels is None:
                            self.oaxis_yticklabels = [f"{tick:.2f}"
                                                      for tick in self.oaxis_yticks]
                        axis.yaxis.set_ticklabels(self.oaxis_yticklabels)
                    else:
                        axis.set_yticks([])
                    # draw bar
                    axis.fill_between(
                        x=[0, 1],
                        y1=[
                            self.group_bins[r-1, 0, 0, 1],
                            self.group_bins[r-1, 0, 0, 1],
                        ],
                        y2=[
                            self.group_bins[r-1, 0, 0, 0],
                            self.group_bins[r-1, 0, 0, 0],
                        ],
                        facecolor="gray",
                        alpha=1 - self.oaxis_bar_transparency
                    )
                    # add label
                    if r % 2 == 0:
                        axis.annotate(
                            s=self.oaxis_ylabel,
                            xy=(0.50, np.mean(self.bounds[2, :])),
                            verticalalignment="center",
                            horizontalalignment="center",
                            rotation=270
                        )
                # scatter
                elif r != 0 and c != self.intervals[1]:
                    axis.scatter(
                            self.grouped_data[r-1, c, :, 0],
                            self.grouped_data[r-1, c, :, 1],
                            marker=self.marker,
                            s=self.markersize
                        )
                    axis.set_xlim([
                        self.bounds[0, 0] - np.abs(0.10 * self.bounds[0, 0]),
                        self.bounds[0, 1] + np.abs(0.10 * self.bounds[0, 1]),
                    ])
                    axis.set_ylim([
                        self.bounds[1, 0] - np.abs(0.10 * self.bounds[1, 0]),
                        self.bounds[1, 1] + np.abs(0.10 * self.bounds[1, 0]),
                    ])

                    if c % 2 == 0 and r == self.intervals[0]:
                        if self.xticks is None:
                            self.xticks = np.linspace(
                                self.bounds[0, 0], self.bounds[0, 1], self.n_xticks
                            )
                        axis.set_xticks(self.xticks)
                        if self.xticklabels is None:
                            self.xticklabels = [f"{ticks:.2f}" for ticks in self.xticks]
                        axis.xaxis.set_ticklabels(self.xticklabels)
                    else:
                        axis.set_xticks([])

                    if r % 2 == 0 and c == 0:
                        if self.yticks is None:
                            self.yticks = np.linspace(
                                self.bounds[1, 0], self.bounds[1, 1], self.n_yticks
                            )
                        axis.set_yticks(self.yticks)
                        if self.yticklabels is None:
                            self.yticklabels = [f"{ticks:.2f}" for ticks in self.yticks]
                        axis.yaxis.set_ticklabels(self.yticklabels)
                    else:
                        axis.set_yticks([])

                    if c % 2 == 1 and r == self.intervals[0]:
                        axis.set_xlabel(self.xlabel)
                    if r % 2 == 1 and c == 0:
                        axis.set_ylabel(self.ylabel)

            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()

            plt.show()
        else:
            for d_set in self.data_sets:
                self.data = d_set
                self.plot()

        return None

    def get_bounds(self):
        self.bounds = np.array(
            [np.nanmin(self.data, axis=0), np.nanmax(self.data, axis=0)]
        ).T

    def get_bins(self):
        self.bins = []
        for d, bound in enumerate(self.bounds):
            if d > 1:
                self.bins.append(np.linspace(bound[0], bound[1], self.intervals[d-2]+1))
        self.group_bins = np.empty(shape=(self.intervals[0], self.intervals[1], 2, 2))
        for r in range(self.intervals[0]):
            for c in range(self.intervals[1]):
                self.group_bins[r, c, :, :] = np.array([
                    [self.bins[0][r], self.bins[0][r+1]],
                    [self.bins[1][c], self.bins[1][c+1]]
                ])
        self.group_bins = np.flip(self.group_bins, axis=0)

    def classify_data(self):
        self.get_bounds()
        self.get_bins()
        self.grouped_data = np.full(shape=(
            self.intervals[0],
            self.intervals[1],
            self.data.shape[0],
            self.data.shape[1]
        ), fill_value=np.nan)
        for r in range(self.intervals[0]):
            for c in range(self.intervals[1]):
                for p, datum in enumerate(self.data):
                    check1 = datum[2] >= self.group_bins[r, c, 0, 0]
                    check2 = datum[2] <= self.group_bins[r, c, 0, 1]
                    check3 = datum[3] >= self.group_bins[r, c, 1, 0]
                    check4 = datum[3] <= self.group_bins[r, c, 1, 1]
                    if np.all([check1, check2, check3, check4]):
                        self.grouped_data[r, c, p, :] = datum
        return self.grouped_data

    def add_data(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data = [self.data, data]


if __name__ == "__main__":
    plotter1 = TrellisPlotter()

    plotter1.data = np.random.normal(0, 1, size=(1000, 4))

    # data = np.full(shape=(2, 2, 2, 2, 4), fill_value=0)
    # for i1, a1 in enumerate([-1, 1]):
    #     for i2, a2 in enumerate([-1, 1]):
    #         for i3, a3 in enumerate([-1, 1]):
    #             for i4, a4 in enumerate([-1, 1]):
    #                 data[i1, i2, i3, i4] = [a1, a2, a3, a4]
    # data = np.reshape(data, newshape=(16, 4))
    # plotter1.add_data(data)

    plotter1.intervals = np.array([3, 3])

    # plotter1.xlabel = "Control 1"
    # plotter1.ylabel = "Control 2"
    # plotter1.oaxis_xlabel = "Control 3"
    # plotter1.oaxis_ylabel = "Control 4"

    plotter1.plot()
