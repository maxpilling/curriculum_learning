import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import fnmatch
import os
import re
import sys
import glob
import numpy as np
import pandas as pd

DIGITS = r"\d+"
TARGET_STRING = "Episode * ended. Score *"


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = "Average of epochs"
        self.width = 1200
        self.height = 900
        self.files = None  # Used to store the CSV files containing epoch data
        self.data = None  # The data to plot
        self.graph = None  # A reference to the graph widget to use
        self.sliding_window_value = (
            5
        )  # Betweeen 0-100; the sliding window value used to smooth the graph
        self.init_ui()

    def init_ui(self):
        # Set the dimensions of the application
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Add a menu bar
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        # Add menu bar options
        open_action = QAction("Open CSV Files", self)
        open_action.triggered.connect(self.open_files)
        open_action.setStatusTip("Open log files to plot")

        save_action = QAction("Save Graph as JPEG", self)
        save_action.triggered.connect(self.save_graph_image)
        save_action.setStatusTip("Save the current graph as a JPEG")

        menu_bar.addAction(open_action)
        menu_bar.addAction(save_action)

        # Create toolbar
        toolbar = self.addToolBar("SliderBar")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setTickInterval(5)
        slider.valueChanged.connect(self.slider_value_changed)
        self.slider = slider
        toolbar.addWidget(slider)

        # Add the graph widget
        graph = PlotCanvas(self, width=12, height=9)
        self.setCentralWidget(graph)
        self.graph = graph

        self.show()

    def slider_value_changed(self):
        self.sliding_window_value = self.slider.value()
        print("Sliding window value now:", self.sliding_window_value)
        self.graph.plot()

    def calculate_and_plot(self):
        # TODO: Break this down into smaller functions
        """
            Calculate the scores and plot them
        """
        self.data = None
        print("Calculating mean and plotting...")
        if self.files == None or self.files == []:
            dialog = QMessageBox(self)
            dialog.setText("No CSVs Loaded!")
            dialog.setModal(True)
            dialog.resize(500, 300)
            dialog.show()

        # Reset so if this has already been called we can read the file again
        for file in self.files:
            file.seek(0)
        scores = self.get_scores_per_file()
        shortened_runs = []
        lowest_size_index = None
        # get the epoch with the lowest number of episodes
        for epoch in scores:
            if lowest_size_index == None:
                lowest_size_index = len(epoch) - 1
            elif len(epoch) < lowest_size_index:
                lowest_size_index = len(epoch) - 1

        # Get the list of scores for each epoch as the same size
        for epoch in scores:
            if len(epoch) > lowest_size_index:
                elements_to_shorten = len(epoch) - lowest_size_index
                new_epoch = epoch[: len(epoch) - elements_to_shorten]
                shortened_runs.append(new_epoch)

        # calculate the mean for each episode
        current_episode = 0
        averaged_scores = []
        while current_episode < lowest_size_index:
            score = 0
            for epoch in shortened_runs:
                score += epoch[current_episode]
            score = score / len(shortened_runs)

            current_episode = current_episode + 1
            averaged_scores.append(score)
        self.data = averaged_scores
        self.graph.plot()

    def save_graph_image(self):
        """
            Saves the current graph as a JPEG file
        """
        self.graph.save_graph()

    def open_files(self):
        """
            Open a list of files and sets a local file object for each.
        """
        self.files = None
        self.data = None
        file_paths = QFileDialog.getOpenFileNames(self, "Choose log files", "")
        self.files = [open(path, "r") for path in file_paths[0]]
        self.calculate_and_plot()

    def get_scores_per_file(self):
        """
            Returns an array of arrays, whereby each array inside the
            main array contains all of the scores for each episode in the epoch
        """
        scores_for_files = []
        for file in self.files:
            scores = []
            for line in file.readlines():
                line = line.strip("\n")
                scores.append(int(line.rsplit(",", 1)[1]))
            scores_for_files.append(scores)
        return scores_for_files


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
            Intialise the graph widget and set some of the main params
        """
        self.parent = parent  # a reference to the main window object
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # Set the graph details
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylabel("Game Score")
        self.ax.set_xlabel("Episode")

    def plot(self):
        """
            Smootht the data and then plot it to the graph
        """
        # Clear the plot
        self.axes.clear()
        self.ax.clear()
        self.draw()

        # convert the list of data to a pandas Series and smooth it using the rolling mean
        data = self.parent.data
        data = pd.Series(data)
        smoothed_data = data.rolling(self.parent.sliding_window_value).mean()

        # Plot the smoothed data
        self.ax.plot(smoothed_data, "r-")
        self.draw()

    def save_graph(self):
        path = QFileDialog.getSaveFileName()
        self.fig.savefig(path[0] + "png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
