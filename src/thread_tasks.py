import os

import numpy as np

import processes as prcs
import smsh5
from my_logger import setup_logger
from tree_model import DatasetTreeNode

logger = setup_logger(__name__)


class OpenFile:
    """ A QRunnable class to create a worker thread for opening h5 file. """

    def __init__(self, file_path: str, irf: bool = False, tmin=None,
                 progress_tracker: prcs.ProgressTracker = None):
        """
        Initiate Open File Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to open a h5 file
        and populate the tree in the mainwindow g

        Parameters
        ----------
        fname : str
            The name of the file.
        irf : bool
            Whether the thread is loading an IRF or not.
        """
        self._file_path = None
        self._is_irf = None
        self._tmin = None

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: str):
        assert type(file_path) is str, "file_path must be of type str"
        assert os.path.exists(file_path) and os.path.isfile(file_path), \
            'file_path is not a path to a valid file'
        self._file_path = file_path

    @property
    def irf(self) -> bool:
        return self._is_irf

    @irf.setter
    def irf(self, is_irf: bool):
        assert type(is_irf) is bool, 'is_irf is not of type bool'
        self._is_irf = is_irf

    @property
    def tmin(self):
        return self._tmin

    @tmin.setter
    def tmin(self, tmin):
        # assert
        self._tmin = tmin

    def open_h5(self) -> smsh5.H5dataset:
        """
        Read the selected h5 file and populates the tree on the gui with the file and the particles.

        Accepts a function that will be used to indicate the current progress.

        As this function is designed to be called from a thread other than the main one, no GUI code
        should be called here.

        Parameters
        ----------
        fname : str
            Path name to h5 file.
        """
        try:
            dataset = self.load_data(self.file_path)

            datasetnode = DatasetTreeNode(self.file_path[0][self.file_path[0].rfind('/') + 1:-3],
                                          dataset, 'dataset')
            add_dataset_sig.emit(datasetnode)

            start_progress_sig.emit(dataset.numpart)
            status_sig.emit("Opening file: Adding particles...")
            for i, particle in enumerate(dataset.particles):
                particlenode = DatasetTreeNode(particle.name, particle, 'particle')
                add_node_sig.emit(particlenode, progress_sig, i)
                progress_sig.emit()
            reset_tree_sig.emit()

            starttimes = []
            tmins = []
            for particle in dataset.particles:
                # Find max, then search backward for first zero to find the best startpoint
                decay = particle.histogram.decay
                histmax_ind = np.argmax(decay)
                reverse = decay[:histmax_ind][::-1]
                zeros_rev = np.where(reverse == 0)[0]
                if len(zeros_rev) != 0:
                    length = 0
                    start_ind_rev = zeros_rev[0]
                    for i, val in enumerate(zeros_rev[:-1]):
                        if zeros_rev[i + 1] - val > 1:
                            length = 0
                            continue
                        length += 1
                        if length >= 10:
                            start_ind_rev = val
                            break
                    start_ind = histmax_ind - start_ind_rev
                    # starttime = particle.histogram.t[start_ind]
                    starttime = start_ind
                else:
                    starttime = 0
                starttimes.append(starttime)

                tmin = np.min(particle.histogram.microtimes)
                tmins.append(tmin)

            av_start = np.average(starttimes)
            set_start_sig.emit(av_start)

            global_tmin = np.min(tmins)
            for particle in dataset.particles:
                particle.tmin = global_tmin

            set_tmin_sig.emit(global_tmin)

            status_sig.emit("Done")
            data_loaded_sig.emit()
        except Exception as err:
            logger.error(err, exc_info=True)
            raise err

    def open_irf(self, fname, tmin) -> None:
        """
        Read the selected h5 file and populates the tree on the gui with the file and the particles.

        Accepts a function that will be used to indicate the current progress.

        As this function is designed to be called from a thread other than the main one, no GUI code
        should be called here.

        Parameters
        ----------
        fname : str
            Path name to h5 file.
        """

        start_progress_sig = self.signals.start_progress
        status_sig = self.signals.status_message
        add_irf_sig = self.signals.add_irf

        try:
            dataset = self.load_data(fname)

            for particle in dataset.particles:
                particle.tmin = tmin
                # particle.tmin = np.min(particle.histogram.microtimes)
            irfhist = dataset.particles[0].histogram
            # irfhist.t -= irfhist.t.min()
            add_irf_sig.emit(irfhist.decay, irfhist.t, dataset)

            start_progress_sig.emit(dataset.numpart)
            status_sig.emit("Done")
        except Exception as err:
            self.signals.error.emit(err)

    def load_data(self, fname):

        auto_prog_sig = self.signals.auto_progress
        bin_size_sig = self.signals.bin_size
        progress_sig = self.signals.progress
        start_progress_sig = self.signals.start_progress

        status_sig = self.signals.status_message

        status_sig.emit("Opening file...")
        dataset = smsh5.H5dataset(fname[0])  # , progress_sig, auto_prog_sig)
        bin_all(dataset, 100, start_progress_sig, progress_sig, status_sig, bin_size_sig)
        start_progress_sig.emit(dataset.numpart)
        status_sig.emit("Opening file: Building decay histograms...")
        dataset.makehistograms()
        return dataset


def bin_all(dataset, bin_size, start_progress_sig, progress_sig, status_sig, bin_size_sig) -> None:
    """

    Parameters
    ----------
    bin_size
    dataset
    start_progress_sig
    progress_sig
    status_sig
    """

    start_progress_sig.emit(dataset.numpart)
    # if not self.data_loaded:
    #     part = "Opening file: "
    # else:
    #     part = ""
    # status_sig.emit(part + "Binning traces...")
    status_sig.emit("Binning traces...")
    dataset.bin_all_ints(bin_size)
    bin_size_sig.emit(bin_size)
    status_sig.emit("Done")