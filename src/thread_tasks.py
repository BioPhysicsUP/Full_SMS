
# from __future__ import annotations
# from typing import TYPE_CHECKING
import os

import numpy as np

import multiprocessing as mp
from processes import ProgressTracker, ProcessProgressTask as PProgTask,\
    ProcessSigPassTask as PSigTask, ProcessProgressCmd as PProgCmd, ProcessProgress, \
    ProcessProgFeedback, PassSigFeedback
from signals import WorkerSigPassType as WSType  #, PassSigFeedback
import multiprocessing as mp
import smsh5
from my_logger import setup_logger
from tree_model import DatasetTreeNode

# if TYPE_CHECKING:

logger = setup_logger(__name__)


class OpenFile:
    """ A QRunnable class to create a worker thread for opening h5 file. """

    def __init__(self, file_path: str,
                 is_irf: bool = False,
                 tmin=None,
                 progress_tracker: ProgressTracker = None):
        """
        Initiate Open File Worker

        Creates a QRunnable object (worker) to be run by a QThreadPool thread.
        This worker is intended to call the given function to open a h5 file
        and populate the tree in the mainwindow g

        Parameters
        ----------
        fname : str
            The name of the file.
        is_irf : bool
            Whether the thread is loading an IRF or not.
        """
        self._file_path = file_path
        self._is_irf = is_irf
        self._tmin = tmin
        self.progress_tracker = progress_tracker

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

    def open_h5(self, feedback_queue: mp.JoinableQueue) -> smsh5.H5dataset:
        """
        Read the selected h5 file and populates the tree on the gui with the file and the particles.

        Accepts a function that will be used to indicate the current progress.

        As this function is designed to be called from a thread other than the main one, no GUI code
        should be called here.

        Parameters
        ----------
        fname : str
            Path name to h5 file.
        feedback_queue : multiprocessing.JoinableQueue
            Queue to send feedback to ProcessThread
        """
        try:
            sig_fb = PassSigFeedback(feedback_queue=feedback_queue)
            prog_fb = ProcessProgFeedback(feedback_queue=feedback_queue)

            dataset = self.load_data(fname=self.file_path, sig_fb=sig_fb, prog_fb=prog_fb)

            datasetnode = DatasetTreeNode(self.file_path[0][self.file_path[0].rfind('/') + 1:-3],
                                          dataset, 'dataset')

            sig_fb.add_datasetnode(node=datasetnode)
            prog_fb.set_status(status="Adding particles...")
            # prog_fb.start(max_value=dataset.numpart)

            all_particles = list()
            for i, particle in enumerate(dataset.particles):
                particlenode = DatasetTreeNode(particle.name, particle, 'particle')
                all_particles.append((particlenode, i))
            sig_fb.add_all_particlenodes(all_nodes=all_particles)

            sig_fb.reset_tree()

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
            sig_fb.set_start(start=av_start)
            global_tmin = np.min(tmins)
            for particle in dataset.particles:
                particle.tmin = global_tmin
            sig_fb.set_tmin(tmin=global_tmin)
            sig_fb.data_loaded()
            prog_fb.set_status(status="Done")

        except Exception as err:
            logger.error(err, exc_info=True)
            raise err

    def open_irf(self, feedback_queue: mp.JoinableQueue) -> None:
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
            sig_fb = PassSigFeedback(feedback_queue=feedback_queue)
            prog_fb = ProcessProgFeedback(feedback_queue=feedback_queue)

            dataset = self.load_data(fname=self.file_path, sig_fb=sig_fb, prog_fb=prog_fb)

            for particle in dataset.particles:
                particle.tmin = self.tmin
                # particle.tmin = np.min(particle.histogram.microtimes)

            irfhist = dataset.particles[0].histogram
            # irfhist.t -= irfhist.t.min()
            sig_fb.add_irf(irfhist.decay, irfhist.t, dataset)
            prog_fb.set_status(status="Done")

            # start_progress_sig.emit(100)
            # status_sig.emit("Done")
        except Exception as err:
            self.signals.error.emit(err)

    def load_data(self, fname:str, sig_fb: PassSigFeedback, prog_fb: ProcessProgFeedback):

        dataset = smsh5.H5dataset(fname[0], sig_fb=sig_fb, prog_fb=prog_fb)
        bin_all(dataset=dataset, bin_size=100, for_irf=self.irf, sig_fb=sig_fb,
                prog_fb=prog_fb)
        dataset.makehistograms()
        return dataset


def bin_all(dataset: smsh5.H5dataset,
            bin_size: float,
            for_irf: bool = False,
            sig_fb: PassSigFeedback = None,
            prog_fb: ProcessProgFeedback = None) -> None:
    """

    Parameters
    ----------
    bin_size
    dataset
    sig_fb
    prog_fb
    """

    if prog_fb:
        prog_fb.start(max_value=0)
        if not for_irf:
            prog_fb.set_status(status="Binning traces...")
        else:
            prog_fb.set_status(status="Binning IRF trace...")
    dataset.bin_all_ints(bin_size, sig_fb=sig_fb, prog_fb=prog_fb)
    if sig_fb:
        sig_fb.bin_size(bin_size=int(bin_size))
