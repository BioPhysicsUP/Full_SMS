import tkinter as tk
from tkinter import ttk
import h5py
import os
from tkinter import filedialog
import matplotlib as mpl
from smsh5 import *
from tcspcfit import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends._backend_tk import ToolTip
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np


class VerticalToolbar(NavigationToolbar2Tk):
    """Subclass of builtin matplotlib toolbar (which is horizontal by default)"""
    def __init__(self, canvas, window):
        NavigationToolbar2Tk.__init__(self, canvas, window)

    def _Button(self, text, file, command, extension='.gif'):
        img_file = os.path.join(
            mpl.rcParams['datapath'], 'images', file + extension)
        im = tk.PhotoImage(master=self, file=img_file)
        b = tk.Button(master=self, text=text, padx=2, pady=2, image=im, command=command)
        b._ntimage = im
        b.pack(side=tk.TOP)
        return b

    def _Spacer(self):
        # Buttons are 30px high, so make this 26px tall with padding to center it
        s = tk.Frame(master=self, width=26, relief=tk.RIDGE, pady=2, bg="DarkGray")
        s.pack(side=tk.TOP, padx=5)
        return s

    def _init_toolbar(self):
        xmin, xmax = self.canvas.figure.bbox.intervalx
        height, width = 50, xmax-xmin
        tk.Frame.__init__(self, master=self.window,
                          width=int(width), height=int(height),
                          borderwidth=2)

        self.update()  # Make axes menu

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                # Add a spacer; return value is unused.
                self._Spacer()
            else:
                button = self._Button(text=text, file=image_file,
                                      command=getattr(self, callback))
                if tooltip_text is not None:
                    ToolTip.createToolTip(button, tooltip_text)

        self.message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, textvariable=self.message)
        self._message_label.pack(side=tk.RIGHT)
        self.pack(side=tk.BOTTOM, fill=tk.X)


class Browser(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent

        self.panes = tk.PanedWindow(self, orient='horizontal')
        self.treeframe = tk.Frame(self.panes)
        self.info = tk.Frame(self.panes)
        self.tree1 = ttk.Treeview(self.treeframe)
        self.infolabel = tk.Label(self.info, text='Hello World')
        self.rasterscan = tk.Label(self.info, text='RS goes here')

        self.tree1.insert('', 'end', 'widgets', text='All files')

        self.panes.add(self.treeframe)
        self.panes.add(self.info)

        self.panes.pack(side='left', fill='both', expand=True)
        # self.treeframe.pack(side='left', fill='both', expand=True)
        # self.info.pack(side='left', fill='both', expand=True)
        self.tree1.pack(side='left', fill='both', expand=True)
        self.infolabel.pack(side='top', fill='both', expand=True)
        self.rasterscan.pack(side='top', fill='both', expand=True)

        self.tree1.bind('<<TreeviewSelect>>', self.parent.plot_trace)

        self.particles = []

    def addparticles(self):

        def sortnumeric(particlename):
            return int(particlename[8:])

        particle_list = []
        for particle in self.parent.meas_file.keys():
            particle_list.append(particle)

        particle_list.sort(key=sortnumeric)

        for index, particle in enumerate(particle_list):
            self.tree1.insert('widgets', 'end', text=particle, iid=index)
            self.particles.append(Particle(self.parent.meas_file, index + 1, self.parent.irf, self.parent.tmin,
                                           self.parent.tmax, self.parent.channelwidth))
            self.particles[-1].makehistogram()


class Plot(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.fig1 = Figure(figsize=(15, 4), dpi=50)
        self.ax1 = self.fig1.add_subplot(111)
        self.controlframe = tk.Frame(self)
        self.plotframe = tk.Frame(self)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.plotframe)  # A tk.DrawingArea.
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side='top', fill='both', expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas1, self.controlframe)
        self.toolbar.update()
        self.canvas1.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.toolbar.pack(side='left')

        self.fig1.tight_layout()

        self.canvas1.mpl_connect("key_press_event", lambda event: self.on_key_press(event, self.canvas1))
        self.plotframe.pack(side='top', fill='both', expand=True)
        self.controlframe.pack(side='top', fill='both', expand=True)

    def on_key_press(self, event, canvas):
        print("you pressed {}".format(event.key))
        key_press_handler(event, canvas, self.toolbar)


class Spectrum(Plot):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        Plot.__init__(self, parent, *args, **kwargs)
        self.parent = parent


class Intensity(Plot):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        Plot.__init__(self, parent, *args, **kwargs)
        self.parent = parent


class Lifetime(Plot):
    def __init__(self, parent, *args, **kwargs):
        Plot.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        fitbutton = ttk.Button(self.controlframe, text='Fit Lifetimes', command=self.parent.fit_lifetime)
        fitbutton.pack(side='right', expand=True)


class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.wm_title("Lifetime Fitting")
        self.parent.geometry('1300x800')

        self.browser = Browser(self)

        # Data file selection
        currentdir = os.path.dirname(os.path.abspath(__file__))
        self.decay_filename = filedialog.askopenfilename(initialdir=currentdir, title="Select decay irf_data",
                                                         filetypes=(("HDF5 files","*.h5"), ("all files","*.*")))
        self.irf_filename = filedialog.askopenfilename(initialdir=currentdir, title="Select IRF",
                                                       filetypes=(("HDF5 files","*.h5"), ("all files","*.*")))
        # self.decay_filename = '/home/bertus/PycharmProjects/SMS-Python-port/LHCII2.h5'
        # self.irf_filename = '/home/bertus/PycharmProjects/SMS-Python-port/IRF 680nm.h5'

        self.meas_file = h5py.File(self.decay_filename, 'r')
        self.irf_file = h5py.File(self.irf_filename, 'r')

        # irf_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/metings/Farooq_intensity/IRF (SLOW APD@680nm).h5', 'r')
        # meas_file = h5py.File('/home/bertus/Documents/Honneurs/Projek/metings/Farooq_intensity/LHCII-PLL(slow APD)-410nW.h5',
        #                       'r')

        irf_data = self.irf_file['Particle 1/Micro Times (s)'][:]

        differences = np.diff(np.sort(irf_data))
        self.channelwidth = np.unique(differences)[1]

        tmin = irf_data.min()
        tmax = irf_data.max()
        window = tmax - tmin
        numpoints = int(window // self.channelwidth)

        t = np.linspace(0, window, numpoints)
        irf_data -= irf_data.min()

        irf, t = np.histogram(irf_data, bins=t)

        self.irf = irf[:-20]  # This is due to some bug in the setup software putting a bunch of very long times in at the end
        self.t = t[:-21]
        self.tmin = tmin
        self.tmax = tmax

        self.lifetime = Lifetime(self)
        self.intensity = Intensity(self)
        self.spectrum = Spectrum(self)

        self.plotframe = tk.Frame(self)

        self.browser.pack(side='left', fill='both', expand=True)
        self.plotframe.pack(side='left', fill='both', expand=True)

        self.lifetime.pack(side='top', fill='both', expand=True)
        self.intensity.pack(side='top', fill='both', expand=True)
        self.spectrum.pack(side='top', fill='both', expand=True)

        self.browser.addparticles()

    def fit_lifetime(self):
        idxs = self.browser.tree1.selection()
        try:
            particle = self.browser.particles[int(idxs[0])]
        except ValueError:
            return
        decay = particle.measured[:-20]
        fit = TwoExp(self.irf, decay, self.t, self.channelwidth, tau=[2.52, 0.336], ploton=True)

    def plot_trace(self, *args):

        self.plot_decay()

        idxs = self.browser.tree1.selection()
        try:
            particle = self.browser.particles[int(idxs[0])]
        except ValueError:
            return

        data = particle.abstimes

        binsize = 10 * 1000000
        endbin = np.int(np.max(data) / binsize)

        binned = np.zeros(endbin)
        for step in range(endbin):
            binned[step] = np.size(data[((step+1)*binsize > data) * (data > step*binsize)])

        binned *= (1000 / 10)

        self.intensity.ax1.clear()
        self.intensity.ax1.plot(binned[:-10])

        self.intensity.canvas1.draw()
        self.intensity.canvas1.get_tk_widget().pack(side='left', fill='both', expand=1)
        self.intensity.toolbar.update()
        self.intensity.canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)

    def plot_decay(self, *args):

        idxs = self.browser.tree1.selection()
        print(idxs)
        try:
            particle = self.browser.particles[int(idxs[0])]
        except ValueError:
            return

        # data = meas_file[particle + '/Micro Times (s)'][:]
        # decay, t = np.histogram(data, bins=1000)
        # t = t[2:]
        # decay = decay[:-1]  # TODO: this should not be hard coded as it is specific to the current irf_data

        self.lifetime.ax1.clear()
        self.lifetime.ax1.semilogy(particle.t[:-21], particle.measured[:-20])
        self.lifetime.ax1.semilogy(self.t, self.irf)

        self.lifetime.canvas1.draw()
        self.lifetime.canvas1.get_tk_widget().pack(side='left', fill='both', expand=1)
        self.lifetime.toolbar.update()
        self.lifetime.canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)


if __name__ == "__main__":
    root = tk.Tk()
    MainApp(root).pack(side="top", fill="both", expand=True)
    for child in root.winfo_children():
        child.pack_configure(padx=10, pady=10, ipadx=10, ipady=10)
    root.mainloop()
