"""Program for viewing SMS data from HDF5 files

Depends on smsh5 and tcspcfit

Bertus van Heerden
University of Pretoria
2018
"""
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import matplotlib as mpl
from src.smsh5 import *
from src.tcspcfit import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends._backend_tk import ToolTip
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
mpl.rcParams.update({'font.size': 22})


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
        self.descriptlabel = tk.Label(self.info, text='Description:')
        self.infolabel = tk.Label(self.info)
        self.checkframe = tk.Frame(self.info)
        # self.rasterscan = tk.Label(self.info, text='RS goes here')
        self.rasterscan = RasterScan(self.info)

        self.bgvar = tk.IntVar()
        self.ignorevar = tk.IntVar()
        self.bgcheckbox = tk.Checkbutton(self.checkframe, variable=self.bgvar, command=self.parent.bgcb)
        self.bglabel = tk.Label(self.checkframe, text='BG:')
        self.ignorecheckbox = tk.Checkbutton(self.checkframe, variable=self.ignorevar, command=self.parent.ignorecb)
        self.ignorelabel = tk.Label(self.checkframe, text='Ignore:')

        self.bgvaluevar = tk.StringVar()
        self.bgentrylabel = tk.Label(self.checkframe, text='BG value:')
        self.bgentry = tk.Entry(self.checkframe, textvariable=self.bgvaluevar)

        self.subtractbg_but = tk.Button(self.checkframe, command=self.parent.subtractbg_cb, text='Subtract BG')
        self.export_but = tk.Button(self.checkframe, command=self.parent.export_cb, text='Export')

        self.tree1.insert('', 'end', 'widgets', text='All files')

        self.panes.add(self.treeframe)
        self.panes.add(self.info)

        self.panes.pack(side='left', fill='both', expand=True)
        # self.treeframe.pack(side='left', fill='both', expand=True)
        # self.info.pack(side='left', fill='both', expand=True)

        self.tree1.pack(side='left', fill='both', expand=True)
        self.descriptlabel.pack(side='top', fill='both', expand=True)
        self.infolabel.pack(side='top', fill='both', expand=True)
        self.checkframe.pack(side='top', fill='both', expand=True)
        self.rasterscan.pack(side='top', fill='x', expand=True)

        self.bglabel.pack(side='left')
        self.bgcheckbox.pack(side='left')
        self.ignorelabel.pack(side='left')
        self.ignorecheckbox.pack(side='left')

        self.subtractbg_but.pack(side='left')
        self.export_but.pack(side='left')

        self.tree1.bind('<<TreeviewSelect>>', self.parent.particle_sel_cb)

        self.particles = []

    def addparticles(self):

        def sortnumeric(particlename):
            return int(particlename[8:])

        particle_list = []
        for particle in self.parent.meas_file.keys():
            particle_list.append(particle)

        particle_list.sort(key=sortnumeric)

        answer = messagebox.askyesno("Question", "Import particle list?")
        if answer:
            currentdir = os.path.dirname(os.path.abspath(__file__))
            # currentdir = '/home/bertus/Documents/Honneurs/Projek/Metings/Intensiteitsanalise/Exported/Gebruik'
            dir = filedialog.askopenfilename(initialdir=currentdir,
                                             filetypes=(('Text files', '*.txt'),))
            # currentdir = os.path.dirname(os.path.abspath(__file__))
            imported_list = np.genfromtxt(dir, skip_header=2, dtype='str', delimiter='\n')
            print(imported_list)

        index = 0
        for particlename in particle_list:
            print(answer, particlename)
            if not answer or particlename in imported_list:
                try:
                    self.tree1.insert('widgets', 'end', text=particlename, iid=index)
                    self.particles.append(Particle(particlename, self.parent.meas_file, index + 1, self.parent.irf, self.parent.tmin,
                                                   self.parent.tmax, self.parent.channelwidth))
                    self.particles[-1].makehistogram()
                    index += 1
                except KeyError:
                    raise


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
        # self.ax1.set_aspect(0.01)


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


class RasterScan(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.fig1 = Figure(figsize=(7, 7), dpi=50)
        self.ax1 = self.fig1.add_subplot(111)
        self.plotframe = tk.Frame(self)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.plotframe)  # A tk.DrawingArea.
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side='top', fill='both', expand=True)

        self.fig1.tight_layout()

        self.plotframe.pack(side='top', fill='both', expand=True)


class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.parent.wm_title("Lifetime Fitting")
        self.parent.geometry('1300x800')

        self.browser = Browser(self)

        # Data file selection
        currentdir = os.path.dirname(os.path.abspath(__file__))
        # currentdir = '/home/bertus/Documents/Honneurs/Projek/Metings/Intensiteitsanalise'
        self.decay_filename = filedialog.askopenfilename(initialdir=currentdir, title="Select decay data",
                                                         filetypes=(("HDF5 files","*.h5"), ("all files","*.*")))
        # currentdir = '/home/bertus/Documents/Honneurs/Projek/Metings/Random'
        self.irf_filename = filedialog.askopenfilename(initialdir=currentdir, title="Select IRF",
                                                       filetypes=(("HDF5 files","*.h5"), ("all files","*.*")))
        # self.decay_filename = '/home/bertus/PycharmProjects/SMS-Python-port/LHCII2.h5'
        # self.irf_filename = '/home/bertus/PycharmProjects/SMS-Python-port/IRF 680nm.h5'

        self.meas_file = h5py.File(self.decay_filename, 'r')
        self.irf_file = h5py.File(self.irf_filename, 'r')

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

        self.isbinned = False

        self.bgparticles = []

    def fit_lifetime(self):
        decay = self.selected_particle.measured[:-20]
        fit = TwoExp(self.irf, decay, self.t, self.channelwidth, tau=[2.52, 0.336], ploton=True)

    def plot_trace(self, *args):

        if not self.isbinned:
            self.binints()
            self.isbinned = True

        binned = self.selected_particle.binned
        self.intensity.ax1.clear()
        self.intensity.ax1.plot(binned[:-10])

        self.intensity.canvas1.draw()
        self.intensity.canvas1.get_tk_widget().pack(side='left', fill='both', expand=1)
        self.intensity.toolbar.update()
        self.intensity.canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)

    def plot_decay(self, *args):

        # data = meas_file[particle + '/Micro Times (s)'][:]
        # decay, t = np.histogram(data, bins=1000)
        # t = t[2:]
        # decay = decay[:-1]  # TODO: this should not be hard coded as it is specific to the current irf_data

        self.lifetime.ax1.clear()
        self.lifetime.ax1.semilogy(self.selected_particle.t[:-21], self.selected_particle.measured[:-20])
        self.lifetime.ax1.semilogy(self.t, self.irf)

        self.lifetime.canvas1.draw()
        self.lifetime.canvas1.get_tk_widget().pack(side='left', fill='both', expand=1)
        self.lifetime.toolbar.update()
        self.lifetime.canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)

    def plot_spectra(self, *args):

        self.spectrum.ax1.clear()
        times = self.selected_particle.spectratimes
        wavelengths = self.selected_particle.wavelengths
        mintime = np.min(times)
        maxtime = np.max(times)
        minwav = np.min(wavelengths)
        maxwav = np.max(wavelengths)
        self.spectrum.ax1.imshow(self.selected_particle.spectra.T, aspect='auto', interpolation='none',
                                 extent=[mintime, maxtime, minwav, maxwav])
        # self.spectrum.ax1.set_aspect(0.005)

        self.spectrum.canvas1.draw()
        self.spectrum.canvas1.get_tk_widget().pack(side='left', fill='both', expand=1)
        self.spectrum.toolbar.update()
        self.spectrum.canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)

    def plot_rs(self, *args):

        self.browser.rasterscan.ax1.clear()
        try:
            self.browser.rasterscan.ax1.imshow(self.selected_particle.rasterscan)
        except AttributeError:
            self.browser.rasterscan.ax1.text(0.3, 0.5, "Raster Scan not available")

        self.browser.rasterscan.canvas1.draw()
        self.browser.rasterscan.canvas1.get_tk_widget().pack(side='left', fill='both', expand=1)

    def binints(self):
        for particle in self.browser.particles:

            data = particle.abstimes

            binsize = 100 * 1000000
            endbin = np.int(np.max(data) / binsize)

            binned = np.zeros(endbin)
            for step in range(endbin):
                binned[step] = np.size(data[((step+1)*binsize > data) * (data > step*binsize)])

            binned *= (1000 / 100)
            particle.binned = binned

    def bgcb(self):
        self.selected_particle.bg = bool(self.browser.bgvar.get())

    def ignorecb(self):
        self.selected_particle.ignore = bool(self.browser.ignorevar.get())

    def particle_sel_cb(self, *args):
        if self.selected_particle is not None:

            self.browser.ignorevar.set(int(self.selected_particle.ignore))
            self.browser.bgvar.set(int(self.selected_particle.bg))

            self.browser.infolabel.config(text=self.selected_particle.description)

            self.plot_decay()
            self.plot_rs()
            self.plot_trace()
            self.plot_spectra()

    def export_cb(self):

        answer = messagebox.askyesno("Question", "Export to hdf? (Otherwise text)")
        currentdir = os.path.dirname(os.path.abspath(__file__))
        # currentdir = '/home/bertus/Documents/Honneurs/Projek/Metings/Intensiteitsanalise/Exported/Gebruik'

        if not answer:
            expfile = filedialog.asksaveasfilename(initialdir=currentdir, filetypes=(('Text files', '*.txt'), ))
            export_list = ['BG value: {}'.format(self.bgav), "Used particles:"]
            for particle in self.browser.particles:
                if not particle.bg and not particle.ignore:
                    export_list.append(particle.name)

            np.savetxt(expfile, export_list, fmt='%s')
        else:
            expfile = filedialog.asksaveasfilename(initialdir=currentdir, filetypes=(('HDF5 files', '*.h5'), ))
            export_file = h5py.File(expfile, 'a')

            ind = 1
            for item in export_file.items():
                ind += 1

            for particle in self.browser.particles:
                if not particle.bg and not particle.ignore:
                    self.meas_file.copy(particle.name, export_file, 'Particle {}'.format(ind), expand_external=True,
                                        expand_refs=True, expand_soft=True)
                    print(ind, particle.name)
                    ind += 1
            print('Exported!')

    def subtractbg_cb(self):

        for particle in self.browser.particles:
            if particle.bg:
                self.bgparticles.append(particle)

        bgtrace = np.concatenate(tuple(particle.binned for particle in self.bgparticles))
        try:
            self.bgav = np.average(bgtrace)
            self.browser.bgvaluevar.set(self.bgav)
        except ValueError:
            self.bgav = self.browser.bgvaluevar.get()

        print('BG value is {}'.format(self.bgav))

        for particle in self.browser.particles:
            particle.binned = particle.binned - self.bgav
            particle.binned = np.clip(particle.binned, 0, None)
        print('Done with BG subtract!')

    @property
    def selected_particle(self):
        idxs = self.browser.tree1.selection()
        try:
            particle = self.browser.particles[int(idxs[0])]
        except ValueError:
            return None
        return particle


if __name__ == "__main__":
    root = tk.Tk()
    MainApp(root).pack(side="top", fill="both", expand=True)
    for child in root.winfo_children():
        child.pack_configure(padx=10, pady=10, ipadx=10, ipady=10)
    root.mainloop()
