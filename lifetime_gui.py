import tkinter as tk
from tkinter import ttk
import h5py
import os
from tkinter import filedialog
import matplotlib as mpl
from tcspcfit import *

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np


def fit_lifetime():
    channelwidth = max(t) / np.size(irf)
    fit = TwoExp(irf, decay, t, channelwidth, tau=[2.52, 0.336], startpoint=np.argmax(irf), endpoint=np.size(decay), ploton=True)

root = tk.Tk()
root.wm_title("Lifetime Fitting")
root.geometry('1000x500')

panes = tk.PanedWindow(orient='horizontal')
panes.pack(fill='both', expand=True)

treeframe = tk.Frame(panes)
panes.add(treeframe)

notebookframe = tk.Frame(panes)
panes.add(notebookframe)

tree1 = ttk.Treeview(treeframe)
tree1.pack(side='top', fill='both', expand=True)
tree1.insert('', 'end', 'widgets', text='All files')

notebook = ttk.Notebook(notebookframe)
notebook.pack(side='top', fill='both', expand=True)

lifetime_frame = tk.Frame(notebookframe)
lifetime_frame.pack(side='top', fill='both', expand=True)

fitbutton = ttk.Button(text='Fit Lifetime', command=fit_lifetime)
fitbutton.pack(side='top', fill='both', expand=True)

resolve_frame = tk.Frame(notebookframe)
resolve_frame.pack(side='top', fill='both', expand=True)

notebook.add(lifetime_frame, text='Lifetime Fitting')
notebook.add(resolve_frame, text='Resolve Levels')

# Data file selection
currentdir = os.path.dirname(os.path.abspath(__file__))
# root.decay_filename = filedialog.askopenfilename(initialdir=currentdir, title="Select decay irf_data",
#                                                  filetypes=(("HDF5 files","*.h5"), ("all files","*.*")))
# root.irf_filename = filedialog.askopenfilename(initialdir=currentdir, title="Select IRF",
#                                                filetypes=(("HDF5 files","*.h5"), ("all files","*.*")))
root.decay_filename = '/home/bertus/PycharmProjects/SMS-Python-port/LHCII2.h5'
root.irf_filename = '/home/bertus/PycharmProjects/SMS-Python-port/IRF 680nm.h5'

f = h5py.File(root.decay_filename, 'r')
irf_file = h5py.File(root.irf_filename, 'r')
irfdata = irf_file['Particle 1/Micro Times (s)'][:]
irf, t = np.histogram(irfdata, bins=1000)


# Sort particles and add to tree
def sortnumeric(particle):
    return int(particle[8:])


particles = []
for particle in f.keys():
    particles.append(particle)

particles.sort(key=sortnumeric)

for index, particle in enumerate(particles):
    tree1.insert('widgets', 'end', text=particle, iid=index)

# Intensity trace figure setup
fig1 = Figure(figsize=(15, 4), dpi=100)
ax1 = fig1.add_subplot(111)
canvas1 = FigureCanvasTkAgg(fig1, master=resolve_frame)  # A tk.DrawingArea.
canvas1.draw()
canvas1.get_tk_widget().pack(side='left', fill='both', expand=1)

toolbar = NavigationToolbar2Tk(canvas1, resolve_frame)
toolbar.update()
canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)

fig1.tight_layout()

# Lifetime figure setup
fig2 = Figure(figsize=(15, 4), dpi=100)
ax2 = fig2.add_subplot(111)
canvas2 = FigureCanvasTkAgg(fig2, master=lifetime_frame)  # A tk.DrawingArea.
canvas2.draw()
canvas2.get_tk_widget().pack(side='left', fill='both', expand=1)

toolbar = NavigationToolbar2Tk(canvas2, lifetime_frame)
toolbar.update()
canvas2.get_tk_widget().pack(side='top', fill='both', expand=1)

fig2.tight_layout()

# Sort out interactive navigation
def on_key_press(event, canvas):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas, toolbar)


canvas1.mpl_connect("key_press_event", lambda event: on_key_press(event, canvas1))
canvas2.mpl_connect("key_press_event", lambda event: on_key_press(event, canvas2))


def plot_trace(*args):

    plot_decay()

    idxs = tree1.selection()
    print(idxs)
    try:
        particle = particles[int(idxs[0])]
    except ValueError:
        return

    data = f[particle + '/Absolute Times (ns)'][:]

    binsize = 10 * 1000000
    endbin = np.int(np.max(data) / binsize)

    binned = np.zeros(endbin)
    for step in range(endbin):
        binned[step] = np.size(data[((step+1)*binsize > data) * (data > step*binsize)])

    binned *= (1000 / 10)

    ax1.clear()
    ax1.plot(binned)

    canvas1.draw()
    canvas1.get_tk_widget().pack(side='left', fill='both', expand=1)
    toolbar.update()
    canvas1.get_tk_widget().pack(side='top', fill='both', expand=1)


def plot_decay(*args):

    global decay

    idxs = tree1.selection()
    print(idxs)
    try:
        particle = particles[int(idxs[0])]
    except ValueError:
        return

    data = f[particle + '/Micro Times (s)'][:]
    decay, t = np.histogram(data, bins=1000)
    t = t[2:]
    decay = decay[:-1]  # TODO: this should not be hard coded as it is specific to the current irf_data

    ax2.clear()
    ax2.semilogy(t, decay)
    ax2.semilogy(t, irf[:-1])

    canvas2.draw()
    canvas2.get_tk_widget().pack(side='left', fill='both', expand=1)
    toolbar.update()
    canvas2.get_tk_widget().pack(side='top', fill='both', expand=1)


tree1.bind('<<TreeviewSelect>>', plot_trace)

for child in root.winfo_children():
    child.pack_configure(padx=10, pady=10)

tk.mainloop()
