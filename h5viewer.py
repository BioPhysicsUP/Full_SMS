import tkinter as tk
import h5py
import os
from tkinter import filedialog
import matplotlib as mpl

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np

root = tk.Tk()
root.wm_title("Embedding in Tk")

# currentdir = os.path.dirname(os.path.abspath(__file__))
# root.filename = filedialog.askopenfilename(initialdir=currentdir, title="Select file",
#                                            filetypes=(("HDF5 files","*.h5"), ("all files","*.*")))
root.filename = '/home/bertus/PycharmProjects/SMS-Python-port/LHCII.h5'
f = h5py.File(root.filename, 'r')

particles = []
for particle in f.keys():
    particles.append(particle)

listbox = tk.Listbox(root)
listbox.pack(side='left', fill='both')

for particle in particles:
    listbox.insert('end', particle)

fig = Figure(figsize=(15, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side='left', fill='both', expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side='top', fill='both', expand=1)


def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas, toolbar)


canvas.mpl_connect("key_press_event", on_key_press)

fig.tight_layout()


def plot_trace(*args):

    idxs = listbox.curselection()
    particle = particles[int(idxs[0])]

    data = f[particle + '/Absolute Times (ns)'][:]

    binsize = 10 * 1000000
    endbin = np.int(np.max(data) / binsize)

    binned = np.zeros(endbin)
    for step in range(endbin):
        binned[step] = np.size(data[((step+1)*binsize > data) * (data > step*binsize)])

    ax.clear()
    ax.plot(binned)

    canvas.draw()
    canvas.get_tk_widget().pack(side='left', fill='both', expand=1)
    toolbar.update()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)



listbox.bind('<<ListboxSelect>>', plot_trace)

for child in root.winfo_children():
    child.pack_configure(padx=10, pady=10)

tk.mainloop()
