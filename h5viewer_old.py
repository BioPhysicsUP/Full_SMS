import matplotlib as mpl
import numpy as np
import tkinter as tk
from tkinter import filedialog
import h5py
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

mpl.rc('text', usetex=True)


def draw_figure(canvas, figure, loc=(0, 0)):
    """ Draw a matplotlib figure onto a Tk canvas

    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo


# Create a canvas
w, h = 1000, 500
root = tk.Tk()
root.title("A figure in a canvas")

currentdir = os.path.dirname(os.path.abspath(__file__))
root.filename = filedialog.askopenfilename(initialdir=currentdir, title="Select file",
                                           filetypes=(("HDF5 files","*.h5"), ("all files","*.*")))

f = h5py.File(root.filename, 'r')
data = f['Particle 1/Absolute Times (ns)'][:]

binsize = 10 * 1000000
endbin = np.int(np.max(data) / binsize)

binned = np.zeros(endbin)
for step in range(endbin):
    binned[step] = np.size(data[((step+1)*binsize > data) * (data > step*binsize)])

frame = tk.Frame(root, width=w, height=h)
frame.grid(row=0,column=0)

canvas = tk.Canvas(frame, width=w, height=h, scrollregion=(0, 0, 2000, 500))

hbar = tk.Scrollbar(frame,orient='horizontal')
hbar.pack(side='bottom',fill='x')
hbar.config(command=canvas.xview)

canvas.config(xscrollcommand=hbar.set)
canvas.pack()

# Create the figure we desire to add to an existing canvas
fig = mpl.figure.Figure(figsize=(40, h / 100))
print(fig.bbox.bounds)
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(binned)

# Keep this handle alive, or else figure will disappear
fig_x, fig_y = 0, 0
fig_photo = draw_figure(canvas, fig, loc=(fig_x, fig_y))
fig_w, fig_h = fig_photo.width(), fig_photo.height()

# Add more elements to the canvas, potentially on top of the figure
# canvas.create_line(200, 50, fig_x + fig_w / 2, fig_y + fig_h / 2)
# canvas.create_text(200, 50, text="Zero-crossing", anchor="s")

# Let Tk take over
tk.mainloop()

