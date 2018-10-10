
from tkinter import *
from tkinter.ttk import *
import matplotlib as mpl
import numpy as np
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg

root = Tk()

# treeframe = Frame(root)
# treeframe.pack(side='left', fill='both', expand=True)
#
# notebookframe = Frame(root)
# notebookframe.pack(side='right', fill='both', expand=True)

panes = PanedWindow(orient='horizontal')
panes.pack(fill='both', expand=True)

treeframe = Frame(panes)
panes.add(treeframe)

notebookframe = Frame(panes)
panes.add(notebookframe)

tree1 = Treeview(treeframe)
tree1.pack(side='top', fill='both', expand=True)
tree1.insert('', 'end', 'widgets', text='All files')
tree1.insert('widgets', 'end', text='trace1')

notebook = Notebook(notebookframe)
notebook.pack(side='top', fill='both', expand=True)

resolve_frame = Frame(notebookframe)
resolve_frame.pack(side='top', fill='both', expand=True)

lifetime_frame = Frame(notebookframe)
lifetime_frame.pack(side='top', fill='both', expand=True)

notebook.add(resolve_frame, text='Resolve Levels')
notebook.add(lifetime_frame, text='Lifetime Fitting')

# testim=PhotoImage(file='fittedlevel.png')
canvas1 = Canvas(resolve_frame, width=400, height=500)
canvas1.pack(side='top', fill='both', expand=True)
# canvas1.create_image(50, 50, image=testim, anchor='nw', tags=('piece'))


def draw_figure(canvas, figure, loc=(0, 0)):
    """ Draw a matplotlib figure onto a Tk canvas

    loc: location of top-left corner of figure on canvas in pixels.
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    """
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo


# Create a canvas

# Generate some example irf_data
X = np.linspace(0, 2 * np.pi, 50)
Y = np.sin(X)

# Create the figure we desire to add to an existing canvas
fig = mpl.figure.Figure(figsize=(6, 3))
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(X, Y)

# Keep this handle alive, or else figure will disappear
fig_x, fig_y = 100, 100
fig_photo = draw_figure(canvas1, fig, loc=(fig_x, fig_y))
fig_w, fig_h = fig_photo.width(), fig_photo.height()

# Add more elements to the canvas, potentially on top of the figure
canvas1.create_line(200, 50, fig_x + fig_w / 2, fig_y + fig_h / 2)
canvas1.create_text(200, 50, text="Zero-crossing", anchor="s")

controlframe = Frame(resolve_frame)
controlframe.pack(side='top', fill='both', expand=True)

binsizelab = Label(controlframe, text='Bin Size (ms):')
binsizelab.pack(side='left', fill='both', expand=True)

binsize = Entry(controlframe)
binsize.pack(side='left', fill='both', expand=True)

applybut = Button(controlframe, text='Apply')
applybut.pack(side='left', fill='both', expand=True)
applyallbut = Button(controlframe, text='Apply to All')
applyallbut.pack(side='left', fill='both', expand=True)

applyconf = Button(controlframe, text='Resolve Levels')
applyconf.pack(side='right', fill='both', expand=True)
confidence = Combobox(controlframe, values=['99%', '95%'])
confidence.pack(side='right', fill='both', expand=True)
conflab = Label(controlframe, text='Confidence Level:')
conflab.pack(side='right', fill='both', expand=True)

for wid in controlframe.winfo_children():
    wid.pack_configure(expand=False, fill='x')


root.mainloop()
