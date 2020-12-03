from termcolor import colored as termcolor
import cProfile, pstats, io

RUNTIME_CONSOLE = True


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def prepare_text(debug_print, debug_from=None):
    if debug_from is None:
        if RUNTIME_CONSOLE:
            prepared_text = f"[DEBUG]\t\t{debug_print}"
        else:
            prepared_text = termcolor("[DEBUG]\t\t", 'blue') + debug_print
    else:
        if RUNTIME_CONSOLE:
            prepared_text = f"[DEBUG]\t\t{debug_print}"
        else:
            prepared_text = termcolor("[DEBUG]\t\t", 'blue') \
                            + termcolor(debug_from + ":\t", 'green') + debug_print
    return prepared_text

def p(debug_print, debug_from=None):
    """Prints text to terminal for debugging."""

    print_text = prepare_text(debug_print, debug_from)

    print(print_text)


def u(debug_print: str, debug_from: str = None, end: bool = False):
    """
    Updates the colourised string of the terminal with debug_print. If end is True the line is ended
	Parameters
	----------
	debug_print: str
		Test to be used updated the terminal with.
	debug_from: str, optional
		To specify where debug text is being called from.
	end: str, optional
		If true the line is ended.
	"""

    print_text = prepare_text(debug_from)

    if not end:
        print('\r' + print_text, end='', flush=True)
    else:
        print('\r' + print_text, flush=True)

# sys.stdout.write('\r' + print_text)
# sys.stdout.flush()
