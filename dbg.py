
from termcolor import colored as termcolor

def p(debug_print, debug_from=None):
	"""Prints text to terminal for debugging."""

	if debug_from is None:
		print_text = termcolor("[DEBUG]\t\t", 'blue') + debug_print
	else:
		print_text = termcolor("[DEBUG]\t\t", 'blue') + termcolor(debug_from + ":\t", 'green') + debug_print

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
	if debug_from is None:
		print_text = termcolor("[DEBUG]\t\t", 'blue') + debug_print
	else:
		print_text = termcolor("[DEBUG]\t\t", 'blue') + termcolor(debug_from + ":\t", 'green') + debug_print
	
	if not end:
		print('\r' + print_text, end='', flush=True)
	else:
		print('\r' + print_text,  flush=True)
	
	# sys.stdout.write('\r' + print_text)
	# sys.stdout.flush()
