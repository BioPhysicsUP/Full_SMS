from termcolor import colored as termcolor


def p(debug_print, debug_from=None):
	"""Prints text to terminal for debugging."""

	if debug_from is None:
		print_text = termcolor("[DEBUG]\t\t", 'blue') + debug_print
	else:
		print_text = termcolor("[DEBUG]\t\t", 'blue') + termcolor(debug_from + ":\t", 'green') + debug_print

	print(print_text)
