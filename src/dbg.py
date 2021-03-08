import cProfile
import io
import pstats
import enum
from typing import Union

try:
    from pympler import asizeof
    has_asizeof = True
except ModuleNotFoundError:
    has_asizeof = False

from termcolor import colored as termcolor

RUNTIME_CONSOLE = True


class MemSizeType(enum.Enum):
    byte = enum.auto()
    bit = enum.auto()

    kilobyte = enum.auto()
    kilobit = enum.auto()

    megabyte = enum.auto()
    megabit = enum.auto()

    gigabyte = enum.auto()
    gigabit = enum.auto()


class MemSize(float):

    def __new__(self, value, size_type):
        return float.__new__(self, value)

    def __init__(self, value: Union[int, float],
                 size_type: Union[str, MemSizeType] = MemSizeType.megabyte):

        float.__init__(value)

        if type(size_type) is str:
            size_type = self.str_2_size_type(size_type)
        self.size_type = size_type

    @staticmethod
    def str_2_size_type(type_str: str):
        if type_str == 'B':
            return MemSizeType.byte
        elif type_str == 'b':
            return MemSizeType.bit
        elif type_str == 'kB' or type_str == 'KB':
            return MemSizeType.kilobyte
        elif type_str == 'kb' or type_str == 'Kb':
            return MemSizeType.kilobit
        elif type_str == 'MB':
            return MemSizeType.megabyte
        elif type_str == 'Mb':
            return MemSizeType.megabit
        elif type_str == 'GB':
            return MemSizeType.gigabyte
        elif type_str == 'Gb':
            return MemSizeType.gigabit
        else:
            assert TypeError("Provided type_str is not valid")

    @staticmethod
    def size_type_2_str(size_type: MemSizeType):
        if size_type is MemSizeType.byte:
            return 'B'
        elif size_type is MemSizeType.bit:
            return 'b'
        elif size_type is MemSizeType.kilobyte:
            return 'kB'
        elif size_type is MemSizeType.kilobit:
            return 'kb'
        elif size_type is MemSizeType.megabyte:
            return 'MB'
        elif size_type is MemSizeType.megabit:
            return 'mb'
        elif size_type is MemSizeType.gigabyte:
            return 'GB'
        elif size_type is MemSizeType.gigabit:
            return 'Gb'
        else:
            assert TypeError("Provided MemType is not valid")

    def __str__(self):
        return super().__str__() + ' ' + self.size_type_2_str(self.size_type)


def get_size(test_obj: object, size_type: Union[str, MemSizeType] = MemSizeType.megabyte):
    if has_asizeof:
        size_byte = asizeof.asizeof(test_obj)

        conv_factor = 1e-6
        if size_type != 'MB':
            if size_type == 'B':
                conv_factor = 1
            elif size_type == 'kB' or size_type == 'KB':
                conv_factor = 1e-3
            elif size_type == 'GB':
                conv_factor = 1e-9
            elif size_type == 'b':
                conv_factor = 8e-6
            elif size_type == 'Mb':
                conv_factor = 8e-6
            elif size_type == 'kb' or size_type == 'Kb':
                conv_factor = 8e-3
            elif size_type == 'Gb':
                conv_factor = 8e-9
            else:
                assert TypeError("Provided size_type is not valid")

        return MemSize(size_byte * conv_factor, size_type=size_type)


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
