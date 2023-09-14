import cProfile
import io
import pstats
import enum
from typing import Union
import inspect

try:
    from pympler import asizeof

    has_asizeof = True
except ModuleNotFoundError:
    has_asizeof = False

# from termcolor import colored as termcolor

RUNTIME_CONSOLE = True
EXCLUSION_TYPES = [int, float, bool]
EXCLUSION_MODULES = ["numpy", "h5py", "h5pickle"]


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
    def __new__(cls, value, size_type):
        return float.__new__(cls, value)

    def __init__(
        self,
        value: Union[int, float],
        size_type: Union[str, MemSizeType] = MemSizeType.megabyte,
    ):
        float.__init__(value)

        if type(size_type) is str:
            size_type = self.str_2_size_type(size_type)
        self.size_type = size_type

    # def __repr__(self):
    #     return self.__str__() + " " + self.size_type_2_str(self.size_type)

    def in_mb(self) -> float:
        conversion_factor = None
        if self.size_type == MemSizeType.bit:
            conversion_factor = (1 / 8) * 1e-6
        elif self.size_type == MemSizeType.byte:
            conversion_factor = 1e-6
        elif self.size_type == MemSizeType.kilobit:
            conversion_factor = (1 / 8) * 1e-3
        elif self.size_type == MemSizeType.kilobyte:
            conversion_factor = 1e-3
        elif self.size_type == MemSizeType.megabit:
            conversion_factor = 1 / 8
        elif self.size_type == MemSizeType.megabyte:
            conversion_factor = 1
        elif self.size_type == MemSizeType.gigabit:
            conversion_factor = (1 / 8) * 1e3
        elif self.size_type == MemSizeType.gigabyte:
            conversion_factor = 1e3
        else:
            raise TypeError("MemSizeType not recognized")
        return self.real * conversion_factor

    @staticmethod
    def str_2_size_type(type_str: str):
        if type_str == "B":
            return MemSizeType.byte
        elif type_str == "b":
            return MemSizeType.bit
        elif type_str == "kB" or type_str == "KB":
            return MemSizeType.kilobyte
        elif type_str == "kb" or type_str == "Kb":
            return MemSizeType.kilobit
        elif type_str == "MB":
            return MemSizeType.megabyte
        elif type_str == "Mb":
            return MemSizeType.megabit
        elif type_str == "GB":
            return MemSizeType.gigabyte
        elif type_str == "Gb":
            return MemSizeType.gigabit
        else:
            assert TypeError("Provided type_str is not valid")

    @staticmethod
    def size_type_2_str(size_type: MemSizeType):
        if size_type is MemSizeType.byte:
            return "B"
        elif size_type is MemSizeType.bit:
            return "b"
        elif size_type is MemSizeType.kilobyte:
            return "kB"
        elif size_type is MemSizeType.kilobit:
            return "kb"
        elif size_type is MemSizeType.megabyte:
            return "MB"
        elif size_type is MemSizeType.megabit:
            return "mb"
        elif size_type is MemSizeType.gigabyte:
            return "GB"
        elif size_type is MemSizeType.gigabit:
            return "Gb"
        else:
            assert TypeError("Provided MemType is not valid")

    def __str__(self):
        return super().__str__() + " " + self.size_type_2_str(self.size_type)


def get_size(test_obj: object, size_type: Union[str, MemSizeType] = MemSizeType.megabyte):
    if has_asizeof:
        size_byte = asizeof.asizeof(test_obj)

        conv_factor = 1e-6
        if size_type != "MB" and size_type != MemSizeType.megabyte:
            if size_type == "B" or size_type == MemSizeType.byte:
                conv_factor = 1
            elif size_type == "kB" or size_type == "KB" or size_type == MemSizeType.kilobyte:
                conv_factor = 1e-3
            elif size_type == "GB" or size_type == MemSizeType.gigabyte:
                conv_factor = 1e-9
            elif size_type == "b" or size_type == MemSizeType.bit:
                conv_factor = 8e-6
            elif size_type == "Mb" or size_type == MemSizeType.megabit:
                conv_factor = 8e-6
            elif size_type == "kb" or size_type == "Kb" or MemSizeType.kilobit:
                conv_factor = 8e-3
            elif size_type == "Gb" or size_type == MemSizeType.gigabit:
                conv_factor = 8e-9
            else:
                assert TypeError("Provided size_type is not valid")

        return MemSize(size_byte * conv_factor, size_type=size_type)


def explore_sizes(
    test_obj: object,
    size_type: Union[str, MemSizeType] = MemSizeType.megabyte,
    max_level=5,
    min_size_mb=None,
    stop_on_max_level=True,
    exclusion_types=None,
    exclusion_modules=None,
    only_show_new=False,
    __current_level=0,
    __objs_tested=None,
):
    if __objs_tested is None:
        __objs_tested = list()
    if exclusion_modules is None:
        exclusion_modules = EXCLUSION_MODULES
    if exclusion_types is None:
        exclusion_types = EXCLUSION_TYPES

    if __current_level == 0:
        header = f"Sizes of {test_obj.__str__()}"
        print(f"{header}\n{'*' * len(header)}")
        level_prepend = ""
    else:
        level_prepend = "|  " * __current_level

    if __current_level > max_level:
        print(level_prepend + "|-- (Max level reached)")
        return True, __objs_tested

    members = inspect.getmembers(test_obj)
    for key, obj in members:
        if not key.startswith("__") and not callable(obj):
            already_explored = False
            size = get_size(test_obj=obj, size_type=size_type)
            if min_size_mb is None or (min_size_mb is not None and size.in_mb() >= min_size_mb):
                already_explored = id(obj) in __objs_tested
                if already_explored and only_show_new:
                    pass
                else:
                    if obj is None or type(obj) is bool:
                        already_explored_text = ""
                    else:
                        already_explored_text = " (already explored)" if already_explored else ""
                    print(f"{level_prepend}|-- {key} ({type(obj)}) -> {size}{already_explored_text}")
            __objs_tested.append(id(obj))
            if already_explored:
                continue
            elif type(obj) in [list, tuple] and len(obj) > 0:
                if not type(obj[0]) in EXCLUSION_TYPES and not type(obj[0]).__module__ in EXCLUSION_MODULES:
                    print(f"{level_prepend}| [0]")
                    max_level_reached, new_objs_tested = explore_sizes(
                        test_obj=obj[0],
                        size_type=size_type,
                        max_level=max_level,
                        min_size_mb=min_size_mb,
                        stop_on_max_level=stop_on_max_level,
                        exclusion_types=exclusion_types,
                        exclusion_modules=exclusion_modules,
                        only_show_new=only_show_new,
                        __current_level=__current_level + 1,
                        __objs_tested=__objs_tested,
                    )
                    if max_level_reached and not __current_level == 0:
                        __objs_tested.extend(new_objs_tested)
                        if stop_on_max_level:
                            return True, __objs_tested
                        else:
                            continue
            elif not type(obj) in EXCLUSION_TYPES and not type(obj).__module__ in EXCLUSION_MODULES:
                max_level_reached, new_objs_tested = explore_sizes(
                    test_obj=obj,
                    size_type=size_type,
                    max_level=max_level,
                    min_size_mb=min_size_mb,
                    stop_on_max_level=stop_on_max_level,
                    exclusion_types=exclusion_types,
                    exclusion_modules=exclusion_modules,
                    only_show_new=only_show_new,
                    __current_level=__current_level + 1,
                    __objs_tested=__objs_tested,
                )
                if max_level_reached and not __current_level == 0:
                    __objs_tested.extend(new_objs_tested)
                    if stop_on_max_level:
                        return True, __objs_tested
                    else:
                        continue
    if not __current_level == 0:
        return False, __objs_tested
    else:
        print("Done")


# task.obj._cpa.levels[0]._particle.dataset.particles[0]._histogram._particle.cpts._cpa.levels[0]._particle.dataset.particles[0].levels_roi[0]._microtimes._particle


def print_size(test_obj: object, size_type: Union[str, MemSizeType] = MemSizeType.megabyte):
    size = get_size(test_obj=test_obj, size_type=size_type)
    print(size)


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
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
            prepared_text = termcolor("[DEBUG]\t\t", "blue") + debug_print
    else:
        if RUNTIME_CONSOLE:
            prepared_text = f"[DEBUG]\t\t{debug_print}"
        else:
            prepared_text = termcolor("[DEBUG]\t\t", "blue") + termcolor(debug_from + ":\t", "green") + debug_print
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
        print("\r" + print_text, end="", flush=True)
    else:
        print("\r" + print_text, flush=True)


# sys.stdout.write('\r' + print_text)
# sys.stdout.flush()
