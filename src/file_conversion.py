import sys
import os
import struct
import codecs

import numpy as np

# import progressbar
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from PyQt5 import uic
import h5py

import file_manager as fm
from my_logger import setup_logger

logger = setup_logger(__name__)

convert_file_dialog_file = fm.path(name="convert_file_dialog.ui", file_type=fm.Type.UI)
UI_Convert_File_Dialog, _ = uic.loadUiType(convert_file_dialog_file)
# convert_csv_dialog_file = fm.path(name="convert_csv_dialog.ui", file_type=fm.Type.UI)
# UI_Convert_CSV_Dialog, _ = uic.loadUiType(convert_csv_dialog_file)


class Pt3Reader:
    """Class for reading .pt3 files"""
    def __init__(self, pt3_file_path):
        self._file_path = pt3_file_path
        file_size = os.stat(pt3_file_path).st_size

        with open(self._file_path, "rb") as f:

            def fr(num_bytes: int = 1):
                read_bytes = f.read(num_bytes)
                if read_bytes != "":
                    return read_bytes
                else:
                    return False

            def fr_str(num_chars: int = 1):
                str_bytes = fr(num_chars)
                if str_bytes:
                    converted_string = str()
                    for char_byte in str_bytes:
                        if char_byte != 0:
                            converted_string += codecs.decode(char_byte.to_bytes(1, "little"), "ascii")
                    return converted_string
                else:
                    return False

            def fr_int32():
                int_bytes = fr(4)
                if int_bytes:
                    return int.from_bytes(int_bytes, byteorder="little", signed=True)
                else:
                    return False

            def fr_uint32():
                uint_bytes = fr(4)
                if uint_bytes:
                    return int.from_bytes(uint_bytes, byteorder="little", signed=False)
                else:
                    return False

            def fr_float():
                float_bytes = fr(4)
                if float_bytes:
                    return struct.unpack("<f", float_bytes)[0]
                else:
                    return False

            self.ident = fr_str(16)
            self.format_version = fr_str(6)
            self.creator_name = fr_str(18)
            self.creator_version = fr_str(12)
            self.file_time = fr_str(18)
            self.crlf = fr_str(2)
            self.comment_field = fr_str(256)

            self.curves = fr_int32()
            self.bits_per_record = fr_int32()
            self.num_routing_channels = fr_int32()
            self.number_of_boards = fr_int32()
            self.active_curve = fr_int32()
            self.measurement_mode = fr_int32()
            self.sub_mode = fr_int32()
            self.range_no = fr_int32()
            self.offset = fr_int32()
            self.acquisition_time = fr_int32()
            self.stop_at = fr_int32()
            self.stop_on_overflow = fr_int32()
            self.restart = fr_int32()
            self.display_lin_log = fr_int32()
            self.display_time_from = fr_int32()
            self.display_time_to = fr_int32()
            self.display_count_from = fr_int32()
            self.display_cout_to = fr_int32()

            self.display_curve_map_to = [None] * 8
            self.display_curve_show = [None] * 8
            for i in range(8):
                self.display_curve_map_to[i] = fr_int32()
                self.display_curve_show[i] = fr_int32()

            self.parameter_start = [None] * 3
            self.parameter_step = [None] * 3
            self.parameter_end = [None] * 3
            for i in range(3):
                self.parameter_start[i] = fr_float()
                self.parameter_step[i] = fr_float()
                self.parameter_end[i] = fr_float()

            self.repeat_mode = fr_int32()
            self.repeats_per_curve = fr_int32()
            self.repeat_time = fr_int32()
            self.repeat_wait = fr_int32()
            self.script_name = fr_str(20)

            self.hardware_ident = fr_str(16)
            self.hardware_version = fr_str(8)
            self.hardware_serial = fr_int32()
            self.sync_devider = fr_int32()
            self.cfd_zero_cross_0 = fr_int32()
            self.cfd_level_0 = fr_int32()
            self.cfd_zero_cross_1 = fr_int32()
            self.cfd_level_1 = fr_int32()
            self.resolution = fr_float()

            self.router_mode_code = fr_int32()
            self.router_enabled = fr_int32()

            class RouterChannel:
                input_type = None
                input_level = None
                input_edge = None
                cfd_present = None
                cfd_level = None
                cfd_zero_cross = None

            self.channels_info = [RouterChannel()] * self.num_routing_channels

            for channel in self.channels_info:
                channel.input_type = fr_int32()
                channel.input_level = fr_int32()
                channel.input_edge = fr_int32()
                channel.cfd_present = fr_int32()
                channel.cfd_level = fr_int32()
                channel.cfd_zero_cross = fr_int32()

            self.external_devices = fr_int32()
            self.reserved_1 = fr_int32()
            self.reserved_2 = fr_int32()
            self.count_rate_0 = fr_int32()
            self.count_rate_1 = fr_int32()
            self.stop_after = fr_int32()
            self.stop_reason = fr_int32()
            self.num_records = fr_uint32()
            self.image_header_size = fr_int32()
            self.image_header = [fr_int32() for _ in range(self.image_header_size)]

            self._overflow_time = 0
            # self._counter_1 = 0
            # self._counter_2 = 0
            # self._counter_3 = 0
            # self._counter_4 = 0
            # self._counter_err = 0
            self._wrap_around = 65536

            self._sync_period = 1e9 / self.count_rate_0  # ns
            self._delay_times = np.zeros(self.num_records)

            class Records:
                def __init__(self, num_records: int):
                    self.count = 0
                    self.micro_times = np.empty(num_records, dtype=np.uint16) * np.nan
                    self.macro_times = np.empty(num_records, dtype=np.uint16) * np.nan

                def trim_empty(self):
                    if self.count == 0:
                        self.micro_times = None
                        self.macro_times = None
                    else:
                        self.micro_times = np.delete(self.micro_times, np.s_[self.count :])
                        self.macro_times = np.delete(self.macro_times, np.s_[self.count :])

            self.channel_records = [Records(self.num_records) for _ in range(self.num_routing_channels)]
            overflow = 0
            bar = False
            if "--dev" in sys.argv:
                # bar = progressbar.ProgressBar(max_value=file_size)
                pass
            for i in range(self.num_records):
                # +---------------+   +---------------+  +---------------+   +---------------+
                # |x|x|x|x|x|x|x|x|   |x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|   |x|x|x|x|x|x|x|x|
                # +---------------+   +---------------+  +---------------+   +---------------+
                t3_record = fr_uint32()
                if t3_record is False:
                    if bar:
                        bar.update(f.tell())
                    break
                # print('{0:32b}'.format(t3_record))

                # +---------------+   +---------------+  +---------------+   +---------------+
                # |x|x|x|x| | | | |   | | | | | | | | |  | | | | | | | | |   | | | | | | | | |
                # +---------------+   +---------------+  +---------------+   +---------------+
                channel = t3_record >> (32 - 4)  # -1  # Not sure about the -1?
                # print('{0:32b}'.format(t3_record >> (32 - 4)))

                if channel != 15:  # Valid record, not overflow
                    c_r = self.channel_records[channel - 1]

                    # +---------------+   +---------------+  +---------------+   +---------------+
                    # | | | | | | | | |   | | | | | | | | |  |x|x|x|x|x|x|x|x|   |x|x|x|x|x|x|x|x|
                    # +---------------+   +---------------+  +---------------+   +---------------+
                    nsync = t3_record & int("11111111" * 2, 2)
                    # print('{0:32b}'.format(t3_record & int('11111111'*2, 2)))

                    # +---------------+   +---------------+  +---------------+   +---------------+
                    # | | | | |x|x|x|x|   |x|x|x|x|x|x|x|x|  | | | | | | | | |   | | | | | | | | |
                    # +---------------+   +---------------+  +---------------+   +---------------+
                    dtime = t3_record >> 16 & int("00001111" + "11111111", 2)
                    # print('{0:32b}'.format(t3_record >> 16 & int('00001111' + '11111111', 2)))

                    c_r.micro_times[c_r.count] = dtime * self.resolution

                    true_sync = overflow + nsync
                    macro_time = (true_sync * self._sync_period) + c_r.micro_times[c_r.count]
                    c_r.macro_times[c_r.count] = macro_time

                    c_r.count += 1
                else:
                    # +---------------+   +---------------+  +---------------+   +---------------+
                    # | | | | | | | | |   | | | | |x|x|x|x|  | | | | | | | | |   | | | | | | | | |
                    # +---------------+   +---------------+  +---------------+   +---------------+
                    markers = t3_record >> 16 & int("00000000" + "00001111", 2)
                    # print('{0:32b}'.format(t3_record >> 16 & int('00000000' + '00001111', 2)))

                    if markers == int("0000", 2):  # then this is a overflow record
                        overflow += self._wrap_around
                if bar:
                    bar.update(f.tell())
            if bar:
                bar.finish()

            for channel in self.channel_records:
                channel.trim_empty()

    def add_particle_to_h5(self, h5_file):
        pass


class CSVReader():
    """Class for reading .csv files"""
    def __init__(self, csv_file_path):
        self._file_path = csv_file_path
        file_size = os.stat(csv_file_path).st_size
        self.macro_times, self.micro_times = np.loadtxt(csv_file_path, delimiter=',', skiprows=1, unpack=True)


class PhotonHDF5Reader():
    """Class for reading Photon-HDF5 files"""
    def __init__(self, hdf5_filepath):
        self._file_path = hdf5_filepath
        file = h5py.File(hdf5_filepath, 'r')
        timestamps = file['photon_data/timestamps']
        timestamp_unit = file['photon_data/timestamps_specs/timestamps_unit'][()]
        nanotimes = file['photon_data/nanotimes']
        nanotimes_unit = file['photon_data/nanotimes_specs/tcspc_unit'][()]
        self.macro_times = timestamps * timestamp_unit * 1e9  # convert from s to ns
        self.micro_times = nanotimes * nanotimes_unit * 1e9  # convert from s to ns
        self.file_time = file['provenance/creation_time']
        self.comment_field = file['description']
        self.creator_name = file['identity/author']


class FileReader():
    """Class for reading source files.

    A relevant lower-level class (currently either Pt3Reader or CSVReader) is
    instantiated to read the source file.

    Parameters
    ----------
    src_format : string
        Source file format (csv or pt3)
    file_path : string
        Source file path
    channel : int
        Photon channel number (only for pt3)
    """

    def __init__(self, src_format, file_path, channel=None):
        self.format = src_format
        self.file_path = file_path
        if self.format == "h5":
            phdf_reader = PhotonHDF5Reader(file_path)
            self.abs_times = phdf_reader.macro_times
            self.micro_times = phdf_reader.micro_times
            self.file_time = phdf_reader.file_time
            self.comment_field = phdf_reader.comment_field
            self.creator_name = phdf_reader.creator_name
        if self.format == "csv":
            csv_reader = CSVReader(file_path)
            self.abs_times = csv_reader.macro_times
            self.micro_times = csv_reader.micro_times
            self.file_time = ""
            self.comment_field = ""
            self.creator_name = ""
        elif self.format == "pt3":
            pt3_reader = Pt3Reader(file_path)
            self.abs_times = pt3_reader.channel_records[channel].macro_times
            self.micro_times = pt3_reader.channel_records[channel].micro_times
            self.file_time = pt3_reader.file_time
            self.comment_field = pt3_reader.comment_field
            self.creator_name = pt3_reader.creator_name


class ConvertFileDialog(QDialog, UI_Convert_File_Dialog):
    def __init__(self, mainwindow):
        QDialog.__init__(self)
        UI_Convert_File_Dialog.__init__(self)
        self.setupUi(self)

        self.mainwindow = mainwindow
        self.parent = mainwindow

        self.cmbFileFormat.currentIndexChanged.connect(self.set_source_format)
        self.btnSourceFolder.clicked.connect(self.set_source_folder)
        self.btnExportFile.clicked.connect(self.set_export_file)
        self.btnConvert.clicked.connect(self.convert)
        self.cbxHasSpectra.stateChanged.connect(self.change_spectra_edt)
        self.edtFileNames_times.textChanged.connect(self.check_ready)
        self.edtFileNames_Spectra.textChanged.connect(self.check_ready)
        if self.cmbFileFormat.currentText() == ".csv":
            self.spbChannel.setEnabled(False)

        self._source_path = None
        self._export_path = None

    def set_source_format(self):
        # channel disabled for .csv files
        self.spbChannel.setEnabled(not self.cmbFileFormat.currentText() == ".csv")

    def set_source_folder(self):
        f_dir = QFileDialog.getExistingDirectory(self)
        if f_dir != ("", ""):
            self._source_path = f_dir

            display_path = f_dir
            if len(f_dir) > 48:
                display_path = f_dir[:22] + "..." + f_dir[-22:]
            self.edtSourceFolder.setText(display_path)
            self.check_ready()

    def set_export_file(self):
        f_dir, _ = QFileDialog.getSaveFileName(filter="HDF5 file (*.h5)")
        if f_dir != ("", ""):
            if not os.path.exists(f_dir):
                self._export_path = f_dir
                display_path = f_dir
                if len(f_dir) > 48:
                    display_path = f_dir[:22] + "..." + f_dir[-22:]
                self.edtExportFile.setText(display_path)
            else:
                msg = QMessageBox(
                    QMessageBox.Warning,
                    "Convert file",
                    "File exists!",
                    buttons=QMessageBox.Ok,
                    parent=self,
                )

                msg.exec_()
            self.check_ready()

    def change_spectra_edt(self):
        has_spectra = self.cbxHasSpectra.isChecked()
        self.edtFileNames_Spectra.setEnabled(has_spectra)
        self.spbExposure.setEnabled(has_spectra)

    def check_ready(self):
        is_ready = False
        if self._export_path and self._source_path:
            times_names = self.edtFileNames_times.text()
            spectra_names = self.edtFileNames_Spectra.text()
            if times_names != "" and spectra_names != "":
                is_ready = True

        self.btnConvert.setEnabled(is_ready)

    def convert(self):
        self.setEnabled(False)
        all_source_files = [
            f for f in os.listdir(self._source_path) if os.path.isfile(os.path.join(self._source_path, f))
        ]

        pt3_f_name = self.edtFileNames_times.text()
        file_ext = self.cmbFileFormat.currentText()
        pt3_fs = [file for file in all_source_files if file.startswith(pt3_f_name) and file.endswith(file_ext)]
        pt3_fs.sort()
        num_files = len(pt3_fs)

        all_okay = False
        if all([file[-len(file_ext):] == file_ext for file in pt3_fs]):
            pt3_nums = [file[len(pt3_f_name) : file.find(".")] for file in pt3_fs]
            if all([num.isalnum() for num in pt3_nums]):
                pt3_nums = [int(num) for num in pt3_nums]
                pt3_nums.sort()
                all_okay = True

        spec_fs = None
        add_spec = self.cbxHasSpectra.isChecked()
        if add_spec:
            all_okay = False
            spec_file_name = self.edtFileNames_Spectra.text()
            spec_fs = [file for file in all_source_files if file.startswith(spec_file_name)]
            spec_fs.sort()
            if len(pt3_fs) == len(spec_fs):
                spec_file_ext = spec_fs[0].find(".")
                if spec_file_ext == -1:
                    spec_nums = [file[len(spec_file_name) :] for file in spec_fs]
                else:
                    spec_nums = [file[len(spec_file_name) : file.find(".")] for file in spec_fs]
                if all([num.isalnum() for num in spec_nums]):
                    spec_nums = [int(num) for num in spec_nums]
                    spec_nums.sort()
                    if pt3_nums == spec_nums:
                        all_okay = True

        if not all_okay:
            message = "photon time files and spectra files dont match, please check"
            msg = QMessageBox(
                QMessageBox.Warning,
                "Convert file",
                message,
                buttons=QMessageBox.Ok,
                parent=self,
            )
            msg.exec_()
            return

        with h5py.File(self._export_path, "w") as h5_f:
            h5_f.attrs.create(name="# Particles", data=num_files)
            channel = self.spbChannel.value() - 1

            try:
                for num in range(num_files):
                    file_reader = FileReader(src_format=file_ext[1:],
                                            file_path=os.path.join(self._source_path, pt3_fs[num]),
                                            channel=channel)
                    part_group = h5_f.create_group("Particle " + str(num + 1))
                    part_group.attrs.create("Date", file_reader.file_time)
                    part_group.attrs.create("Discription", file_reader.comment_field)
                    part_group.attrs.create("Intensity?", 1)
                    part_group.attrs.create("RS Coord. (um)", [0, 0])
                    part_group.attrs.create("Spectra?", int(add_spec))
                    part_group.attrs.create("User", file_reader.creator_name)

                    abs_times = file_reader.abs_times
                    micro_times = file_reader.micro_times
                    part_group.create_dataset("Absolute Times (ns)", dtype=np.uint64, data=abs_times)
                    part_group.create_dataset("Micro Times (s)", dtype=np.float64, data=micro_times)

                    if add_spec:
                        spec_data = np.loadtxt(os.path.join(self._source_path, spec_fs[num]))
                        wavelengths = spec_data[:, 0]
                        spec_data = np.delete(spec_data, 0, axis=1).T
                        exposure = self.spbExposure.value()
                        spec_t_series = np.array([(n + 1) * exposure for n in range(spec_data.shape[0])])

                        spec_dataset = part_group.create_dataset(
                            "Spectra (counts\s)", dtype=np.float64, data=spec_data
                        )
                        spec_dataset.attrs.create("Exposure Times (s)", exposure)
                        spec_dataset.attrs.create(
                            "Spectra Abs. Times (s)",
                            dtype=np.float64,
                            data=spec_t_series,
                        )
                        spec_dataset.attrs.create("Wavelengths", dtype=np.float64, data=wavelengths)

            except Exception as e:
                raise
                logger.error(e)

            logger.info("Finished converting files")
            self.setEnabled(True)


if __name__ == "__main__":
    pt3_file = Pt3Reader("C:\\Google Drive\\Current_Projects\\Full_SMS\\test_data\\trace0.pt3")
    # print('here')
