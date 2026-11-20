from api import DataAPI


def test_files_are_sorted_by_name_so_that_last_file_is_the_newest(tmp_path):
    names = [
        "20261120_190003_data.h5",
        "20261120_190001_util.h5",
        "20261120_185959_filt.h5",
    ]
    for name in names:
        (tmp_path / name).touch()

    dut = DataAPI(tmp_path)
    dut.select_file(-1)

    assert [file.name for file in dut.get_overview_files] == sorted(names)
    assert dut.get_file_name() == "20261120_190003_data.h5"
