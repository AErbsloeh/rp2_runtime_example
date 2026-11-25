from shutil import rmtree

from elasticai.creator.file_generation import find_project_root

pytest_plugins = ["elasticai.creator.testing.cocotb_pytest"]

SIMULATION_FOLDER = "build_test"


def pytest_sessionstart(session):
    rmtree(find_project_root() / SIMULATION_FOLDER, ignore_errors=True)
