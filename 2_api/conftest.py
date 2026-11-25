import pytest

TEMP_FOLDER = "temp_test"


def pytest_sessionstart(session):
    from shutil import rmtree

    from api import get_path_to_project

    path2temp = get_path_to_project()
    rmtree(path2temp / TEMP_FOLDER, ignore_errors=True)
    (path2temp / TEMP_FOLDER).mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    from shutil import rmtree

    from api import get_path_to_project

    path2temp = get_path_to_project()
    rmtree(path2temp / TEMP_FOLDER, ignore_errors=True)


@pytest.fixture(scope="session")
def path():
    from api import get_path_to_project

    return get_path_to_project(TEMP_FOLDER)
