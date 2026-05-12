from exact import __version__
from exact.config import ROOT_DIR, config
from exact.router import TaskType, route_task


def test_config_paths():
    assert ROOT_DIR.is_dir()
    assert config.model_name


def test_router_physics_keyword():
    task, _, _ = route_task("Compute voltage across a 10 ohm resistor", None)
    assert task is TaskType.PHYSICS


def test_router_logic_when_premises():
    task, _, _ = route_task("Can I graduate?", ["If GPA > 2.0 then eligible."])
    assert task is TaskType.LOGIC


def test_version_defined():
    assert __version__
