import traceback
from exact.type2.solving.pot_solver import generate_pot_code
from exact.config import get_settings

try:
    settings = get_settings()
    generate_pot_code('test', 'test', None, settings, {})
except Exception as e:
    traceback.print_exc()
