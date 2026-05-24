from abc import ABC, abstractmethod

class PoTPromptInterface(ABC):
    """
    Abstract base class for all Program of Thoughts (PoT) prompt builders.
    Provides an interface to easily swap between different few-shot strategies.
    """
    @abstractmethod
    def build_prompt(self, question: str, target_unit: str) -> str:
        pass

class ZeroShotPoTPrompt(PoTPromptInterface):
    """Zero-shot prompt: Contains no examples, relying entirely on the LLM's internal knowledge."""
    def build_prompt(self, question: str, target_unit: str) -> str:
        prompt = f"""You are a physics and Python programming expert.\n"
        "Solve the following problem by writing Python code.\n"
        "Use the `pint` library to manage units. Assign the final result magnitude to the variable `ans`.\n\n"
        "[Problem]\n"
        "Question: {question}. Return the result in {target_unit}.\n"
        "# Python code, return ans\n"
        "```python\n"
        """
        return prompt

class OneShotPoTPrompt(PoTPromptInterface):
    """One-shot prompt: Provides a single, strong example (Capacitor energy)."""
    def build_prompt(self, question: str, target_unit: str) -> str:
        prompt = f"""You are a physics and Python programming expert.\n"
        "Solve the following problem by writing Python code.\n"
        "Use the `pint` library to manage units. Assign the final result magnitude to the variable `ans`.\n\n"
        "[Example]\n"
        "Question: Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V. Return the result in J.\n"
        "# Python code, return ans\n"
        "```python\n"
        "import pint\n"
        "ureg = pint.UnitRegistry()\n"
        "Q_ = ureg.Quantity\n\n"
        "C = Q_(100, 'uF')\n"
        "U = Q_(30, 'V')\n"
        "E = 0.5 * C * (U**2)\n"
        "ans = float(E.to('J').magnitude)\n"
        "```\n\n"
        "[Problem]\n"
        "Question: {question}. Return the result in {target_unit}.\n"
        "# Python code, return ans\n"
        "```python\n"
        """
        return prompt

class EightShotPoTPrompt(PoTPromptInterface):
    """Eight-shot prompt: Provides 8 diverse examples spanning different physics domains."""
    def build_prompt(self, question: str, target_unit: str) -> str:
        examples = """[Example 1]\n\
Question: Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V. Return the result in J.\n\
# Python code, return ans\n\
```python\n\
import pint\n\
ureg = pint.UnitRegistry()\n\
Q_ = ureg.Quantity\n\
C = Q_(100, 'uF')\n\
U = Q_(30, 'V')\n\
E = 0.5 * C * (U**2)\n\
ans = float(E.to('J').magnitude)\n\
```\n\n\
[Example 2]\n\
Question: A car travels 100 km in 2 hours. What is its average speed in m/s? Return the result in m/s.\n\
# Python code, return ans\n\
```python\n\
import pint\n\
ureg = pint.UnitRegistry()\n\
Q_ = ureg.Quantity\n\
d = Q_(100, 'km')\n\
t = Q_(2, 'hour')\n\
v = d / t\n\
ans = float(v.to('m/s').magnitude)\n\
```\n\n\
[Example 3]\n\
Question: Find the force required to accelerate a 50 kg mass at 2 m/s^2. Return the result in N.\n\
# Python code, return ans\n\
```python\n\
import pint\n\
ureg = pint.UnitRegistry()\n\
Q_ = ureg.Quantity\n\
m = Q_(50, 'kg')\n\
a = Q_(2, 'm/s**2')\n\
F = m * a\n\
ans = float(F.to('N').magnitude)\n\
```\n\n\
[Example 4]\n\
Question: A resistor of 10 ohms has a current of 2 A. What is the voltage? Return the result in V.\n\
# Python code, return ans\n\
```python\n\
import pint\n\
ureg = pint.UnitRegistry()\n\
Q_ = ureg.Quantity\n\
R = Q_(10, 'ohm')\n\
I = Q_(2, 'A')\n\
V = I * R\n\
ans = float(V.to('V').magnitude)\n\
```\n\n\
[Example 5]\n\
Question: Calculate the power dissipated by a 5 ohm resistor with 3 A current. Return the result in W.\n\
# Python code, return ans\n\
```python\n\
import pint\n\
ureg = pint.UnitRegistry()\n\
Q_ = ureg.Quantity\n\
R = Q_(5, 'ohm')\n\
I = Q_(3, 'A')\n\
P = (I**2) * R\n\
ans = float(P.to('W').magnitude)\n\
```\n\n\
[Example 6]\n\
Question: What is the kinetic energy of a 2 kg object moving at 3 m/s? Return the result in J.\n\
# Python code, return ans\n\
```python\n\
import pint\n\
ureg = pint.UnitRegistry()\n\
Q_ = ureg.Quantity\n\
m = Q_(2, 'kg')\n\
v = Q_(3, 'm/s')\n\
KE = 0.5 * m * (v**2)\n\
ans = float(KE.to('J').magnitude)\n\
```\n\n\
[Example 7]\n\
Question: A 10 N force acts over a distance of 5 m. What is the work done? Return the result in J.\n\
# Python code, return ans\n\
```python\n\
import pint\n\
ureg = pint.UnitRegistry()\n\
Q_ = ureg.Quantity\n\
F = Q_(10, 'N')\n\
d = Q_(5, 'm')\n\
W = F * d\n\
ans = float(W.to('J').magnitude)\n\
```\n\n\
[Example 8]\n\
Question: Find the frequency of a wave with a period of 0.05 s. Return the result in Hz.\n\
# Python code, return ans\n\
```python\n\
import pint\n\
ureg = pint.UnitRegistry()\n\
Q_ = ureg.Quantity\n\
T = Q_(0.05, 's')\n\
f = 1 / T\n\
ans = float(f.to('Hz').magnitude)\n\
```"""
        prompt = f"""You are a physics and Python programming expert.\n"
        "Solve the following problem by writing Python code.\n"
        "Use the `pint` library to manage units. Assign the final result magnitude to the variable `ans`.\n\n"
        f"{examples}\n\n"
        "[Problem]\n"
        "Question: {question}. Return the result in {target_unit}.\n"
        "# Python code, return ans\n"
        "```python\n"
        """
        return prompt


class Type2JsonFewShotPoTPrompt(PoTPromptInterface):
    """Few-shot prompt block for the Type 2 JSON PoT solver."""

    def build_prompt(self, question: str, target_unit: str) -> str:
        return f"""{self.examples()}

[Problem]
Question: {question}
Return the result in {target_unit}.
Return JSON only with keys code, explanation, answer_unit, formula_ids_used."""

    @staticmethod
    def examples() -> str:
        return """[Few-shot examples: valid JSON outputs]

[Example 1]
Question: Calculate the energy stored in capacitor C when C = 100 uF and U = 30 V.
Formula context IDs: capacitor_energy
Format note: code is a JSON string with escaped newlines, and formula_ids_used is a JSON array.
JSON:
{
  "code": "import pint\\nureg = pint.UnitRegistry()\\nQ_ = ureg.Quantity\\nC = Q_(100, 'uF')\\nU = Q_(30, 'V')\\nE = 0.5 * C * U**2\\nans = float(E.to('J').magnitude)\\nans_unit = 'J'",
  "explanation": "Used capacitor_energy: W = 1/2 * C * U^2.",
  "answer_unit": "J",
  "formula_ids_used": ["capacitor_energy"]
}

[Example 2]
Question: Calculate the voltage when current is 2 A and resistance is 5 ohm.
Formula context IDs: ohm_voltage
Format note: do not use markdown fences in the final JSON response.
JSON:
{
  "code": "import pint\\nureg = pint.UnitRegistry()\\nQ_ = ureg.Quantity\\nI = Q_(2, 'A')\\nR = Q_(5, 'ohm')\\nU = I * R\\nans = float(U.to('V').magnitude)\\nans_unit = 'V'",
  "explanation": "Used ohm_voltage: U = I * R.",
  "answer_unit": "V",
  "formula_ids_used": ["ohm_voltage"]
}

[Example 3]
Question: Three equal charges q = 1.6e-19 C are placed at the vertices of an equilateral triangle with side 16 cm. Find the net electric force on one charge.
Formula context IDs: net_force_equal_coulomb_equilateral
Format note: strict JSON only; no extra keys.
JSON:
{
  "code": "import math\\nimport pint\\nureg = pint.UnitRegistry()\\nQ_ = ureg.Quantity\\nk = Q_(8.9875517923e9, 'N*m^2/C^2')\\nq = Q_(1.6e-19, 'C')\\nr = Q_(16, 'cm')\\nF_one = k * q**2 / r**2\\nF_net = math.sqrt(3) * F_one\\nans = float(F_net.to('N').magnitude)\\nans_unit = 'N'",
  "explanation": "Used net_force_equal_coulomb_equilateral. The two equal forces have a 60 degree included angle, so the resultant is sqrt(3) times one Coulomb force.",
  "answer_unit": "N",
  "formula_ids_used": ["net_force_equal_coulomb_equilateral"]
}"""

def get_prompt_builder(prompt_type: str = 'one_shot') -> PoTPromptInterface:
    if prompt_type == 'zero_shot':
        return ZeroShotPoTPrompt()
    elif prompt_type == 'eight_shot':
        return EightShotPoTPrompt()
    elif prompt_type == 'type2_json_few_shot':
        return Type2JsonFewShotPoTPrompt()
    else:
        return OneShotPoTPrompt()
