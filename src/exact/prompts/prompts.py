import json
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
        examples = [
            (
                "Calculate the energy stored in capacitor C when C = 100 uF and U = 30 V.",
                "capacitor_energy",
                {
                    "code": (
                        "import pint\n"
                        "ureg = pint.UnitRegistry()\n"
                        "Q_ = ureg.Quantity\n"
                        "C = Q_(100, 'uF')\n"
                        "U = Q_(30, 'V')\n"
                        "E = 0.5 * C * U**2\n"
                        "ans = float(E.to('J').magnitude)\n"
                        "ans_unit = 'J'"
                    ),
                    "explanation": "Used capacitor_energy: W = 1/2 * C * U^2.",
                    "answer_unit": "J",
                    "formula_ids_used": ["capacitor_energy"],
                },
            ),
            (
                "Calculate the voltage when current is 2 A and resistance is 5 ohm.",
                "ohm_voltage",
                {
                    "code": (
                        "import pint\n"
                        "ureg = pint.UnitRegistry()\n"
                        "Q_ = ureg.Quantity\n"
                        "I = Q_(2, 'A')\n"
                        "R = Q_(5, 'ohm')\n"
                        "U = I * R\n"
                        "ans = float(U.to('V').magnitude)\n"
                        "ans_unit = 'V'"
                    ),
                    "explanation": "Used ohm_voltage: U = I * R.",
                    "answer_unit": "V",
                    "formula_ids_used": ["ohm_voltage"],
                },
            ),
            (
                "Three equal charges q = 1.6e-19 C are placed at the vertices of an equilateral triangle with side 16 cm. Find the net electric force on one charge.",
                "net_force_equal_coulomb_equilateral",
                {
                    "code": (
                        "import math\n"
                        "import pint\n"
                        "ureg = pint.UnitRegistry()\n"
                        "Q_ = ureg.Quantity\n"
                        "k = Q_(8.9875517923e9, 'N*m^2/C^2')\n"
                        "q = Q_(1.6e-19, 'C')\n"
                        "r = Q_(16, 'cm')\n"
                        "F_one = k * q**2 / r**2\n"
                        "F_net = math.sqrt(3) * F_one\n"
                        "ans = float(F_net.to('N').magnitude)\n"
                        "ans_unit = 'N'"
                    ),
                    "explanation": "Used net_force_equal_coulomb_equilateral. The two equal forces have a 60 degree included angle, so the resultant is sqrt(3) times one Coulomb force.",
                    "answer_unit": "N",
                    "formula_ids_used": ["net_force_equal_coulomb_equilateral"],
                },
            ),
        ]
        lines = ["[Few-shot examples: valid strict JSON outputs]"]
        for index, (question, formula_id, payload) in enumerate(examples, start=1):
            lines.extend(
                [
                    "",
                    f"[Example {index}]",
                    f"Question: {question}",
                    f"Formula context IDs: {formula_id}",
                    "JSON:",
                    json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                ]
            )
        return "\n".join(lines)

def get_prompt_builder(prompt_type: str = 'one_shot') -> PoTPromptInterface:
    if prompt_type == 'zero_shot':
        return ZeroShotPoTPrompt()
    elif prompt_type == 'eight_shot':
        return EightShotPoTPrompt()
    elif prompt_type == 'type2_json_few_shot':
        return Type2JsonFewShotPoTPrompt()
    else:
        return OneShotPoTPrompt()
