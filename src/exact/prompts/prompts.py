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
                        "C = Q_(100, 'microfarad')\n"
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
