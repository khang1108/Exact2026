from __future__ import annotations


class Type2JsonFewShotPoTPrompt:
    """Few-shot examples for the Type 2 JSON PoT solver."""

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
