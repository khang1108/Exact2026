from __future__ import annotations

from exact.type2.formulas.bank import FORMULAS


def test_formula_bank_includes_new_electrostatics_and_material_formulas() -> None:
    ids = {formula.id for formula in FORMULAS}

    assert "electric_field_point_charge" in ids
    assert "electric_potential_energy_two_point_charges" in ids
    assert "net_force_equal_coulomb_equilateral" in ids
    assert "capacitance_from_energy_voltage" in ids
    assert "resistivity_from_resistance_area_length" in ids


def test_formula_bank_includes_focused_electricity_seed_formulas() -> None:
    ids = {formula.id for formula in FORMULAS}

    expected = {
        "charge_from_current_time",
        "current_from_charge_time",
        "voltage_from_power_current",
        "current_from_power_voltage",
        "resistance_from_power_current",
        "current_from_power_resistance",
        "voltage_from_power_resistance",
        "resistance_from_voltage_power",
        "energy_from_power_time",
        "power_from_energy_time",
        "series_resistance",
        "parallel_resistance_two",
        "series_capacitance_two",
        "parallel_capacitance",
        "capacitor_energy_from_charge_voltage",
        "capacitor_energy_from_charge_capacitance",
        "voltage_from_capacitor_energy_capacitance",
        "charge_from_capacitor_energy_capacitance",
        "electric_field_uniform_from_voltage_distance",
        "voltage_from_uniform_electric_field",
        "electric_potential_point_charge",
    }

    assert expected <= ids


def test_formula_bank_stays_focused_on_requested_domains() -> None:
    domains = {formula.domain for formula in FORMULAS}

    assert domains <= {"circuits", "capacitance", "inductance", "electrostatics"}
