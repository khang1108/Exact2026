from exact.type2.extraction import extract_question
from exact.type2.normalization import normalize_units_and_notation


def _extract(text: str):
    return normalize_units_and_notation(extract_question(text))


def test_canonical_symbol_mapping_electric_field():
    extraction = _extract("E = 5 N/C")
    assert extraction.quantities[0].dimension == "electric_field"


def test_canonical_symbol_mapping_energy():
    extraction = _extract("E = 5 J")
    assert extraction.quantities[0].dimension == "energy"


def test_canonical_symbol_mapping_time():
    extraction = _extract("t = 2 s")
    assert extraction.quantities[0].dimension == "time"


def test_canonical_symbol_mapping_length():
    extraction = _extract("s = 2 m")
    assert extraction.quantities[0].dimension == "length"


def test_indexed_quantities_resistance():
    extraction = _extract("R1 = 4 ohm and R2 = 6 ohm")
    assert [q.symbol for q in extraction.quantities] == ["R1", "R2"]
    assert all(q.dimension == "resistance" for q in extraction.quantities)
    assert extraction.quantities[0].index == "1"
    assert extraction.quantities[1].index == "2"


def test_equality_chain_charge_quantities():
    extraction = _extract("q1 = q2 = q3 = 1.2 × 10^-6 C")
    assert [q.symbol for q in extraction.quantities] == ["q1", "q2", "q3"]
    assert all(q.dimension == "charge" for q in extraction.quantities)
    assert all(q.value == 1.2e-6 for q in extraction.quantities)
    assert extraction.relations[0].type == "equal_quantities"


def test_capacitor_equality_chain():
    extraction = _extract("C1 = C2 = 3 uF")
    assert [q.symbol for q in extraction.quantities] == ["C1", "C2"]
    assert all(q.dimension == "capacitance" for q in extraction.quantities)


def test_equal_vector_contributions_with_angle():
    extraction = _extract("two equal electric fields of magnitude E at an angle of 60 degrees")
    assert len(extraction.vector_contribution_groups) == 1
    group = extraction.vector_contribution_groups[0]
    assert group.count == 2
    assert group.quantity_dimension == "electric_field"
    assert group.magnitude_symbol == "E"
    assert group.angle_between_deg == 60


def test_equal_vector_contributions_without_angle():
    extraction = _extract("two equal forces of magnitude F")
    assert len(extraction.vector_contribution_groups) == 1
    group = extraction.vector_contribution_groups[0]
    assert group.count == 2
    assert group.quantity_dimension == "force"
    assert group.angle_between_deg is None


def test_phrase_style_voltage_battery_extraction():
    extraction = _extract("Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12 V battery.")
    voltage = next(q for q in extraction.quantities if q.dimension == "voltage")
    assert voltage.value == 12
    assert voltage.unit == "V"
    assert "battery" in (voltage.evidence or "")


def test_phrase_style_charge_distance_speed_field_extraction():
    extraction = _extract("A particle with charge 2 uC moves at 10 m/s perpendicular to a 0.5 T magnetic field.")
    dimensions = {q.dimension for q in extraction.quantities}
    assert "charge" in dimensions
    assert "speed" in dimensions
    assert "magnetic_field" in dimensions


def test_parallel_relation_extraction():
    extraction = _extract("R1 and R2 are connected in parallel")
    relation = extraction.relations[0]
    assert relation.type == "connected_in_parallel"
    assert relation.symbols == ["R1", "R2"]


def test_series_relation_inferred_from_two_resistors_statement():
    extraction = _extract("Two resistors R1 = 4 ohm and R2 = 6 ohm are in series")
    relation = extraction.relations[0]
    assert relation.type == "connected_in_series"
    assert relation.symbols == ["R1", "R2"]
