from exact.common.schemas import PredictionRequest
from exact.type2.pipeline import run_type2_internal_pipeline


def test_parallel_resistor_current_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="parallel-current",
            type="type2",
            question="Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12 V battery. Find the total current.",
        )
    )
    assert result.answer == "5"
    assert result.unit == "A"


def test_series_resistor_current_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="series-current",
            type="type2",
            question="Two resistors R1 = 4 ohm and R2 = 6 ohm are in series across a 12 V battery. Find the total current.",
        )
    )
    assert result.answer == "1.2"
    assert result.unit == "A"


def test_capacitor_energy_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="capacitor-energy",
            type="type2",
            question="A capacitor of 3 uF is connected to a 12 V battery. Find the energy stored.",
        )
    )
    assert result.unit == "J"


def test_disconnected_dielectric_capacitor_energy_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="dielectric-disconnected-energy",
            type="type2",
            question=(
                "An air-filled parallel plate capacitor with capacitance C = 500 pF is charged to a voltage U = 300 V. "
                "The capacitor is then disconnected from the source and immersed in a liquid dielectric with a relative "
                "permittivity (dielectric constant) of ε_r = 2. What is the electric field energy stored between the "
                "plates of the capacitor?"
            ),
        )
    )
    assert result.answer == "1.125e-05"
    assert result.unit == "J"


def test_connected_dielectric_capacitor_energy_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="dielectric-connected-energy",
            type="type2",
            question=(
                "An air-filled parallel-plate capacitor has a capacitance C = 500 pF and is charged to a voltage U = 300 V. "
                "The capacitor remains connected to the voltage source while it is immersed in a liquid dielectric with "
                "a dielectric constant ε = 2. What is the electric field energy between the plates of the capacitor?"
            ),
        )
    )
    assert result.answer == "4.5e-05"
    assert result.unit == "J"


def test_electrostatics_distance_phrase_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="electrostatics-distance",
            type="type2",
            question="A charge of 2 uC is placed 5 cm away from a point. Find the electric field at the point.",
        )
    )
    assert result.unit == "N/C"


def test_speed_phrase_magnetic_force_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="magnetic-force",
            type="type2",
            question="A particle with charge 2 uC moves at 10 m/s perpendicular to a 0.5 T magnetic field. Find the magnetic force.",
        )
    )
    assert result.unit == "N"


def test_equal_charges_midpoint_field_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="equal-charges-midpoint-field",
            type="type2",
            question=(
                "Two electric charges, q1 = q2 = 5 x 10^-9 C, are placed 10 cm apart in a vacuum. "
                "What is the magnitude of the electric field strength at the midpoint of the line segment connecting the two charges?"
            ),
        )
    )
    assert result.answer == "0"
    assert result.unit == "V/m"


def test_opposite_charges_collinear_field_smoke():
    result = run_type2_internal_pipeline(
        PredictionRequest(
            query_id="opposite-charges-collinear-field",
            type="type2",
            question=(
                "Two electric charges q1 = 5 x 10^-9 C and q2 = -5 x 10^-9 C are placed at two points separated by "
                "10 cm in a vacuum. Calculate the magnitude of the electric field strength at a point located on the "
                "straight line passing through the two charges, and which is 5 cm from q1 and 15 cm from q2."
            ),
        )
    )
    assert result.answer == "19972.3"
    assert result.unit == "V/m"
