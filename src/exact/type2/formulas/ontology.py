from __future__ import annotations


ONTOLOGY: dict[str, dict[str, tuple[str, ...]]] = {
    "mechanics": {
        "kinematics": ("speed", "velocity", "acceleration", "distance", "time", "motion"),
        "dynamics": ("force", "mass", "newton", "friction", "weight"),
        "energy": ("work", "kinetic", "potential energy", "mechanical energy", "power"),
        "momentum": ("momentum", "impulse", "collision"),
    },
    "electricity": {
        "electrostatics": ("charge", "coulomb", "electric field", "field strength", "electric potential"),
        "potential": ("potential", "voltage", "potential difference", "potential energy"),
        "capacitors": ("capacitor", "capacitance", "dielectric", "parallel plate", "stored energy"),
        "dc_circuits": ("current", "resistance", "resistor", "ohm", "series", "parallel"),
        "ac_circuits": ("impedance", "reactance", "rms", "phase", "resonance", "rlc"),
    },
    "thermal": {
        "heat": ("heat", "specific heat", "calorimetry", "temperature"),
        "gas_laws": ("pressure", "volume", "temperature", "gas", "mole"),
    },
    "optics": {
        "geometric_optics": ("lens", "mirror", "focal", "image", "refraction"),
    },
    "waves": {
        "waves": ("wave", "wavelength", "frequency", "period", "sound"),
    },
    "magnetism": {
        "magnetism": ("magnetic field", "magnetic flux", "solenoid", "transformer", "inductor"),
    },
}

