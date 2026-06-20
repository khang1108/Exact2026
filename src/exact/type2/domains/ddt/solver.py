from __future__ import annotations

import math

from exact.type2.domains.ddt.schemas import DdtAnswer, DdtContract, DdtQuantity

MU0 = 4 * math.pi * 1e-7


def solve_ddt_contract(contract: DdtContract) -> DdtAnswer | None:
    conceptual = _solve_conceptual(contract)
    if conceptual is not None:
        return conceptual
    q = contract.quantities
    if contract.family == "SOLENOID_FIELD":
        return _solve_solenoid_field(contract, q)
    if contract.family == "SOLENOID_INDUCTANCE":
        return _solve_solenoid_inductance(contract, q)
    if contract.family == "INDUCED_EMF":
        return _solve_induced_emf(contract, q)
    if contract.family == "MAGNETIC_FLUX":
        return _solve_flux(contract, q)
    if contract.family in {"RLC_REACTANCE", "RLC_IMPEDANCE", "RLC_RESONANCE"}:
        return _solve_rlc(contract, q)
    if contract.family == "LC_ENERGY":
        return _solve_lc_energy(contract, q)
    return None


def _solve_solenoid_field(contract: DdtContract, q: dict[str, DdtQuantity]) -> DdtAnswer | None:
    n = _val(q, "turn_density")
    if n is None:
        N, length = _val(q, "N"), _val(q, "length")
        if N is not None and length:
            n = N / length
    I = _val(q, "I")
    if contract.target == "turn_density":
        N, length = _val(q, "N"), _val(q, "length")
        if N is not None and length:
            return _answer(N / length, "turns/m", "Computed turn density n=N/l.")
    if contract.target == "energy_density":
        B = _val(q, "B")
        if B is not None:
            return _answer((B * B) / (2 * MU0), "J/m^3", "Computed magnetic energy density B^2/(2 mu0).")
        if n is not None and I is not None:
            B = MU0 * n * I
            return _answer((B * B) / (2 * MU0), "J/m^3", "Computed magnetic energy density B^2/(2 mu0).")
    if contract.target == "energy":
        area = _area(q.get("area"))
        length = _val(q, "length")
        if n is not None and I is not None and area is not None and length is not None:
            B = MU0 * n * I
            return _answer((B * B) / (2 * MU0) * area * length, "J", "Computed solenoid magnetic-field energy density times volume.")
    if n is not None and I is not None:
        return _answer(MU0 * n * I, "T", "Computed solenoid field B=mu0*n*I.")
    return None


def _solve_solenoid_inductance(contract: DdtContract, q: dict[str, DdtQuantity]) -> DdtAnswer | None:
    N, length, area = _val(q, "N"), _val(q, "length"), _area(q.get("area"))
    if N is not None and length and area is not None:
        return _answer(MU0 * N * N * area / length, "H", "Computed air-core solenoid inductance.")
    return None


def _solve_induced_emf(contract: DdtContract, q: dict[str, DdtQuantity]) -> DdtAnswer | None:
    L, t = _val(q, "L"), _val(q, "time")
    i0, i1 = _val(q, "I_initial"), _val(q, "I_final")
    if L is not None and t and i0 is not None and i1 is not None:
        return _answer(abs(L * (i1 - i0) / t), "V", "Computed induced EMF magnitude epsilon=L*|dI/dt|.")
    flux, t = _val(q, "flux"), _val(q, "time")
    if flux is not None and t:
        N = _val(q, "N")
        turns = N if N is not None else 1.0
        return _answer(abs(turns * flux / t), "V", "Computed average EMF from flux linkage change.")
    U, t, i0, i1 = _val(q, "U"), _val(q, "time"), _val(q, "I_initial"), _val(q, "I_final")
    if U is not None and t and i0 is not None and i1 is not None:
        return _answer(abs(U * t / (i1 - i0)), "H", "Rearranged epsilon=L*|dI/dt|.")
    return None


def _solve_flux(contract: DdtContract, q: dict[str, DdtQuantity]) -> DdtAnswer | None:
    B = _val(q, "B")
    area = _area(q.get("area"))
    N = _val(q, "N")
    if contract.target == "flux_linkage":
        flux = _val(q, "flux")
        if flux is not None and N is not None:
            return _answer(N * flux, "Wb", "Computed flux linkage N*Phi.")
    if B is None:
        field = _solve_solenoid_field(contract, q)
        if field is not None:
            try:
                B = float(field.answer)
            except ValueError:
                B = None
    if B is not None and area is not None:
        value = B * area
        if N is not None and contract.target == "flux_linkage":
            value *= N
        return _answer(value, "Wb", "Computed magnetic flux Phi=B*A.")
    return None


def _solve_rlc(contract: DdtContract, q: dict[str, DdtQuantity]) -> DdtAnswer | None:
    f, C, L, R, Z, U = _val(q, "frequency"), _cap(q.get("C")), _val(q, "L"), _val(q, "R"), _val(q, "Z"), _val(q, "U")
    if contract.target == "capacitive_reactance" and f and C:
        if R is not None and Z:
            return _answer(math.sqrt(max(Z * Z - R * R, 0.0)), "Ω", "Computed capacitive reactance magnitude from Z and R.")
        return _answer(1 / (2 * math.pi * f * C), "Ω", "Computed capacitive reactance.")
    if contract.target == "inductive_reactance" and f and L:
        return _answer(2 * math.pi * f * L, "Ω", "Computed inductive reactance.")
    if contract.target == "impedance":
        xl = _val(q, "X_L") if _val(q, "X_L") is not None else (2 * math.pi * f * L if f and L else None)
        xc = _val(q, "X_C") if _val(q, "X_C") is not None else (1 / (2 * math.pi * f * C) if f and C else None)
        if R is not None and xl is not None and xc is not None:
            return _answer(math.sqrt(R * R + (xl - xc) ** 2), "Ω", "Computed series RLC impedance.")
    if contract.target == "power_factor" and R is not None and Z:
        return _answer(R / Z, None, "Computed power factor cos(phi)=R/Z.")
    if contract.target == "current" and U is not None and Z:
        return _answer(U / Z, "A", "Computed RMS current I=U/Z.")
    if contract.target == "voltage" and R is not None and _val(q, "I") is not None:
        I = _val(q, "I")
        xl = 2 * math.pi * f * L if f and L else 0.0
        impedance = math.sqrt(R * R + xl * xl)
        return _answer(impedance * I, "V", "Computed RMS voltage U=I*sqrt(R^2+X_L^2).")
    if contract.target == "power" and U is not None and R is not None:
        if Z:
            current = U / Z
            return _answer(current * current * R, "W", "Computed active power P=(U/Z)^2*R.")
        return _answer(U * U / R, "W", "At resonance, active power is P=U^2/R.")
    return None


def _solve_lc_energy(contract: DdtContract, q: dict[str, DdtQuantity]) -> DdtAnswer | None:
    L, I = _val(q, "L"), _val(q, "I")
    if L is not None and I is not None:
        return _answer(0.5 * L * I * I, "J", "Computed magnetic energy 1/2 L I^2.")
    C, Q = _cap(q.get("C")), _val(q, "Q")
    if C and Q is not None:
        return _answer(Q / C, "V", "Computed capacitor voltage U=Q/C.")
    return None


def _solve_conceptual(contract: DdtContract) -> DdtAnswer | None:
    answers = {
        "turns_double_field_double": ("Doubled", "B=mu0*(N/l)*I, so B is proportional to N."),
        "ideal_solenoid_external_field_zero": ("Approximately zero", "An ideal long solenoid has negligible external magnetic field."),
        "disconnect_induced_emf_opposes_change": ("An induced electromotive force in the opposite direction appears", "Lenz's law says induced EMF opposes the current change."),
        "unit_inductance_henry": ("Henry (H)", "The SI unit of inductance is the henry."),
        "unit_emf_volt": ("Volt (V)", "Electromotive force is measured in volts."),
        "self_inductance_not_current": ("Current intensity", "Self-inductance depends on geometry and core properties, not current in the linear model."),
    }
    if contract.relation in answers:
        ans, exp = answers[contract.relation]
        return DdtAnswer(ans, None, exp, ["Matched DDT conceptual contract."], 0.9)
    return None


def _val(q: dict[str, DdtQuantity], key: str) -> float | None:
    item = q.get(key)
    return item.value if item is not None else None


def _area(item: DdtQuantity | None) -> float | None:
    if item is None:
        return None
    unit = item.unit.lower()
    if "cm" in unit:
        return item.value * 1e-4
    return item.value


def _cap(item: DdtQuantity | None) -> float | None:
    if item is None:
        return None
    unit = item.unit.lower()
    if "uf" in unit or "µf" in unit or "μf" in unit:
        return item.value * 1e-6
    return item.value


def _answer(value: float, unit: str | None, explanation: str) -> DdtAnswer:
    return DdtAnswer(_format(value), unit, explanation, ["Solved reconciled DDT contract deterministically."], 0.92)


def _format(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-2):
        return f"{value:.6g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")
