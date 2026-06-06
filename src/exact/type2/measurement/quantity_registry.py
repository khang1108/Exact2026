from __future__ import annotations


SUPPORTED_SYSTEM_TYPES = {
    "least_count_error",
    "true_vs_measured_error",
    "direct_uncertainty",
    "repeated_measurement",
    "propagation",
    "circuit_lab_propagation",
}

SUPPORTED_TARGETS = {
    "value",
    "absolute_error",
    "relative_error",
    "percentage_error",
    "mean_value",
    "mean_absolute_error",
    "result_with_uncertainty",
}

SUPPORTED_AST_OPS = {"add", "sub", "mul", "div", "pow", "const"}

