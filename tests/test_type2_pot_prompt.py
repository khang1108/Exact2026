from __future__ import annotations

from exact.type2.extraction.llm_structured import _build_pot_messages


def test_pot_prompt_includes_json_few_shot_vector_example() -> None:
    messages = _build_pot_messages(
        "Three equal charges are placed at the vertices of an equilateral triangle.",
        "Use formulas.",
        "net_force_equal_coulomb_equilateral",
    )
    prompt = "\n".join(str(message["content"]) for message in messages)

    assert "net_force_equal_coulomb_equilateral" in prompt
    assert "resultant is sqrt(3) times one Coulomb force" in prompt
    assert "Do not import numpy" in prompt
    assert "do not use Python triple quotes" in prompt
