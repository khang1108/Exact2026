#!/usr/bin/env python3
"""Smoke-test the self-hosted parser service with one concurrent FOL batch."""

from __future__ import annotations

import asyncio

from exact.type1.parser import FOLParser, build_parser_client_from_settings


SMOKE_TEST_PREMISES = [
    "Alice studies logic.",
    "Every student studies hard.",
    "If a student studies hard, then the student passes.",
]


async def main() -> None:
    """Connect to the configured parser server and print parsed FOL trees."""

    client = build_parser_client_from_settings()
    if client is None:
        raise RuntimeError("EXACT_TYPE1_PARSER_BASE_URL is not configured")

    try:
        parser = FOLParser(client)
        trees = await parser.parse_many(SMOKE_TEST_PREMISES)
        for premise, tree in zip(SMOKE_TEST_PREMISES, trees, strict=True):
            print(f"NL:  {premise}")
            print(f"FOL: {tree}\n")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
