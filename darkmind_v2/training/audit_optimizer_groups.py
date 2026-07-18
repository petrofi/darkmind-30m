"""Run the Phase 4C Base V1 optimizer parameter-group audit."""

from __future__ import annotations

import json

from darkmind_v2.training.phase4c_diagnostics import optimizer_group_audit


def main() -> None:
    payload = optimizer_group_audit()
    print(json.dumps({key: value for key, value in payload.items() if key != "parameters"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
