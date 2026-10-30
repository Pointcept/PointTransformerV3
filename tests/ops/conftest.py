# Adapted from Pointcept2 tests/ops/conftest.py @ 771a949.
"""tests/ops fixtures: torch's compiler configs are restored after every test.

pytest 9's ``monkeypatch.setattr`` records the old value from the target's instance ``__dict__``
and, finding none, undoes the patch with ``delattr``. torch's ``ConfigModule``
(``torch._inductor.config``, ``torch._dynamo.config`` and their sub-configs such as
``inductor_config.triton``) answers ``delattr`` by hiding the entry, so after a test that
monkeypatches a compiler setting every later compile in the process fails with
"torch._inductor.config.<key> does not exist". ``load_config`` of a snapshot assigns every entry
again, which also unhides it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def restore_compiler_configs() -> Iterator[None]:
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    saved = [(module, module.save_config()) for module in (inductor_config, dynamo_config)]
    yield
    for module, snapshot in saved:
        module.load_config(snapshot)
