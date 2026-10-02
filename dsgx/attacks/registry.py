"""Attack registry. An attack maps an MCQ item to the prompt string the model sees.

Experiment branches register attacks with `@register("name")`. Attacks that produce hazardous
text must write it only under $DSG_PRIVATE (rule 3.8); prompts are not stored in items.parquet.
"""
from dsgx.data.mcq import format_prompt

ATTACKS = {}


def register(name):
    def deco(cls):
        ATTACKS[name] = cls
        cls.name = name
        return cls
    return deco


def make_attack(cfg: dict | None, seed: int = 0):
    import importlib

    import dsgx.attacks.transforms  # noqa: F401  (registers the shared attack suite)

    cfg = cfg or {"name": "none"}
    if cfg.get("module"):
        importlib.import_module(cfg["module"])
    name = cfg["name"]
    if name not in ATTACKS:
        raise KeyError(f"unknown attack {name!r}; known: {sorted(ATTACKS)}")
    return ATTACKS[name](cfg, seed)


class Attack:
    name = "?"

    def __init__(self, cfg, seed=0):
        self.cfg, self.seed = cfg, seed

    def params(self) -> dict:
        return {k: v for k, v in self.cfg.items() if k not in ("name", "_exp_id")}

    def prompt(self, item) -> tuple[str, dict]:
        """Return (prompt, per-item info such as pad_len)."""
        raise NotImplementedError


@register("none")
class NoAttack(Attack):
    def prompt(self, item):
        return format_prompt(item), {"pad_len": 0}
