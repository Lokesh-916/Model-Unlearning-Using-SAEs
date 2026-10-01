"""Method registry. A method installs hooks on a shared model bundle and records gate stats.

Experiment branches register new methods with `@register("name")`.
"""
from dsgx.data import activation_cache as ac
from dsgx.methods import dsg

METHODS = {}


def register(name):
    def deco(cls):
        METHODS[name] = cls
        cls.name = name
        return cls
    return deco


def make_method(cfg: dict, bundle, seed: int = 0):
    import importlib

    import dsgx.methods.gates  # noqa: F401  (registers gated / composite)

    if cfg.get("module"):  # experiment branches register their own methods
        importlib.import_module(cfg["module"])
    name = cfg["name"]
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; known: {sorted(METHODS)}")
    return METHODS[name](cfg, bundle, seed)


class Method:
    name = "?"
    records_gate = False

    def __init__(self, cfg, bundle, seed=0):
        self.cfg, self.bundle, self.seed = cfg, bundle, seed
        self.info = {}

    def install(self):
        self.bundle.model.reset_hooks()

    def remove(self):
        self.bundle.model.reset_hooks()

    def set_lengths(self, lengths):
        pass

    def pop_records(self):
        return []


@register("base")
class Base(Method):
    """The unguarded model."""


class _DSG(Method):
    faithful = True
    records_gate = True

    def __init__(self, cfg, bundle, seed=0):
        super().__init__(cfg, bundle, seed)
        c = dict(cfg)
        cache = ac.open_cache(bundle.model_name, bundle.sae_release, bundle.sae_id,
                              c.get("forget_corpus", "bio-forget-corpus"),
                              c.get("retain_corpus", "wikitext"), c.get("calib_seed", seed),
                              c.get("dataset_size", 1024), c.get("seq_len", 1024))
        if "features" in c:
            feats = list(c["features"])
        else:
            feats = dsg.select_features(cache, int(c.get("n_features", 20)),
                                        float(c.get("retain_pct", 95)),
                                        round_like_txt=c.get("round_like_txt", True))
        tau = c.get("tau")
        if tau is None:
            tau = dsg.calibrate_tau(cache, feats, float(c.get("tau_pct", 95)),
                                    exclude_special=not self.faithful)
        self.features, self.tau = feats, float(tau)
        self.multiplier = float(c.get("multiplier", 500))
        self.hook = dsg.DSGHook(bundle.sae, feats, self.multiplier, self.tau, faithful=self.faithful)
        self.info = {"features": feats, "tau": self.tau, "multiplier": self.multiplier,
                     "cache": str(cache.path), "faithful": self.faithful}

    def install(self):
        self.bundle.model.reset_hooks()
        self.bundle.model.add_hook(self.bundle.hook_name, self.hook)

    def set_lengths(self, lengths):
        self.hook.lengths = lengths

    def pop_records(self):
        return self.hook.pop_records()


@register("dsg-faithful")
class DSGFaithful(_DSG):
    faithful = True


@register("dsg-fixed")
class DSGFixed(_DSG):
    faithful = False
