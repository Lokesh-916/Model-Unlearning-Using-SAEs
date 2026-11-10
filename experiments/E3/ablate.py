"""Method `zero-ablate`: on every input (forced-on, no gate), set the chosen SAE features to 0 at every position
except BOS, decode, and add the SAE error back (so only the ablated features change the residual stream).

method config: {name: zero-ablate, module: experiments.E3.ablate, features_exp: E3, features_name: <file stem>}
or {..., dsg_features: true} for DSG's own selected features (n_features / retain_pct, from the activation cache).
"""
import torch

from dsgx.data import activation_cache as ac
from dsgx.methods import dsg
from dsgx.methods.registry import Method, register

from experiments.E3.common import load_features


class ZeroAblateHook:
    def __init__(self, sae, features):
        self.sae = sae
        self.features = [int(f) for f in features]
        self.idx = torch.tensor(self.features, dtype=torch.long, device=sae.W_dec.device)

    def __call__(self, resid, hook=None):
        if not self.features:
            return resid
        acts = self.sae.encode(resid)
        error = resid - self.sae.decode(acts)
        acts[:, 1:, self.idx] = 0.0
        return self.sae.decode(acts) + error


@register("zero-ablate")
class ZeroAblate(Method):
    def __init__(self, cfg, bundle, seed=0):
        super().__init__(cfg, bundle, seed)
        c = dict(cfg)
        if c.get("dsg_features"):
            cache = ac.open_cache(ac.model_tag(bundle), bundle.sae_release, bundle.sae_id,
                                  c.get("forget_corpus", "bio-forget-corpus"), c.get("retain_corpus", "wikitext"),
                                  c.get("calib_seed", 0), c.get("dataset_size", 1024), c.get("seq_len", 1024))
            feats = dsg.select_features(cache, int(c.get("n_features", 20)), float(c.get("retain_pct", 95)))
        else:
            feats = load_features(c["features_exp"], c["features_name"])
        self.features = feats
        self.hook = ZeroAblateHook(bundle.sae, feats)
        self.info = {"features": feats, "n_features": len(feats), "intervention": "zero-ablate, forced-on"}

    def install(self):
        self.bundle.model.reset_hooks()
        self.bundle.model.add_hook(self.bundle.hook_name, self.hook)
