from dsgx.labels import reference_label


def test_rmu_reference_is_labelled_unverified():
    cfg = {"model": {"name": "gemma-2-2b-it", "weights": "AMindToThink/gemma-2-2b-it_RMU_s200_a300_layer3"}}
    assert reference_label(cfg).startswith("unverified reference")


def test_main_runs_have_no_label():
    assert reference_label({"model": {"name": "gemma-2-2b-it"}}) is None
    assert reference_label({}) is None
    assert reference_label(None) is None
