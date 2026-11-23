import pytest
import yaml

from nasdaq_signal.config import ConfigError, load_config


def _write_modified(tmp_path, mutate):
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    mutate(cfg)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


def test_loads_default_config():
    cfg = load_config()
    assert cfg["horizons"]["mid"]["days"] == 63
    assert cfg["indicators"]["vix"]["horizons"]["mid"]["transform"] == "contrarian"


def test_category_weights_sum_to_one_per_horizon():
    cfg = load_config()
    for h in cfg["horizons"]:
        assert sum(w[h] for w in cfg["category_weights"].values()) == pytest.approx(1.0)


def test_rejects_unknown_category(tmp_path):
    p = _write_modified(tmp_path, lambda c: c["indicators"]["vix"].update(category="foo"))
    with pytest.raises(ConfigError):
        load_config(p)


def test_rejects_weights_not_summing_to_one(tmp_path):
    p = _write_modified(tmp_path, lambda c: c["category_weights"]["market"].update(short=0.35))
    with pytest.raises(ConfigError):
        load_config(p)


def test_accepts_exclude_override(tmp_path):
    p = _write_modified(tmp_path, lambda c: c["indicators"]["curve"].update(horizons={"short": {"exclude": True}}))
    assert load_config(p)["indicators"]["curve"]["horizons"]["short"]["exclude"] is True


def test_rejects_non_bool_exclude(tmp_path):
    p = _write_modified(tmp_path, lambda c: c["indicators"]["curve"].update(horizons={"short": {"exclude": "yes"}}))
    with pytest.raises(ConfigError):
        load_config(p)
