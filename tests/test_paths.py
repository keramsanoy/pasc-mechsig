import importlib
import os


def test_paths_are_overridable_by_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PASC_COHORT_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("PASC_RESULTS_DIR", str(tmp_path / "r"))
    import pasc.config.paths as paths
    importlib.reload(paths)
    assert paths.COHORT_DIR == tmp_path / "c"
    assert paths.MAIN_RESULTS_DIR == tmp_path / "r" / "main"
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()


def test_schema_default_and_override(monkeypatch):
    import pasc.config.omop as omop
    importlib.reload(omop)
    assert omop.CDM_SCHEMA == "CDMPHI"
    monkeypatch.setenv("OMOP_CDM_SCHEMA", "OMOP_CDM")
    importlib.reload(omop)
    assert omop.CDM_SCHEMA == "OMOP_CDM"
    monkeypatch.delenv("OMOP_CDM_SCHEMA")
    importlib.reload(omop)


def test_connect_refuses_without_configuration(monkeypatch):
    for k in list(os.environ):
        if k.startswith("OMOP_"):
            monkeypatch.delenv(k)
    monkeypatch.chdir("/")  # no .env in the working directory
    import pytest
    from pasc.db import connect
    with pytest.raises(RuntimeError, match="OMOP_DB_HOST"):
        connect()
