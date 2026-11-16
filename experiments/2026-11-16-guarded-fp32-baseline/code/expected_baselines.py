"""Resolve audited baseline runtime schemas without importing CUDA packages."""
import importlib.util
from pathlib import Path

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def inherited(repository):
    repo=Path(repository)
    return (module('full_range_expected',repo/'experiments/2026-11-15-range-worker-sweep/code/expected.py'),module('full_mono_expected',repo/'experiments/2026-11-14-range-bitpack-paired/code/gpusparse-u64/expected_config.py'))

