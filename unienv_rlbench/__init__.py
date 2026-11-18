"""RLBench adaptor for UniEnv.

``space_metadata`` only needs unienv. ``FromRLBenchEnv`` needs RLBench and PyRep
(CoppeliaSim), so it is imported on first access.
"""
from .space_metadata import *
from .space_metadata import __all__ as _space_metadata_all

__all__ = ["FromRLBenchEnv", *_space_metadata_all]


def __getattr__(name):
    if name == "FromRLBenchEnv":
        try:
            from .env_wrapper import FromRLBenchEnv
        except ImportError as error:
            raise ImportError(
                "FromRLBenchEnv needs RLBench, PyRep and CoppeliaSim 4.1, which are not on PyPI: "
                "see https://github.com/UniEnvOrg/unienv-rlbench#installation"
            ) from error
        return FromRLBenchEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
