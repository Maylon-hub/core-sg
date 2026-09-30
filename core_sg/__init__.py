from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("core-sg-mustache")
except PackageNotFoundError:
    __version__ = "0+unknown"

from .core_sg import CoreSG
from .estimators import CoreSGClusterer

__all__ = ["CoreSG", "CoreSGClusterer"]
