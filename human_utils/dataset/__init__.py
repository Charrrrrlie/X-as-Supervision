from .imdb import IMDB

# human pose

from .hm36 import hm36
from .mpi_inf_3dhp import mpi_inf_3dhp
from .mpii import mpii

__all__ = {
    'human36': hm36,
    'mpi_inf_3dhp': mpi_inf_3dhp,
    'mpii': mpii,
}