"""SOAPify is a support module for helping the user in calculating SOAP fingerprints

SOAPify contains submodules for a basic time analysis of a trajecory of classifications
and for calculating the SOAP fingerprints using the soap engine from quippy or dscribe
"""


from .classify import *
from .distances import *
from .saponify import *
from .utils import *
from .transitions import *
from .engine import *
from .analysis import *

__version__ = "v0.1.0"
