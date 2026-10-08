"""Find the route with the most (or least) climbing from a starting point, on OpenStreetMap trails and roads."""
from .api import Route, SideTrip, Spurred, find_route, read_gpx, spurify
from .core import TOPOLOGIES, OptionError, VertmaxxerError

__version__ = "0.1.0"
__all__ = ["find_route", "spurify", "read_gpx", "Route", "SideTrip", "Spurred", "TOPOLOGIES", "OptionError", "VertmaxxerError"]
