#!/usr/bin/python
# -*-coding:utf-8 -*-
"""
Unitary tests of the functions in ../kabl/ephemerid.py
Must be executed inside the `tests/` directory
"""

import datetime as dt
from kabl.ephemerid import Sun

def test_ephemerid_5june2020():
    # Coord Meteopole (Toulouse, France)
    la = 43.57627
    lo = 1.37822
    t = dt.datetime(2020,6,5,12)
    s = Sun(lat=la, long=lo)
    
    assert s.sunrise(t) == dt.time(4,14,32) and s.sunset(t) == dt.time(19,31,16)

