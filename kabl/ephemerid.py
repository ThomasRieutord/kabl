#!/usr/bin/python
# -*-coding:utf-8 -*-
"""
Copyright (C) 2019  Michel Landers <https://michelanders.nl>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from math import cos, sin, acos, asin, tan
from math import degrees as deg, radians as rad
from datetime import date, datetime, time


class Sun:
    """  
 Calculate sunrise and sunset based on equations from NOAA 
 http://www.srrb.noaa.gov/highlights/sunrise/calcdetails.html 
 
 typical use, calculating the sunrise at the present day: 
  
 import datetime 
 import sunrise 
 s = sun(lat=49,long=3) 
 print('sunrise at ',s.sunrise(when=datetime.datetime.now()) 
 """

    def __init__(self, lat=52.37, long=4.90):  # default Amsterdam
        self.lat = lat
        self.long = long

    def sunrise(self, when=None):
        """ 
  return the time of sunrise as a datetime.time object 
  when is a datetime.datetime object. If none is given 
  a local time zone is assumed (including daylight saving 
  if present) 
  """
        if when is None:
            when = datetime.now()
        self.__preptime(when)
        self.__calc()
        return Sun.__timefromdecimalday(self.sunrise_t)

    def sunset(self, when=None):
        if when is None:
            when = datetime.now()
        self.__preptime(when)
        self.__calc()
        return Sun.__timefromdecimalday(self.sunset_t)

    def solarnoon(self, when=None):
        if when is None:
            when = datetime.now()
        self.__preptime(when)
        self.__calc()
        return Sun.__timefromdecimalday(self.solarnoon_t)

    @staticmethod
    def __timefromdecimalday(day):
        """ 
  returns a datetime.time object. 
   
  day is a decimal day between 0.0 and 1.0, e.g. noon = 0.5 
  """
        hours = 24.0 * day
        h = int(hours)
        minutes = (hours - h) * 60
        m = int(minutes)
        seconds = (minutes - m) * 60
        s = int(seconds)
        return time(hour=h, minute=m, second=s)

    def __preptime(self, when):
        """ 
  Extract information in a suitable format from when,  
  a datetime.datetime object. 
  """
        # datetime days are numbered in the Gregorian calendar
        # while the calculations from NOAA are distibuted as
        # OpenOffice spreadsheets with days numbered from
        # 1/1/1900. The difference are those numbers taken for
        # 18/12/2010
        self.day = when.toordinal() - (734124 - 40529)
        t = when.time()
        self.time = (t.hour + t.minute / 60.0 + t.second / 3600.0) / 24.0

        # TO BE MODIFIED if changing timezone
        self.timezone = 0
        offset = when.utcoffset()
        if not offset is None:
            self.timezone = offset.seconds / 3600.0

    def __calc(self):
        """ 
  Perform the actual calculations for sunrise, sunset and 
  a number of related quantities. 
   
  The results are stored in the instance variables 
  sunrise_t, sunset_t and solarnoon_t 
  """
        timezone = self.timezone  # in hours, east is positive
        longitude = self.long  # in decimal degrees, east is positive
        latitude = self.lat  # in decimal degrees, north is positive

        time = self.time  # percentage past midnight, i.e. noon  is 0.5
        day = self.day  # daynumber 1=1/1/1900

        Jday = day + 2415018.5 + time - timezone / 24  # Julian day
        Jcent = (Jday - 2451545) / 36525  # Julian century

        Manom = 357.52911 + Jcent * (35999.05029 - 0.0001537 * Jcent)
        Mlong = 280.46646 + Jcent * (36000.76983 + Jcent * 0.0003032) % 360
        Eccent = 0.016708634 - Jcent * (0.000042037 + 0.0001537 * Jcent)
        Mobliq = (
            23
            + (
                26
                + ((21.448 - Jcent * (46.815 + Jcent * (0.00059 - Jcent * 0.001813))))
                / 60
            )
            / 60
        )
        obliq = Mobliq + 0.00256 * cos(rad(125.04 - 1934.136 * Jcent))
        vary = tan(rad(obliq / 2)) * tan(rad(obliq / 2))
        Seqcent = (
            sin(rad(Manom)) * (1.914602 - Jcent * (0.004817 + 0.000014 * Jcent))
            + sin(rad(2 * Manom)) * (0.019993 - 0.000101 * Jcent)
            + sin(rad(3 * Manom)) * 0.000289
        )
        Struelong = Mlong + Seqcent
        Sapplong = Struelong - 0.00569 - 0.00478 * sin(rad(125.04 - 1934.136 * Jcent))
        declination = deg(asin(sin(rad(obliq)) * sin(rad(Sapplong))))

        eqtime = 4 * deg(
            vary * sin(2 * rad(Mlong))
            - 2 * Eccent * sin(rad(Manom))
            + 4 * Eccent * vary * sin(rad(Manom)) * cos(2 * rad(Mlong))
            - 0.5 * vary * vary * sin(4 * rad(Mlong))
            - 1.25 * Eccent * Eccent * sin(2 * rad(Manom))
        )

        hourangle = deg(
            acos(
                cos(rad(90.833)) / (cos(rad(latitude)) * cos(rad(declination)))
                - tan(rad(latitude)) * tan(rad(declination))
            )
        )

        self.solarnoon_t = (720 - 4 * longitude - eqtime + timezone * 60) / 1440
        self.sunrise_t = self.solarnoon_t - hourangle * 4 / 1440
        self.sunset_t = self.solarnoon_t + hourangle * 4 / 1440


########################
#      TEST BENCH      #
########################
# Launch with
# >> python ephemerid.py
#
# For interactive mode
# >> python -i ephemerid.py
#
if __name__ == "__main__":
    # Coord Meteopole (Toulouse, France)
    la = 43.57627
    lo = 1.37822
    s = Sun(lat=la, long=lo)
    print("Today is ", datetime.today())
    print("Sunrise:", s.sunrise(), "Solarnoon:", s.solarnoon(), "Sunset:", s.sunset())
