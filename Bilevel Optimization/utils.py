"""
Utility functions for FIFA 2026 Bilevel Optimization
"""

import math
import numpy as np
from datetime import datetime, timedelta
from config import JET_LAG_PENALTY_TIMES, JET_LAG_PENALTY_VALUES, ALTITUDE_THRESHOLD, ALTITUDE_PENALTY_SCALE


def great_circle_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance in kilometers between two points.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (degrees)
        lat2, lon2: Latitude and longitude of second point (degrees)
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def round_trip_distance(camp_lat, camp_lon, venues_lats, venues_lons):
    """
    Calculate total round-trip distance from camp to three venues and back.
    
    Args:
        camp_lat, camp_lon: Base camp coordinates
        venues_lats, venues_lons: Lists of venue coordinates (3 venues per team)
    
    Returns:
        Total round-trip distance in kilometers (2 * sum of distances)
    """
    total_distance = 0
    for v_lat, v_lon in zip(venues_lats, venues_lons):
        total_distance += great_circle_distance(camp_lat, camp_lon, v_lat, v_lon)
    
    # Round trip (there and back)
    return 2 * total_distance


def perceived_kickoff_time(local_time_hour, venue_tz, camp_tz):
    """
    Calculate perceived kickoff time at the base camp (adjusted for time zone).
    
    Args:
        local_time_hour: Kickoff time in hours (0-24) local time at venue
        venue_tz: UTC offset of venue (hours)
        camp_tz: UTC offset of base camp (hours)
    
    Returns:
        Perceived kickoff time in hours (0-24) at the base camp
    """
    # Convert to UTC
    utc_hour = (local_time_hour - venue_tz) % 24
    
    # Convert to base camp local time
    perceived_hour = (utc_hour + camp_tz) % 24
    
    return perceived_hour


def jet_lag_penalty(perceived_hour):
    """
    Calculate jet-lag penalty based on perceived kickoff time.
    Uses piecewise linear interpolation based on config.
    
    Args:
        perceived_hour: Perceived local kickoff time in hours (0-24)
    
    Returns:
        Jet-lag penalty in hours (equivalent)
    """
    # Ensure hour is in [0, 24)
    hour = perceived_hour % 24
    
    # Find the segment
    for i in range(len(JET_LAG_PENALTY_TIMES) - 1):
        t1, t2 = JET_LAG_PENALTY_TIMES[i], JET_LAG_PENALTY_TIMES[i + 1]
        p1, p2 = JET_LAG_PENALTY_VALUES[i], JET_LAG_PENALTY_VALUES[i + 1]
        
        if t1 <= hour < t2:
            # Linear interpolation
            penalty = p1 + (p2 - p1) * (hour - t1) / (t2 - t1)
            return penalty
    
    # If hour >= last time (should be 24), wrap to first penalty
    return JET_LAG_PENALTY_VALUES[0]


def altitude_disruption(venue_elev, camp_elev):
    """
    Calculate altitude disruption penalty.
    
    Args:
        venue_elev: Elevation at venue (meters)
        camp_elev: Elevation at base camp (meters)
    
    Returns:
        Altitude disruption penalty
    """
    elev_diff = abs(venue_elev - camp_elev)
    
    if elev_diff <= ALTITUDE_THRESHOLD:
        return 0
    else:
        return (elev_diff - ALTITUDE_THRESHOLD) / ALTITUDE_PENALTY_SCALE


def parse_time(time_str):
    """Parse time string in HH:MM format to float hours."""
    h, m = map(int, time_str.split(':'))
    return h + m / 60


def parse_datetime(date_str, time_str):
    """Parse date and time strings to datetime object."""
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")


def countries_differ(country1, country2):
    """Check if two countries are different (for border-crossing count)."""
    return country1.upper() != country2.upper()


def calculate_rest_hours(kickoff_1, kickoff_2):
    """
    Calculate rest time in hours between two consecutive kickoffs.
    Kickoffs are datetime objects.
    
    Args:
        kickoff_1: First kickoff datetime
        kickoff_2: Second kickoff datetime (should be after first)
    
    Returns:
        Rest time in hours
    """
    if kickoff_2 < kickoff_1:
        raise ValueError("Second kickoff must be after first kickoff")
    
    rest = (kickoff_2 - kickoff_1).total_seconds() / 3600
    return rest


def indicator_same_city_same_day(venue_id_1, venue_id_2, date_1, date_2, venue_data):
    """
    Check if two matches are in the same city on the same day.
    
    Args:
        venue_id_1, venue_id_2: Venue identifiers
        date_1, date_2: Match dates
        venue_data: Dictionary mapping venue_id to venue info
    
    Returns:
        1 if same city and same day, 0 otherwise
    """
    if date_1 != date_2:
        return 0
    
    city_1 = venue_data[venue_id_1]["city"]
    city_2 = venue_data[venue_id_2]["city"]
    
    return 1 if city_1.upper() == city_2.upper() else 0


def wbgt_excess_heat_load(wbgt, threshold=28):
    """
    Calculate excess heat load above threshold.
    
    Args:
        wbgt: Wet Bulb Globe Temperature
        threshold: Heat threshold (default 28°C)
    
    Returns:
        max(0, wbgt - threshold)
    """
    return max(0, wbgt - threshold)


def clamp(value, min_val, max_val):
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def normalize_angle_hour(hour):
    """Normalize an hour value to [0, 24)."""
    return hour % 24


def broadcast_quality(slot_hour, primetime_start, primetime_end, audience_weight):
    """
    Calculate broadcast quality for a given slot in a market.
    Higher quality if slot overlaps with prime time.
    
    Args:
        slot_hour: Slot time in UTC (float hours)
        primetime_start: Prime time start in market local time (float hours)
        primetime_end: Prime time end in market local time (float hours)
        audience_weight: Population/viewership weight
    
    Returns:
        Broadcast quality score (0 to audience_weight)
    """
    # This is a simplified version; actual implementation may need
    # to account for time zone conversions between UTC and market local time
    
    # For now, assume overlap is proportional to distance from prime time
    if primetime_start <= slot_hour <= primetime_end:
        return audience_weight
    else:
        # Penalty for off-prime-time
        return audience_weight * 0.5


def mean_absolute_difference(values):
    """
    Calculate mean absolute difference of a list of values.
    (Used as a surrogate for Gini coefficient or CV)
    
    Args:
        values: List of numerical values
    
    Returns:
        Mean absolute difference
    """
    if len(values) <= 1:
        return 0
    
    values = np.array(values)
    mad = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            mad += abs(values[i] - values[j])
    
    # Average pairwise difference
    n = len(values)
    return mad / (n * (n - 1) / 2) if n > 1 else 0


def read_csv_safe(filepath):
    """
    Safely read a CSV file and return data.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        List of dictionaries (one per row)
    """
    import csv
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
