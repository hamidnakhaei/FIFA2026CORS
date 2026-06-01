"""
Configuration and constants for FIFA 2026 Bilevel Optimization
"""

# =================================================================
# Hard Constraints Parameters
# =================================================================

# Minimum rest between matches (kickoff-to-kickoff in hours)
REST_MIN = 72

# Match duration (in hours)
MATCH_DURATION = 105 / 60

# Stadium turnover window
TURNOVER_WINDOW = REST_MIN + MATCH_DURATION

# Mandated match count per country
MATCH_COUNTS_BY_COUNTRY = {
    "USA": 52,
    "MEX": 10,
    "CAN": 10
}

# Host nation restrictions (teams restricted to specific venues)
HOST_NATIONS = ["MEX", "CAN"]  # Only Mexico and Canada have host-nation restrictions

# =================================================================
# KPI Weights and Normalization
# =================================================================

# Default weights for KPIs (can be adjusted for Pareto sweeps)
KPI_WEIGHTS = {
    1.2: 1.0,  # Intra-Group Travel Dispersion
    1.3: 1.0,  # Circadian Shift Cost (Jet-Lag)
    1.4: 1.0,  # Match-Venue Geographic Dispersion
    1.6: 1.0,  # Rest Asymmetry Between Opponents
    1.7: 1.0,  # Entry & Visa Restriction Exposure
    2.2: 1.0,  # Per-Team Heat Load
    2.4: 1.0,  # Altitude Disruption Index
    3.3: 1.0,  # Round-Order Balance (First-Mover)
    4.1: 1.0,  # Venue-Load Balance
    4.2: 1.0,  # Same-City Overlap & Fan Accessibility
    5.1: 1.0,  # Prime-Time Alignment (to be negated)
    5.2: 1.0,  # Marquee-Match Slot Quality & Overlap
    5.3: 1.0,  # Host-City Economic Equity
}

# =================================================================
# US Entry Restrictions
# =================================================================

# Teams under full US entry ban (cannot use US base camps)
FULL_BAN_TEAMS = []

# Teams under US visa-bond programme (soft penalty for US camps)
VISA_BOND_TEAMS = []

# US visa-bond penalty (km-equivalent)
VISA_BOND_PENALTY = 500

# =================================================================
# Jet-Lag Penalty Function (Piecewise Linear)
# =================================================================
# Perceived kickoff time (hours after midnight, local time at base camp)
# Penalty function φ() in hours
# Typical piecewise linear: penalty increases as kickoff drifts from night/early morning

JET_LAG_PENALTY_TIMES = [0, 3, 6, 9, 12, 15, 18, 21, 24]  # Hour boundaries
JET_LAG_PENALTY_VALUES = [2, 1.5, 1, 0.5, 0.5, 1, 1.5, 2, 2]  # Penalties (hours equivalent)

# =================================================================
# Heat Load (WBGT) Thresholds
# =================================================================

WBGT_HEAT_THRESHOLD = 28  # Degrees Celsius

# =================================================================
# Altitude Disruption Function
# =================================================================

ALTITUDE_THRESHOLD = 500  # Meters
ALTITUDE_PENALTY_SCALE = 1000  # Denominator for penalty calculation

# =================================================================
# Big-M values
# =================================================================

# For conditional optimality cuts (base camp selection)
BIG_M_CAMP_EXCLUSIVITY = 100000  # Large enough to deactivate constraints

# For order constraints
BIG_M_ORDER = 10000

# For rest gap count
BIG_M_REST_GAP = 200  # Upper bound on rest difference (in hours)

# =================================================================
# Model Settings
# =================================================================

GUROBI_TIME_LIMIT = 3600  # Seconds
GUROBI_MIP_GAP = 0.01  # 1% optimality gap
GUROBI_NUM_THREADS = 4

# Verbosity: 0=off, 1=warning, 2=normal, 3=detailed, 4=maximum
GUROBI_VERBOSITY = 2

# =================================================================
# Data Paths
# =================================================================

DATA_DIR = "data"
MATCHES_FILE = f"{DATA_DIR}/matches.csv"
VENUES_FILE = f"{DATA_DIR}/venues.csv"
TEAMS_FILE = f"{DATA_DIR}/teams.csv"
BASE_CAMPS_FILE = f"{DATA_DIR}/base_camps.csv"
WEATHER_FILE = f"{DATA_DIR}/weather.csv"
BROADCAST_FILE = f"{DATA_DIR}/broadcast_markets.csv"
