"""
Configuration and constants for FIFA 2026 optimization.
"""

class config_params:

    def __init__(self):

        # KPI Weights (from Algorithm.tex - 13 selected KPIs)
        self.KPI_WEIGHTS = {
            "kpi_1_2": 1.00,   # Travel distance
            "kpi_1_3": 0.08,   # Jet lag
            "kpi_1_4": 0.07,   # Geographic Dispersion
            "kpi_1_6": 0.05,   # Rest Asymmetry 
            "kpi_1_7": 0.05,   # US entry/visa exposure
            "kpi_2_2": 0.12,   # Heat Load
            "kpi_3_3": 0.08,   # Round-Order Balance Index (First-Mover)
            "kpi_4_1": 0.10,   # Venue-Load Balance
            "kpi_4_2": 0.08,   # Fan Accessibility and Same-City Overlap
            "kpi_5_2": 0.06,   # Marquee-Match Slot Quality and Overlap Penalty
            "kpi_5_3": 0.04,   # Host-City Economic Equity
        }

        # Hard constraints constants
        self.MATCH_COUNT = 72
        self.TEAM_COUNT = 48
        self.GROUP_COUNT = 12
        self.STADIUM_COUNT = 16
        self.MATCHES_PER_TEAM = 3
        self.MATCHES_PER_GROUP = 6

        self.COUNTRY_MATCH_ALLOCATION = {
            "USA": 52,
            "MEX": 10,
            "CAN": 10,
        }

        self.US_VISA_BAN_TEAMS = ["IRN", "HAI", "SEN"]
        self.US_VISA_BOND_TEAMS = ["ALG", "CPV", "TUN"]  

        # Time constants
        self.MIN_REST_HOURS = 72  # Minimum rest between matches (kickoff to kickoff)
        self.MIN_MATCH_INTERVAL_HOURS = 5  # Minimum time between two matches at the same stadium for rest and logistics
        self.MATCH_DURATION_HOURS = 105 / 60  # Match duration in hours

        # Optimization parameters
        self.DEFAULT_STEP_A_TIME_LIMIT = 300  # seconds
        self.DEFAULT_STEP_B_TIME_LIMIT = 300  # seconds
        self.DEFAULT_MAX_OUTER_ITERATIONS = 5

        # Solver settings
        self.DEFAULT_SOLVER = "glpk"  # Options: glpk, gurobi, cbc
        self.SOLVER_OPTIONS = {
            "glpk": {"tmlim": 300},
            "gurobi": {"TimeLimit": 300, "Threads": 4},
            "cbc": {"sec": 300},
        }

        # Data paths
        self.DATA_DIR = "data"
        self.SOLUTION_OUTPUT_DIR = "solutions"
