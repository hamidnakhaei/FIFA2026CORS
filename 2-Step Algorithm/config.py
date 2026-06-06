"""
Configuration and constants for FIFA 2026 optimization.
"""

class config_params:

    def __init__(self):

        # KPI Weights (from Algorithm.tex - 13 selected KPIs)
        # NOTE: Travel weight increased to 15.0 from 6.0 to prioritize geographic clustering
        self.KPI_WEIGHTS = {
            "kpi_1_2": 15.00,  # Travel distance (increased from 6.0 to reduce scattering)
            "kpi_2_2": 5.00,   # Heat Load
            "kpi_4_1": 5.00,   # Venue-Load Balance
            "kpi_5_2": 10.00,   # Rest Time
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
        self.DEFAULT_SOLVER = "gurobi"  # Options: glpk, gurobi, cbc
        self.SOLVER_OPTIONS = {
            "glpk": {"tmlim": 300},
            "gurobi": {"TimeLimit": 300, "Threads": 4},
            "cbc": {"sec": 300},
        }

        # Data paths
        self.DATA_DIR = "data"
        self.SOLUTION_OUTPUT_DIR = "solutions"
