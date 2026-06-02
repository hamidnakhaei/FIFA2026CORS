"""
Configuration and constants for FIFA 2026 optimization.
"""

class config_params:

    def __init__(self):

        # KPI Weights (from Algorithm.tex - 13 selected KPIs)
        self.KPI_WEIGHTS = {
            "kpi_1_2": 0.10,   # Travel distance
            "kpi_1_3": 0.08,   # Travel frequency
            "kpi_1_4": 0.08,   # Jet lag
            "kpi_1_6": 0.05,   # Altitude shock
            "kpi_1_7": 0.05,   # US entry/visa exposure
            "kpi_2_2": 0.12,   # Rest violations
            "kpi_2_4": 0.10,   # Weather (WBGT)
            "kpi_3_3": 0.08,   # Geographic dispersion
            "kpi_4_1": 0.10,   # Broadcast value
            "kpi_4_2": 0.08,   # Equal broadcast distribution
            "kpi_5_1": 0.05,   # Venue variety
            "kpi_5_2": 0.06,   # Strong teams broadcast
            "kpi_5_3": 0.04,   # Roofed stadiums
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
        self.MATCH_DURATION_HOURS = 105 / 60  # Match duration in hours

        # Optimization parameters
        self.DEFAULT_STEP_A_TIME_LIMIT = 300  # seconds
        self.DEFAULT_STEP_B_MAX_ITERATIONS = 50
        self.DEFAULT_MAX_OUTER_ITERATIONS = 5
        self.DEFAULT_CONVERGENCE_TOL = 0.01

        # Solver settings
        self.DEFAULT_SOLVER = "glpk"  # Options: glpk, gurobi, cbc
        self.SOLVER_OPTIONS = {
            "glpk": {"tmlim": 300},
            "gurobi": {"TimeLimit": 300, "Threads": 4},
            "cbc": {"sec": 300},
        }

        # Temperature schedule for simulated annealing
        self.TEMPERATURE_SCHEDULES = ["exponential", "linear", "constant"]
        self.DEFAULT_TEMPERATURE_SCHEDULE = "exponential"

        # Data paths
        self.DATA_DIR = "data"
        self.SOLUTION_OUTPUT_DIR = "solutions"
