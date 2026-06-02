# FIFA 2026 World Cup Group-Stage Two-Step Optimization Framework

A Python-based optimization framework implementing a two-step iterative algorithm for optimizing the FIFA 2026 World Cup group-stage match schedule and team base camps.

## Overview

The framework solves a coupled optimization problem:
- **Step A (Schedule Optimization)**: Solve a Mixed-Integer Linear Program (MILP) to assign matches to time slots and stadiums, minimizing weighted KPIs
- **Step B (Base Camp Optimization)**: Re-optimize each team's base camp location using simulated annealing with Metropolis exploration

These steps alternate iteratively until convergence.

### Algorithm

The algorithm minimizes a weighted sum of 13 Key Performance Indicators (KPIs):
- **Travel**: Distance, frequency, jet lag, altitude shock
- **Health**: Rest violations, weather (WBGT) exposure
- **Geographic**: Dispersion across venues
- **Broadcast**: Value, equity, strong teams in prime time
- **Venue**: Variety, roofed stadiums

See `Algorithm.tex` for mathematical formulation.

## Project Structure

```
FIFA2026CORS/
├── data/                          # Input data (CSVs)
│   ├── matches.csv
│   ├── venues.csv
│   ├── teams.csv
│   ├── base_camps.csv
│   ├── weather.csv
│   └── broadcast_markets.csv
│
├── data_loader.py                 # Data loading and preprocessing
├── kpis.py                        # KPI calculations (all 13 metrics)
├── schedule_optimizer.py          # Step A: MILP schedule solver
├── base_camp_optimizer.py         # Step B: Simulated annealing for base camps
├── main.py                        # Main algorithm orchestrator
├── utils.py                       # Validation and reporting utilities
├── config.py                      # Configuration and constants
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Module Descriptions

### `data_loader.py`
Loads and preprocesses data from CSV files.
- `DataLoader` class: Main interface for data access
- Methods: `get_matches()`, `get_venues()`, `get_teams()`, `get_base_camps()`, etc.
- Computes sets, indices, and parameters for optimization

**Usage:**
```python
from data_loader import DataLoader
loader = DataLoader(data_dir="data")
data = loader.load_all()
M, T, S, I, G, M_i, M_g, T_r, S_c = loader.get_sets_and_indices()
params = loader.get_parameters()
```

### `kpis.py`
Computes all 13 KPIs based on schedule and base camp assignments.
- `KPICalculator` class: Main KPI computation engine
- Methods: `compute_all_kpis()`, `_kpi_1_2_travel_distance()`, etc.

**Usage:**
```python
from kpis import KPICalculator
kpi_calc = KPICalculator(loader, params)
kpis = kpi_calc.compute_all_kpis(schedule, base_camp_assignment)
```

### `schedule_optimizer.py`
Implements Step A: MILP-based schedule optimization with hard constraints H1-H8.
- `ScheduleOptimizer` class: Builds and solves the schedule MILP
- Constraints: Match once (H1), round-robin (H2), rest (H5), country allocation (H8), etc.

**Usage:**
```python
from schedule_optimizer import ScheduleOptimizer
optimizer = ScheduleOptimizer(loader, kpi_calc, base_camp_assignment)
result = optimizer.solve(time_limit=300, solver_name="glpk")
schedule = result["schedule"]  # Dict: match_id -> (time_slot, stadium)
```

### `base_camp_optimizer.py`
Implements Step B: Simulated annealing for base camp re-optimization.
- `BaseCampOptimizer` class: Metropolis moves and temperature control
- Minimizes KPIs while schedule is fixed

**Usage:**
```python
from base_camp_optimizer import BaseCampOptimizer
optimizer = BaseCampOptimizer(loader, kpi_calc, schedule)
result = optimizer.optimize(
    initial_assignment=base_camp_assignment,
    max_iterations=500,
    temperature_schedule="exponential"
)
best_assignment = result["best_assignment"]
```

### `main.py`
Orchestrates the two-step algorithm with iteration control.
- `TwoStepOptimizer` class: Main algorithm driver
- Runs alternating Steps A and B with convergence checking

**Usage:**
```python
from main import TwoStepOptimizer
optimizer = TwoStepOptimizer(data_dir="data", solver_name="glpk", verbose=True)
result = optimizer.run(
    max_iterations=3,
    step_a_time_limit=300,
    step_b_iterations=500,
    convergence_tol=0.01
)
optimizer.save_solution("best_solution.pkl")
```

### `utils.py`
Solution validation, post-processing, and reporting.
- `SolutionValidator` class: Validates hard constraints H1-H8
- `SolutionReporter` class: Generates reports and exports to CSV

**Usage:**
```python
from utils import SolutionValidator, SolutionReporter
validator = SolutionValidator(loader)
is_valid = validator.validate_all(schedule)

reporter = SolutionReporter(loader)
reporter.print_schedule_summary(schedule)
reporter.export_to_csv(schedule, base_camp_assignment)
```

### `config.py`
Global configuration: KPI weights, constants, optimization parameters.

## Installation

### Prerequisites
- Python 3.8+
- An MILP solver (GLPK, Gurobi, or CBC)

### Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install an MILP solver:
   - **GLPK** (free): `pip install glpk` or use system package manager
   - **Gurobi** (commercial): Download from https://www.gurobi.com
   - **CBC** (free): `pip install coincbc`

3. Verify Pyomo can find the solver:
```python
from pyomo.environ import SolverFactory
solver = SolverFactory("glpk")
```

## Usage

### Quick Start

Run the complete two-step optimization:
```bash
python main.py
```

### Custom Configuration

```python
from main import TwoStepOptimizer

optimizer = TwoStepOptimizer(
    data_dir="data",
    solver_name="glpk",  # or "gurobi", "cbc"
    verbose=True
)

result = optimizer.run(
    max_iterations=5,
    step_a_time_limit=600,   # 10 minutes per Step A solve
    step_b_iterations=1000,  # More SA iterations for better base camps
    convergence_tol=0.001
)

print(f"Best objective: {result['best_objective']:.2f}")
optimizer.save_solution("my_solution.pkl")
```

### Validate a Solution

```python
from utils import SolutionValidator
validator = SolutionValidator(loader)
constraints_ok = validator.validate_all(schedule)
for constraint, valid in constraints_ok.items():
    print(f"{constraint}: {'✓' if valid else '✗'}")
```

### Generate Reports

```python
from utils import SolutionReporter
reporter = SolutionReporter(loader)
reporter.print_schedule_summary(schedule)
reporter.print_base_camp_summary(base_camp_assignment)
reporter.export_to_csv(schedule, base_camp_assignment, output_prefix="fifa2026")
```

## Data Format

All input data is in `data/` folder as CSV files:

| File | Purpose |
|------|---------|
| `matches.csv` | 72 group-stage matches with teams, group, round |
| `venues.csv` | 16 stadiums with location, time zone, cluster |
| `teams.csv` | 48 teams with FIFA ranking, group assignment |
| `base_camps.csv` | ~80 candidate/confirmed base camp facilities |
| `weather.csv` | Hourly temperature at each venue |
| `broadcast_markets.csv` | Prime-time windows and audience weights |

See `data/data_schema.md` for detailed schema.

## Output

The optimization produces:
- `best_solution.pkl`: Pickled solution dict with schedule and base camp assignment
- `output_schedule.csv`: Human-readable match schedule
- `output_base_camps.csv`: Human-readable team base camp assignments

## Key Parameters

### Optimization Control
- `max_iterations`: Maximum outer loop iterations (default: 3)
- `step_a_time_limit`: MILP solver time per iteration in seconds (default: 300)
- `step_b_iterations`: Simulated annealing iterations per base camp optimization (default: 500)
- `convergence_tol`: Stop if objective change < tolerance (default: 0.01)

### Temperature Schedule (Simulated Annealing)
- `exponential`: T(k) = T₀ * exp(-3k/K)
- `linear`: T(k) = T₀ * (1 - k/K)
- `constant`: T(k) = T₀

## Solver Configuration

### GLPK (free, often sufficient)
```python
optimizer = TwoStepOptimizer(solver_name="glpk")
```

### Gurobi (commercial, faster for large instances)
```python
optimizer = TwoStepOptimizer(solver_name="gurobi")
# Requires Gurobi license
```

### CBC (free, alternative)
```python
optimizer = TwoStepOptimizer(solver_name="cbc")
```

## Troubleshooting

### "Solver not found" error
Install a solver: `pip install glpk` or use system package manager.

### MILP solver takes too long
- Reduce `step_a_time_limit` (trade quality for speed)
- Use Gurobi instead of GLPK for larger problems
- Reduce `max_iterations` to test quickly

### Memory issues with large instances
- Run on a machine with more RAM
- Reduce problem size or use sparse representations (requires code modification)

## Algorithm Details

See `Algorithm.tex` for complete mathematical formulation including:
- Sets and indices (M, T, S, I, G, etc.)
- Decision variables (x_mts, y_gt, u_ib)
- Objective function (weighted sum of 13 KPIs)
- Hard constraints (H1-H8)
- Metropolis exploration for base camps

## Performance

Typical runtime (3 iterations, 300s per Step A, 500 SA iterations per Step B):
- **GLPK**: ~30-45 minutes
- **Gurobi**: ~10-15 minutes

## References

- Team UofT Champions (2026). "A Two-Step Iterative Base-Camp Optimisation Model with Stochastic Exploration for the 2026 FIFA World Cup Group-Stage Schedule"
- CORS 2026 OR Challenge

## License

University of Toronto

## Authors

Team UofT Champions, CORS 2026
