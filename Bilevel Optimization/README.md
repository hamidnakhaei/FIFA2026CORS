# FIFA 2026 Bilevel Optimization - Modular Implementation

This directory contains a modular implementation of the bilevel optimization framework for the 2026 FIFA World Cup group-stage scheduling problem, as described in `Bilevel Opt.tex`.

## Problem Overview

**Bilevel Structure:**
- **Upper Level (Leader)**: FIFA chooses the schedule (match → slot × stadium)  
- **Lower Level (Followers)**: Each of 48 teams independently chooses a base camp from an approved set to minimize its round-trip travel cost

**Key Features:**
- Considers all base camps for each team to choose from
- Models base camp exclusivity (one facility hosts at most one team)
- Includes hard scheduling constraints (H1-H8) from the feasibility model
- Implements soft KPIs with normalizing weights
- Uses McCormick linearization for binary products (z_ib × x_mts)
- Handles US entry restrictions (hard bans and soft visa-bond penalties)

## Project Structure

```
├── config.py                    # Configuration constants and parameters
├── utils.py                     # Utility functions (distances, penalties, etc.)
├── data_loader.py               # Load and manage CSV data
├── parameter_builder.py         # Precompute model parameters
├── model_builder.py             # Basic Gurobi model construction
├── model_builder_advanced.py    # Advanced model with full KPI support
├── solver.py                    # Main orchestrator (load → build → solve → extract)
├── output_handler.py            # Export solution to Excel files
├── requirements.txt             # Python dependencies (gurobipy, pandas, numpy)
├── README.md                    # This file
├── output/                      # Output directory (created on first run)
│   ├── optimized_schedule.xlsx  # Optimized schedule
│   └── base_camp_assignments.xlsx # Team base camp choices
└── data/                        # Data files
    ├── matches.csv
    ├── venues.csv
    ├── teams.csv
    ├── base_camps.csv
    ├── weather.csv
    └── broadcast_markets.csv
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
**Note**: Gurobi requires a license (free for academic use at universities)

### 2. Run Full Optimization
```bash
python solver.py
```

### 3. Use in Python
```python
from solver import FIFA2026Solver

solver = FIFA2026Solver()
solution = solver.run_full_pipeline(time_limit=3600, mip_gap=0.01)
solver.print_solution_summary()
solver.save_solution("results.txt")
```

## Module Reference

- **config.py**: Global settings, KPI weights, constraints, solver parameters
- **utils.py**: Great-circle distance, jet-lag, altitude, border-crossing calculations
- **data_loader.py**: Load CSVs, build index structures, query data
- **parameter_builder.py**: Precompute distances, penalties, KPI constants
- **model_builder.py**: Core Gurobi model with hard constraints and lower-level logic
- **model_builder_advanced.py**: Extended model with full KPI implementation
- **solver.py**: Pipeline orchestrator (load → build → solve → extract)
- **output_handler.py**: Export solution to Excel files

## Output Files

After solving, the following Excel files are automatically generated in the `output/` directory:

### 1. `optimized_schedule.xlsx`
Optimized match schedule matching the structure of input `matches.csv`:
- **Columns**: match_id, group, round, team_a_id, team_b_id, venue_id, date, kickoff_local
- **Contents**: 72 matches with optimized venue assignments and times
- **Structure**: Same as input for easy comparison

### 2. `base_camp_assignments.xlsx`
Team base camp assignments from the solution:
- **Columns**: team_id, team_name, base_camp_id, training_site, city, country, lat, lon, utc_offset_june
- **Contents**: 48 teams with their chosen base camp facility
- **Structure**: Enriched version combining team and camp information
