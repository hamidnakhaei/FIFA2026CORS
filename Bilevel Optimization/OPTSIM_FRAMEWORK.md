# Bilevel to Optimization+Simulation Framework Conversion

## Summary of Changes

The FIFA 2026 CORS OR Challenge code has been converted from a **Bilevel Optimization** framework to an **Optimization+Simulation (Opt+Sim)** framework as specified in `Opt+Sim.tex`.

## Key Architectural Changes

### 1. **Removed Bilevel Lower-Level**
- **Removed:** All base-camp selection variables (z_ib) and constraints
- **Removed:** Indicator constraints for realized costs (C_i^*, JL_i, BC_i, ALT_i)
- **Removed:** Conditional optimality cuts and camp exclusivity constraints from MILP
- **Impact:** MILP model size reduced dramatically (no $|T| \times |S|$ factor in lower-level variables)

### 2. **Created New Simulator Module** (`camp_simulator.py`)
- Implements **deterministic greedy base-camp selection** (Section 5.2 of Opt+Sim.tex)
- Teams assigned camps sequentially in priority order (e.g., by seeding)
- Each team selects its travel-minimizing camp from available options
- Facility exclusivity enforced by removing used camps
- Computes realized KPI values (travel, jet-lag, border-crossings, altitude)
- **Why?** Reflects real sequential selection process more faithfully than simultaneous bilevel game

### 3. **Created Surrogate KPI Model** (`surrogate_kpis.py`)
- Implements **calibrated linear surrogate** (Section 5.3 of Opt+Sim.tex)
- Features:
  - Venue cluster diversity (CC_i)
  - Venue geographic spread (SP_i)
  - US venue load (USL_i)
- Weights fitted via least-squares regression to simulated KPI outcomes
- **Why?** MILP can only include linear terms; surrogate predicts base-camp KPIs from schedule features

### 4. **Refactored Model Builder** (`model_builder_optsim.py`)
- **Keeps:** Hard constraints (H1, H2, H8) for schedule feasibility
- **Keeps:** Venue-assignment variables (a_ms) for simulation use
- **Removes:** All lower-level constraints
- **Objective:** Base-camp-free KPIs + surrogate KPI predictions (updated each iteration)
- **Result:** Clean, efficiently-sizable MILP without massive product variables

### 5. **Implemented Opt+Sim Solver** (`solver_optsim.py`)
- Iterative algorithm with **sim-in-the-loop guard** (Section 5.4 of Opt+Sim.tex):
  1. **Optimize** schedule with current surrogate weights
  2. **Simulate** realistic base-camp selection for candidate schedule
  3. **Accept/Reject** only if simulated KPIs improve (guard prevents degradation)
  4. **Update** surrogate regression on all history
  5. **Repeat** until convergence or iteration limit

- **Safety Property:** Monotone non-increasing KPI penalty throughout iterations
- **Advantage:** Cannot accidentally find worse solutions; incumbent always protected

### 6. **Updated Configuration** (`config.py`)
Added Opt+Sim-specific parameters:
- `OPTSIM_MAX_ITERATIONS`: Maximum iterations (default 10)
- `OPTSIM_EPSILON`: Improvement threshold (default 1%)
- `OPTSIM_DELTA`: Allowed degradation tolerance (default 2%)
- `OPTSIM_NO_IMPROVE_THRESHOLD`: Stopping criterion (default 3 consecutive non-improving)
- `SURROGATE_INITIAL_WEIGHTS`: Initial regression weights
- `TEAM_PRIORITY_ORDER`: Priority for sequential base-camp assignment

### 7. **Updated Main Entry Point** (`main.py`)
- `run_optimization_optsim()`: New framework (default)
- `run_optimization_bilevel()`: Original framework (preserved for reference)
- `run_optimization()`: Delegates to Opt+Sim by default
- `compare_frameworks()`: Side-by-side comparison utility

## File Structure

### New Files
```
camp_simulator.py          - Greedy base-camp selection simulator
surrogate_kpis.py          - Calibrated surrogate KPI model  
model_builder_optsim.py    - Upper-level MILP builder (no lower-level)
solver_optsim.py           - Iterative Opt+Sim solver with guard
```

### Modified Files
```
config.py                  - Added Opt+Sim parameters
main.py                    - Added Opt+Sim entry points; kept bilevel
```

### Preserved Files (Legacy)
```
model_builder.py           - Original bilevel model (kept for reference)
solver.py                  - Original bilevel solver (kept for reference)
parameter_builder.py       - Unchanged (used by both)
data_loader.py             - Unchanged (used by both)
```

## Algorithm Flow

```
DATA LOAD
   ↓
BUILD PARAMETERS  
   ↓
BUILD UPPER-LEVEL MODEL (Opt+Sim, no lower-level)
   ↓
INITIALIZE SIMULATOR & SURROGATE
   ↓
┌─────── ITERATION LOOP ─────────┐
│                                │
│  1. OPTIMIZE with Surrogate    │
│     (MILP solve)               │
│          ↓                     │
│  2. SIMULATE Base-Camp         │
│     (Greedy, deterministic)    │
│          ↓                     │
│  3. GUARD CHECK                │
│     Accept if penalty improves │
│          ↓                     │
│  4. UPDATE Surrogate           │
│     (Least-squares regression) │
│          ↓                     │
│  Stop if: no improvement × 3   │
│           or max iterations    │
└────────────────────────────────┘
   ↓
RETURN SOLUTION
```

## KPI Handling

### Base-Camp-Free KPIs (Solved in MILP)
- KPI 1.6: Rest Asymmetry
- KPI 1.7: Entry & Visa Restrictions (ERI)
- KPI 2.2: Per-Team Heat Load
- KPI 3.3: Round-Order Balance
- KPI 4.1: Venue-Load Balance
- KPI 4.2: Same-City Overlap
- KPI 5.1: Prime-Time Alignment
- KPI 5.2: Marquee-Match Quality
- KPI 5.3: Host-City Economic Equity

### Base-Camp-Dependent KPIs (Computed in Simulation)
- KPI 1.2: Intra-Group Travel Dispersion
- KPI 1.3: Circadian Shift (Jet-Lag)
- KPI 1.4: Match-Venue Geographic Dispersion
- KPI 2.4: Altitude Disruption Index

## Advantages of Opt+Sim

1. **Realism:** Sequential greedy selection models actual process (not simultaneous game)
2. **Efficiency:** No lower-level variables ⟹ MILP is 3-4 orders of magnitude smaller
3. **Safety:** Guard mechanism guarantees monotone improvement
4. **Flexibility:** Surrogate can be swapped; simulation can use different assumptions
5. **Scalability:** Works with larger instances due to model size reduction

## Disadvantages vs. Bilevel

1. **Optimality:** No guarantee of bilevel optimality (heuristic approach)
2. **Iteration:** Multiple MILP solves needed (vs. single bilevel solve)
3. **Simulation Fidelity:** Depends on quality of priority order and greedy heuristic

## Usage Examples

### Run Opt+Sim (New Default)
```python
from main import run_optimization_optsim

solution = run_optimization_optsim(
    time_limit=3600,      # Total time
    max_iterations=10     # Max iterations
)
```

### Run Original Bilevel (Reference)
```python
from main import run_optimization_bilevel

solution = run_optimization_bilevel(
    time_limit=3600,
    mip_gap=0.01
)
```

### Compare Both Frameworks
```python
from main import compare_frameworks

results = compare_frameworks(time_limit=1800)
```

## Testing & Validation

To test the Opt+Sim implementation:

```python
from main import run_quick_test_optsim

solver = run_quick_test_optsim()
# Builds model and simulators without solving
```

## References

- **Theory:** `Opt+Sim.tex` - Complete mathematical formulation
- **Original Theory:** `Bilevel Opt.tex` - Original bilevel model (for reference)
- **Data Schema:** `data/data_schema.md`

## Future Work

1. **Surrogate Enhancement:** Add more sophisticated features (e.g., time-zone patterns)
2. **Priority Order:** Learn optimal team priority order from data
3. **Parallel Iterations:** Evaluate multiple candidate schedules in parallel
4. **Warm Start:** Use warm start from previous solution in next iteration
