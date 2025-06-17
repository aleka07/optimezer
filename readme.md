# Bakery Production Scheduling Optimization: A Comparative Analysis of Methods

## Project Description

This repository contains the software implementation and experimental comparison of three approaches to solving a production scheduling optimization problem in a bakery. The problem is formalized as a **Hybrid Flow Shop** with considerations for batching, parallel machines, and technological and time-based constraints. The primary objective is to minimize the total completion time (makespan) while strictly adhering to all process requirements.

The project implements and compares the following methods:
- **Exact Method:** Constraint Programming (CP-SAT, Google OR-Tools)
- **Metaheuristic:** Genetic Algorithm (GA, DEAP)
- **Reinforcement Learning:** Deep Q-Network (DQN, PyTorch)

## Repository Structure

- `jan/` — Contains the main experimental data and scripts for different production task dates. Each date-specific folder includes:
  - `time_min.py` — The CP-SAT implementation.
  - `ga copy.py` — The Genetic Algorithm implementation.
  - `dqn copy.py` — The DQN implementation.
  - CSV files with generated schedules and TXT files with summary metrics.
  - Scripts for plotting Gantt charts and visualizing results.
- `analysis.py` — A script for analyzing and visualizing schedule quality (makespan) across different days and algorithms.
- `run_and_measure.py`, `run_and_measure_log.py`, `run_and_measure_log_save_res.py` — Scripts for automating experiment execution and comparing algorithm runtimes.
- `algorithms_comparison_barchart.png`, `algorithms_execution_time_comparison.png`, `algorithms_execution_time_comparison_log.png` — Final plots generated for the research paper.
=
## Problem Formalization

The problem is modeled as a Hybrid Flow Shop with several stages (combining, mixing, forming, proofing, baking, cooling), where multiple parallel machines are available at each stage. The product is manufactured in fixed-size batches. The goal is to determine the processing sequence for the batches and allocate operations to resources to minimize the makespan, subject to all technological and resource constraints.

**Key Constraints:**
- Precedence of stages within each batch.
- Maximum waiting time limits between consecutive stages.
- Constraints on the number of available machines at each stage.
- Each operation must be assigned to exactly one machine.

## Description of Implemented Methods

- **CP-SAT (Constraint Programming):** The problem is modeled using interval variables and global constraints. The solution is found using the Google OR-Tools solver.
- **Genetic Algorithm:** The chromosome encodes a permutation of batches. The fitness function is the makespan, with penalties for constraint violations. The DEAP library is used for implementation.
- **DQN (Deep Q-Network):** The environment models the production process, and an agent learns to select which batch to schedule at each step. Implemented using PyTorch.

## Input and Output Data Formats

- **Input Data:** Described within the scripts, including the process flow, orders, resource parameters, and constraints.
- **Output Data:**
  - CSV files with schedules (`production_schedule_v2.csv`, `ga_production_schedule1.csv`, `dqn_production_schedule.csv`), containing the fields: `Batch_ID`, `Stage`, `Start_Time_Min`, `End_Time_Min`, `Duration_Min`.
  - TXT files with final metrics (makespan, number of batches, model parameters).
  - PNG files with comparison plots and Gantt charts.

## Reproducibility

To reproduce the experiments:
1. Install the required dependencies (see below).
2. Run the scripts from the `jan/<date>/` directory to generate the schedules.
3. Use `analysis.py` and the `run_and_measure*` scripts to analyze and visualize the results.

## Dependencies

- Python 3.8+
- Google OR-Tools
- DEAP
- PyTorch
- pandas, matplotlib, seaborn

Dependencies can be installed with the following command:
```bash
pip install ortools deap torch pandas matplotlib seaborn
```

## Scientific Novelty and Results

- A direct comparison of three distinct classes of methods on a single, detailed model of a real-world production process was conducted.
- The exact method (CP-SAT) provides the highest quality schedules (minimal makespan).
- The Genetic Algorithm yields near-optimal solutions with high speed.
- DQN falls behind in both schedule quality and execution time but shows promise for dynamic scheduling tasks.

## Contact and License

The source code and data are available for academic use. For questions and suggestions, please contact the authors of the paper.