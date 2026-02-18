# Queue Scheduling under Uncertainties — Reinforcement Learning

A Reinforcement Learning solution for dynamic queue scheduling in a multi-server environment with walk-in customers and appointment-based patients. The agent learns a scheduling policy that maximises a composite score balancing **waiting time**, **appointment compliance**, and **throughput**.

---

## Problem

A set of servers (agents) must decide, at each decision point, which waiting customer to serve next. Customers arrive stochastically and may or may not have a pre-scheduled appointment. The goal is to minimise waiting times while honouring appointment slots, under uncertainty about future arrivals and service durations.

The composite evaluation score weights:

- **40%** — waiting time score
- **40%** — appointment compliance score
- **20%** — throughput (customers served)

---

## Results — Solution Progression

| Version  | Action Space            | Score      | Description                                                      |
| -------- | ----------------------- | ---------- | ---------------------------------------------------------------- |
| `first`  | Patient ID (51 actions) | ~60%       | Agent selects a specific patient from the queue                  |
| `second` | Task ID + VIP + Hold    | ~70%       | Agent selects a task category or appointment group               |
| `third`  | VIP / Walk-in / Hold    | ~90%       | 3-action space with ROI-based customer selection                 |
| `fourth` | 4 Semantic strategies   | **~92.5%** | Agent picks a scheduling strategy; env resolves to best customer |

---

## Installation

```bash
pip install uv
```

```bash
uv venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

```bash
uv pip install -r requirements.txt
```

**Key dependencies:** `gymnasium==1.2.3`, `stable-baselines3==2.7.1`, `sb3-contrib==2.7.1`, `torch==2.10.0`

---

## Usage

### Step 1 — Train a model

```bash
python -m app.main --solution fourth          # best solution (default)
python -m app.main --solution third
python -m app.main --solution second
python -m app.main --solution first
```

Trained models are saved to `app/data/models/ppo_{solution}.zip`.

### Step 2 — Grade (50 instances)

```bash
python evaluate.py --solution fourth          # grade the fourth solution (default)
python evaluate.py --solution third
python evaluate.py --solution second
python evaluate.py --solution first
```

| Argument     | Values                            | Default  | Description                     |
| ------------ | --------------------------------- | -------- | ------------------------------- |
| `--solution` | `first` `second` `third` `fourth` | `fourth` | Which trained model to evaluate |

What it does:

- Loads `app/data/models/ppo_{solution}.zip`
- Registers the matching environment
- Runs the policy on all 50 instances from `./instance_set/`
- Validates each result (no overlaps, correct arrivals, correct service durations)
- Prints the final composite score (0–100)

> If no trained model is found it exits with an error and tells you which training command to run first.

### Quick evaluate (1 instance, with logs)

```bash
python -m app.main --solution fourth --mode evaluate
```

### All training options

```bash
python -m app.main --solution fourth --mode train --seed 42 --timesteps 250000
```

| Argument      | Values                            | Default  | Description                           |
| ------------- | --------------------------------- | -------- | ------------------------------------- |
| `--solution`  | `first` `second` `third` `fourth` | `fourth` | Action space version                  |
| `--mode`      | `train` `evaluate`                | `train`  | Train then evaluate, or evaluate only |
| `--seed`      | integer                           | `42`     | Random seed for reproducibility       |
| `--timesteps` | integer                           | `250000` | Number of PPO training timesteps      |

### TensorBoard — training reward curve

```bash
tensorboard --logdir=app/data/logs
```

## Project Structure

```
.
├── app/
│   ├── main.py                          # Train entry point — argparse, seed, dynamic registration
│   ├── InstanceGenerator.py             # Generate simulation data files
│   │
│   ├── data/
│   │   ├── config/queue_config.json     # Scenario configuration
│   │   ├── data_files/                  # Single-instance files for quick testing
│   │   ├── models/                      # Saved PPO models (ppo_first.zip … ppo_fourth.zip)
│   │   ├── logs/                        # TensorBoard logs
│   │   └── results/                     # CSV results from quick evaluate
│   │
│   ├── simulation/
│   │   ├── envs/
│   │   │   ├── Env.py                   # Abstract base environment
│   │   │   ├── ChildEnvV1.py            # V1 — Patient ID (dict obs, 51 actions)
│   │   │   ├── ChildEnvV2.py            # V2 — Task ID + VIP (flat obs, n+2 actions)
│   │   │   ├── ChildEnvV3.py            # V3 — VIP/Walk-in + ROI (flat obs, 3 actions)
│   │   │   └── ChildEnvV4.py            # V4 — 4 Semantic strategies (flat obs, 4 actions)
│   │   │
│   │   ├── policies/
│   │   │   ├── Policy.py                # Abstract base policy (simulate loop)
│   │   │   ├── PolicyEvaluation.py      # Composite score evaluator
│   │   │   ├── ChildPolicyV1.py         # V1 — MultiInputPolicy
│   │   │   ├── ChildPolicyV2.py         # V2 — MlpPolicy
│   │   │   ├── ChildPolicyV3.py         # V3 — MlpPolicy
│   │   │   └── ChildPolicyV4.py         # V4 — MlpPolicy, decaying LR, [128,64] arch
│   │   │
│   │   └── events/                      # Simulation event system
│   │
│   └── domain/                          # Domain objects (Customer, Server, Appointment)
│
├── evaluate.py                          # Grading script — 50 instances, --solution flag
├── instance_set/                        # 50 test instances used for grading
│   ├── timeline_0.json … timeline_49.json
│   ├── average_matrix_0.json … average_matrix_49.json
│   ├── appointments_0.json … appointments_49.json
│   └── unavailability_0.json … unavailability_49.json
│
└── solutions_code/                      # Development history (reference only)
    ├── First action space (Patient ID)(60.)/
    ├── Second action space (Task ID)(70.06)/
    ├── Third action space + ROI (90.38)/
    └── Fourth Action Space - Semantic Action - 92.52/
```
