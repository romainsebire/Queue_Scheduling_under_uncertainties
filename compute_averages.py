"""
Compute averages of Customer waiting time, Appointment compliance, and Unserved customers
across all 50 CSV result files in results/tmp, using the same logic as PolicyEvaluation.

Each result_tmp_{N}_0.csv uses instance_set/timeline_{N}.json and appointments_{N}.json.
"""

import os
import json
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
RESULTS_DIR   = "results/tmp"
INSTANCE_DIR  = "instance_set"

UNBEARABLE_WAIT            = 60
UNBEARABLE_WAIT_APPOINTMENT = 30
EPSILON_APPOINTMENT        = 3
APPOINTMENTS_MAX_EARLY     = 60

# ── Metric functions (mirrors PolicyEvaluation) ────────────────────────────
def compute_waiting_score(clients_history: list[dict],
                          total_customers: int,
                          appointments: dict) -> float:
    scores = []
    for c in clients_history:
        if c["client"] not in appointments:
            wait_time = c["start"] - c["arrival"]
            score = max(0.0, 100 * (1 - wait_time / UNBEARABLE_WAIT))
            scores.append(score)

    number_unserved = total_customers - len(clients_history)
    denom = len(scores) + number_unserved
    if denom == 0:
        return 0.0
    return sum(scores) / denom


def compute_appointment_compliance(clients_history: list[dict],
                                   timeline: list,
                                   appointments: dict) -> float:
    if not appointments:
        return 100.0

    served_lookup   = {c["client"]: c["start"] for c in clients_history}
    customer_ids    = {row[0] for row in timeline}

    scores    = []
    no_valid  = True

    for customer_id, appt_time in appointments.items():
        if customer_id not in customer_ids:
            continue
        no_valid = False

        service_time = served_lookup.get(customer_id, -1)

        if service_time == -1:
            scores.append(0.0)
            continue

        diff = service_time - appt_time
        if abs(diff) <= EPSILON_APPOINTMENT:
            scores.append(100.0)
        elif (service_time < appt_time - EPSILON_APPOINTMENT and
              service_time > appt_time - APPOINTMENTS_MAX_EARLY):
            scores.append(100 * (1 + (diff + EPSILON_APPOINTMENT) /
                                 (APPOINTMENTS_MAX_EARLY - EPSILON_APPOINTMENT)))
        elif (service_time > appt_time + EPSILON_APPOINTMENT and
              service_time < appt_time + UNBEARABLE_WAIT_APPOINTMENT):
            scores.append(100 / (UNBEARABLE_WAIT_APPOINTMENT - EPSILON_APPOINTMENT) *
                          (appt_time - service_time + UNBEARABLE_WAIT_APPOINTMENT))
        else:
            scores.append(0.0)

    if no_valid:
        return 100.0
    return sum(scores) / len(scores)


def compute_unserved_score(clients_history: list[dict], total_customers: int) -> float:
    if total_customers == 0:
        return 100.0
    return 100 * (len(clients_history) / total_customers)


# ── Process all CSV files ─────────────────────────────────────────────────────
csv_files = sorted(
    [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")],
    key=lambda x: int(x.split("_")[2])
)

results = []
for fname in csv_files:
    # Extract instance id from filename: result_tmp_{instance_id}_{run}.csv
    parts = fname.replace(".csv", "").split("_")
    instance_id = int(parts[2])

    # Load per-instance data
    with open(f"{INSTANCE_DIR}/timeline_{instance_id}.json", "r", encoding="utf-8") as f:
        timeline = json.load(f)
    with open(f"{INSTANCE_DIR}/appointments_{instance_id}.json", "r", encoding="utf-8") as f:
        appointments_raw = json.load(f)

    total_customers = len(timeline)
    appointments    = {row[0]: row[2] for row in appointments_raw}

    # Read CSV
    df = pd.read_csv(os.path.join(RESULTS_DIR, fname), sep=";")
    clients_history = df.to_dict(orient="records")

    grade_wait  = compute_waiting_score(clients_history, total_customers, appointments)
    grade_appt  = compute_appointment_compliance(clients_history, timeline, appointments)
    grade_unsrv = compute_unserved_score(clients_history, total_customers)

    results.append({
        "file":                   fname,
        "instance_id":            instance_id,
        "total_customers":        total_customers,
        "served_customers":       len(clients_history),
        "waiting_score":          grade_wait,
        "appointment_compliance": grade_appt,
        "unserved_score":         grade_unsrv,
    })

df_results = pd.DataFrame(results)

avg_wait  = df_results["waiting_score"].mean()
avg_appt  = df_results["appointment_compliance"].mean()
avg_unsrv = df_results["unserved_score"].mean()

weights = {"waiting": 0.4, "appointment": 0.4, "unserved": 0.2}
final_grade = (weights["waiting"] * avg_wait +
               weights["appointment"] * avg_appt +
               weights["unserved"] * avg_unsrv)

print(f"\n{'='*65}")
print(f"  Average scores across {len(csv_files)} instances")
print(f"{'='*65}")
print(f"{'Metric':30} | {'Avg Score':>10} | {'Weight':>6} | {'Weighted':>8}")
print(f"{'-'*65}")
print(f"{'Customer waiting time':30} | {avg_wait:10.2f} | {weights['waiting']:6.2f} | {avg_wait*weights['waiting']:8.2f}")
print(f"{'Appointment compliance':30} | {avg_appt:10.2f} | {weights['appointment']:6.2f} | {avg_appt*weights['appointment']:8.2f}")
print(f"{'Unserved customers':30} | {avg_unsrv:10.2f} | {weights['unserved']:6.2f} | {avg_unsrv*weights['unserved']:8.2f}")
print(f"{'-'*65}")
print(f"{'FINAL GRADE (avg)':30} | {final_grade:10.2f}")
print(f"{'='*65}")

