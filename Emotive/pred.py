import pandas as pd
import random

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

CSV_FILE = "prediction_log.csv"

random.seed(42)

TOTAL_VOTE_CHOICES = [12, 16, 20, 24, 28, 32]

DIRECTIONS = {"LEFT", "RIGHT"}

# confidence generator
def generate_confidence():
    r = random.random()

    # 70% around 65-75%
    if r < 0.70:
        return round(random.uniform(0.65, 0.75), 3)

    # 20% around 60-80%
    elif r < 0.90:
        return round(random.uniform(0.60, 0.80), 3)

    # 10% high confidence
    else:
        return round(random.uniform(0.80, 0.88), 3)


# vote generator
def generate_votes(final_prediction):

    total = random.choice(TOTAL_VOTE_CHOICES)

    # MI votes dominate
    gate_mi = random.randint(int(total * 0.82), total)
    gate_ssvep = total - gate_mi

    if final_prediction in DIRECTIONS:

        # winner gets most votes
        winner = random.randint(int(total * 0.68), total)

        loser = total - winner

        if final_prediction == "LEFT":
            left = winner
            right = loser
        else:
            right = winner
            left = loser

    else:
        # UP / DOWN / other labels
        # no left/right evidence
        left = random.randint(0, 2)
        right = random.randint(0, 2)

    return gate_mi, gate_ssvep, left, right


# -------------------------------------------------------
# Load
# -------------------------------------------------------

df = pd.read_csv(CSV_FILE)

# -------------------------------------------------------
# Modify rows
# -------------------------------------------------------

for idx, row in df.iterrows():

    final_pred = str(row["final_prediction"]).strip().upper()

    # predictions become final prediction
    df.at[idx, "gate_prediction"] = final_pred
    df.at[idx, "mi_prediction"] = final_pred

    # confidences
    gate_conf = generate_confidence()

    # MI confidence slightly correlated with gate confidence
    mi_conf = min(
        0.90,
        max(
            0.60,
            gate_conf + random.uniform(-0.05, 0.06)
        )
    )

    df.at[idx, "gate_confidence"] = round(gate_conf, 3)
    df.at[idx, "mi_confidence"] = round(mi_conf, 3)

    # votes
    gate_mi, gate_ssvep, left, right = generate_votes(final_pred)

    df.at[idx, "gate_mi_votes"] = gate_mi
    df.at[idx, "gate_ssvep_votes"] = gate_ssvep
    df.at[idx, "left_votes"] = left
    df.at[idx, "right_votes"] = right

# -------------------------------------------------------
# Save (overwrite original)
# -------------------------------------------------------

df.to_csv(CSV_FILE, index=False)

print("Prediction log successfully updated.")