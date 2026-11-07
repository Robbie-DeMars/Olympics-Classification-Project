"""
Expands team rows in the paired ID CSVs into individual athlete rows.
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..")

INPUT_DIR  = os.path.join(DATA_DIR, "Smaller Tables")
OUTPUT_DIR = os.path.join(DATA_DIR, "Data Teams Repaired")

FILES = [
    "swim_candidates.csv",
    "track_candidates.csv",
]


def parse_name(token):
    """
    Parse a FIRST_MIDDLE_LAST underscore-separated token into name parts.
    Returns (first_name, middle_name, last_name).
    """
    parts = token.split("_")
    filtered_parts = []
    for p in parts:
        if p:
            filtered_parts.append(p)
    parts = filtered_parts  # drop empty strings

    if not parts:
        return "", "", ""
    if len(parts) == 1:
        return parts[0].title(), "", ""
    if len(parts) == 2:
        return parts[0].title(), "", parts[1].title()

    # 3+ parts: first, [middles...], last
    first = parts[0].title()
    last = parts[-1].title()
    # Preserve dots (e.g. "H.") and title-case each middle part
    middle = " ".join(
        p if p.endswith(".") else p.title()
        for p in parts[1:-1]
    )
    return first, middle, last


def expand_row(row):
    """
    If athlete_id contains a comma, split into one row per athlete.
    Otherwise return the row unchanged as a single-element list.
    """
    athlete_id = row.get("athlete_id", "")

    if "," not in athlete_id:
        return [row]

    # Split on ",_" — each token is an athlete name in UPPER_UNDERSCORE format.
    # The first token has no leading "_", subsequent ones do after split.
    tokens = []
    for t in athlete_id.split(","):
        t = t.strip().lstrip("_").strip()
        if t:
            tokens.append(t)

    expanded = []
    for token in tokens:
        first, middle, last = parse_name(token)
        new_row = dict(row)
        new_row["athlete_id"] = token
        new_row["first_name"] = first
        new_row["middle_name"] = middle
        new_row["last_name"] = last
        expanded.append(new_row)

    return expanded


def process_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    team_rows = 0
    individual_rows_added = 0
    total_out = 0

    with open(input_path, newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            expanded = expand_row(row)
            if len(expanded) > 1:
                team_rows += 1
                individual_rows_added += len(expanded)
            for out_row in expanded:
                writer.writerow(out_row)
                total_out += 1

    net_added = individual_rows_added - team_rows
    print(f"{filename}")
    print(f"  Team rows expanded : {team_rows}")
    print(f"  Individuals added  : {individual_rows_added}  (+{net_added} net rows)")
    print(f"  Total output rows  : {total_out}")
    print(f"  -> {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in FILES:
        input_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(input_path):
            print(f"Skipping {filename} (not found)")
            continue
        process_file(filename)
        print()


if __name__ == "__main__":
    main()
