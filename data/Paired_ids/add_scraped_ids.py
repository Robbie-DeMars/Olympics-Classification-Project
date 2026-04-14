"""
Adds a scraped_id column to swim_candidates.csv and track_candidates.csv
by matching athletes from the scraped athletes.csv on (first_name, last_name).

Match strategies:
  1. Exact (first, last) after Unicode normalization (handles accents)
  2. Reversed (last, first) — catches East Asian name ordering
  3. Partial first name (scraped first startswith candidate first, or vice versa)
     e.g. "Mitch" matches "Mitchell"

If a name key resolves to multiple scraped athletes, tiebreakers are applied
in order:
  1. nationality_id (country code) match
  2. year_of_birth match
Remaining ambiguities are flagged in the output.

Output files have scraped_id prepended and a match_notes column appended.
"""

import csv
import os
import unicodedata
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..")

SCRAPED_ATHLETES  = os.path.join(DATA_DIR, "Scraped Data/athletes.csv")
COUNTRIES         = os.path.join(DATA_DIR, "Scraped Data/countries.csv")
ATHLETE_EVENT     = os.path.join(DATA_DIR, "Scraped Data/athlete_event.csv")
EVENTS            = os.path.join(DATA_DIR, "Scraped Data/events.csv")
OLYMPICS          = os.path.join(DATA_DIR, "Scraped Data/olympics.csv")
INPUT_OUTPUT = [
    (os.path.join(DATA_DIR, "Data Teams Repaired/swim_candidates.csv"),
     os.path.join(DATA_DIR, "Paired_ids/swim/swim_candidates_with_ids.csv"),
     os.path.join(DATA_DIR, "Smaller Tables/swim_combined.csv")),
    (os.path.join(DATA_DIR, "Data Teams Repaired/track_candidates.csv"),
     os.path.join(DATA_DIR, "Paired_ids/track/track_candidates_with_ids.csv"),
     os.path.join(DATA_DIR, "Smaller Tables/track_combined.csv")),
]


def normalize(s):
    """Lowercase + strip accents."""
    s = s.strip().lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def load_scraped_athlete_games(athlete_event_path, events_path, olympics_path):
    """
    Build {scraped_athlete_id: set_of_year_strs} by joining:
      athlete_event -> events (olympics_id) -> olympics (year)
    """
    olympics_to_year = {}
    with open(olympics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            olympics_to_year[row["olympics_id"].strip()] = row["year"].strip()

    event_to_year = {}
    with open(events_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = olympics_to_year.get(row["olympics_id"].strip(), "")
            if year:
                event_to_year[row["events_id"].strip()] = year

    athlete_years = defaultdict(set)
    with open(athlete_event_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year = event_to_year.get(row["event_id"].strip(), "")
            if year:
                athlete_years[row["athlete_id"].strip()].add(year)

    return dict(athlete_years)


def load_candidate_games(combined_path):
    """
    Build {candidate_athlete_id: set_of_year_strs} from a combined CSV.
    Year is extracted from the event_id token, e.g.
      "20KM_RACE_WALK_MEN_2008_Beijing" -> "2008"
    """
    candidate_years = defaultdict(set)
    with open(combined_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row["athlete_id"].strip()
            for part in row["event_id"].strip().split("_"):
                if part.isdigit() and len(part) == 4:
                    candidate_years[aid].add(part)
                    break
    return dict(candidate_years)


def load_nationality_map(countries_path):
    """Build {numeric_id_str: country_code} from countries.csv."""
    mapping = {}
    with open(countries_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["id"].strip()] = row["country_id"].strip().upper()
    return mapping


def load_scraped_athletes(path, nat_map):
    """
    Build lookup indices from the scraped athletes file.
    Attaches a resolved country_code to each row using nat_map.
    Returns:
      index_fl: (first_norm, last_norm) -> [rows]   (normal ordering)
      index_lf: (last_norm, first_norm) -> [rows]   (reversed ordering)
    """
    index_fl = defaultdict(list)
    index_lf = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = normalize(row["first_name"])
            ln = normalize(row["last_name"])
            row["country_code"] = nat_map.get(row.get("nationality_id", "").strip(), "")
            index_fl[(fn, ln)].append(row)
            index_lf[(ln, fn)].append(row)
    return index_fl, index_lf


def partial_first_match(index_fl, fn_cand, ln_cand):
    """
    Look for scraped rows where last names match and one first name
    starts with the other (handles Mitch/Mitchell, etc.).
    """
    results = []
    for (fn_sc, ln_sc), rows in index_fl.items():
        if ln_sc != ln_cand:
            continue
        if fn_sc.startswith(fn_cand) or fn_cand.startswith(fn_sc):
            results.extend(rows)
    return results


def pick_best(candidates, yob_str, nat_str="", cand_years=None, scraped_games=None):
    """
    From a list of candidate scraped rows, pick the best using tiebreakers:
      1. nationality (country code)
      2. year_of_birth
      3. Olympic games overlap
    Returns (scraped_id_str, is_ambiguous).
    """
    if len(candidates) == 1:
        return candidates[0]["athlete_id"], False

    # Tiebreaker 1: nationality
    if nat_str:
        nat_upper = nat_str.strip().upper()
        nat_matched = []
        for r in candidates:
            if r.get("country_code", "") == nat_upper:
                nat_matched.append(r)
        if len(nat_matched) == 1:
            return nat_matched[0]["athlete_id"], False
        if nat_matched:  # narrowed but still ambiguous
            candidates = nat_matched

    # Tiebreaker 2: year_of_birth
    if yob_str:
        try:
            yob = int(yob_str)
            matched = []
            for r in candidates:
                if r.get("year_of_birth", "").strip() == str(yob):
                    matched.append(r)
            if len(matched) == 1:
                return matched[0]["athlete_id"], False
            if matched:  # narrowed but still ambiguous
                candidates = matched
        except ValueError:
            pass

    # Tiebreaker 3: Olympic games overlap
    if cand_years and scraped_games:
        games_matched = []
        for r in candidates:
            if cand_years & scraped_games.get(r["athlete_id"], set()):
                games_matched.append(r)
        if len(games_matched) == 1:
            return games_matched[0]["athlete_id"], False
        if games_matched:  # narrowed but still ambiguous
            candidates = games_matched

    ids = sorted(set(r["athlete_id"] for r in candidates))
    return "|".join(ids), True


def lookup(index_fl, index_lf, fn_cand, ln_cand, yob_str, nat_str="",
           cand_years=None, scraped_games=None):
    """
    Try matching strategies in order, return (scraped_id, note).
    """
    fn = normalize(fn_cand)
    ln = normalize(ln_cand)

    # 1. Exact (first, last)
    rows = index_fl.get((fn, ln), [])
    if rows:
        sid, ambig = pick_best(rows, yob_str, nat_str, cand_years, scraped_games)
        return sid, ("ambiguous_exact" if ambig else "exact")

    # 2. Reversed (last↔first) — East Asian name ordering
    rows = index_lf.get((fn, ln), [])
    if rows:
        sid, ambig = pick_best(rows, yob_str, nat_str, cand_years, scraped_games)
        return sid, ("ambiguous_reversed" if ambig else "reversed")

    # 3. Partial first name match
    rows = partial_first_match(index_fl, fn, ln)
    if rows:
        sid, ambig = pick_best(rows, yob_str, nat_str, cand_years, scraped_games)
        return sid, ("ambiguous_partial_first" if ambig else "partial_first")

    return "", "no_match"


def process(input_path, output_path, index_fl, index_lf, scraped_games, candidate_years):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats = defaultdict(int)
    by_category = defaultdict(list)

    with open(input_path, newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        fieldnames = ["scraped_id"] + reader.fieldnames + ["match_notes"]

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            fn = row.get("first_name", "")
            ln = row.get("last_name", "")
            yob = row.get("year_of_birth", "").strip()
            nat = row.get("nationality_id", "").strip()
            cand_yrs = candidate_years.get(row.get("athlete_id", "").strip(), set())

            # Skip rows with no name at all
            if not fn.strip() and not ln.strip():
                scraped_id, note = "", "no_name"
            else:
                scraped_id, note = lookup(index_fl, index_lf, fn, ln, yob, nat,
                                          cand_yrs, scraped_games)

            stats[note] += 1
            row["scraped_id"] = scraped_id
            row["match_notes"] = note
            writer.writerow(row)
            by_category[note].append(dict(row))

    base, ext = os.path.splitext(output_path)
    for note, rows in by_category.items():
        cat_path = f"{base}_{note}{ext}"
        with open(cat_path, "w", newline="", encoding="utf-8") as fcat:
            cat_writer = csv.DictWriter(fcat, fieldnames=fieldnames)
            cat_writer.writeheader()
            cat_writer.writerows(rows)

    total = sum(stats.values())
    print(f"\n=== {input_path} ===")
    print(f"  Total rows: {total}")
    for note, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {note:30s}: {count:5d}  ({pct:.1f}%)")
    print(f"  -> Written to {output_path}")


def main():
    print(f"Loading nationality map from {COUNTRIES} ...")
    nat_map = load_nationality_map(COUNTRIES)
    print(f"  Loaded {len(nat_map)} country entries")

    print(f"Loading scraped athletes from {SCRAPED_ATHLETES} ...")
    index_fl, index_lf = load_scraped_athletes(SCRAPED_ATHLETES, nat_map)
    print(f"  Loaded {sum(len(v) for v in index_fl.values())} athletes, "
          f"{len(index_fl)} unique (first, last) keys")

    print(f"Loading scraped athlete games from {ATHLETE_EVENT} ...")
    scraped_games = load_scraped_athlete_games(ATHLETE_EVENT, EVENTS, OLYMPICS)
    print(f"  Loaded games for {len(scraped_games)} scraped athletes")

    for input_path, output_path, combined_path in INPUT_OUTPUT:
        print(f"Loading candidate games from {combined_path} ...")
        candidate_years = load_candidate_games(combined_path)
        print(f"  Loaded games for {len(candidate_years)} candidates")
        process(input_path, output_path, index_fl, index_lf, scraped_games, candidate_years)

    print("\nDone.")


if __name__ == "__main__":
    main()
