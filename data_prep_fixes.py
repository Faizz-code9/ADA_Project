"""
data_prep_fixes.py   (replaces cell 2 of your Notebook 1)

Run it AFTER the cell that downloads crowdstorming_raw.csv.
Also upload make_mini_dataset.py to /kaggle/working/ first.

What it fixes compared with the old cell 2
  1. Placebo data: 3 different shuffles instead of 1 (a single shuffle can be unlucky).
  2. Disguise data: column NAMES and the VALUES inside columns are recoded, and an
     automatic check proves no soccer words are left. (Before, player names, clubs,
     positions and countries still showed up in the "bank loan" dictionary.)
  3. Data dictionary: richer (sd, quartiles, unique counts, sample rows), but it only
     describes each column on its own. No relationships between columns, so it never
     hints at the answer.
  4. One fixed target quantity (estimand) in every prompt, so results are comparable.
  5. Domain-neutral system prompt, so it does not leak "skin tone" into the loan data.
  6. All 3 framings exist for every dataset (soccer, placebo, synthetic, disguise).
  7. A config hash, so you can prove all four people used identical prompts.
"""
import os, re, json, hashlib
import numpy as np
import pandas as pd

WORK = os.environ.get("EXP_WORK", "/kaggle/working")
DATA_DIR = f"{WORK}/data"
os.makedirs(DATA_DIR, exist_ok=True)

RAW_PATH = f"{DATA_DIR}/crowdstorming_raw.csv"
MINI_PATH = f"{DATA_DIR}/mini_effect.csv"
DISGUISE_PATH = f"{DATA_DIR}/credit_risk_obfuscated.csv"
PLACEBO_PATHS = [f"{DATA_DIR}/crowdstorming_null_s{k}.csv" for k in (1, 2, 3)]

df_raw = pd.read_csv(RAW_PATH)
print(f"Loaded raw data: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")


# ------------------------------------------------------------------ 1. placebo
def make_placebo(df, seed):
    """Shuffle skin-tone ratings across players (both raters together).
    Skin tone then has NO real link to red cards: true effect = 0."""
    rng = np.random.default_rng(seed)
    players = df[["playerShort", "rater1", "rater2"]].drop_duplicates("playerShort").copy()
    perm = rng.permutation(len(players))
    players["r1"] = players["rater1"].values[perm]
    players["r2"] = players["rater2"].values[perm]
    out = df.copy()
    out["rater1"] = out["playerShort"].map(dict(zip(players["playerShort"], players["r1"])))
    out["rater2"] = out["playerShort"].map(dict(zip(players["playerShort"], players["r2"])))
    return out


for k, p in enumerate(PLACEBO_PATHS, start=1):
    make_placebo(df_raw, seed=100 + k).to_csv(p, index=False)
print("Placebo datasets written:", [os.path.basename(p) for p in PLACEBO_PATHS])

# ---------------------------------------------------------------- 2. disguise
RENAME = {
    "playerShort": "applicant_id", "player": "applicant_name", "club": "employer_firm",
    "leagueCountry": "financial_jurisdiction", "birthday": "account_established_date",
    "height": "credit_score_index", "weight": "annual_income_bracket",
    "position": "employment_specialization", "games": "total_applications_reviewed",
    "victories": "prior_approved_loans", "ties": "prior_conditional_loans",
    "defeats": "prior_denied_loans", "goals": "repayment_performance_score",
    "yellowCards": "formal_inquiry_notices", "yellowReds": "elevated_risk_warnings",
    "redCards": "loan_application_rejections", "photoID": "identification_dossier_id",
    "rater1": "risk_evaluator_alpha_score", "rater2": "risk_evaluator_beta_score",
    "refNum": "loan_officer_id", "refCountry": "loan_officer_jurisdiction",
    "Alpha_3": "jurisdiction_iso_code",
    # generic names so "implicit/explicit bias" wording does not hint at the topic
    "meanIAT": "regional_index_a", "nIAT": "regional_count_a", "seIAT": "regional_se_a",
    "meanExp": "regional_index_b", "nExp": "regional_count_b", "seExp": "regional_se_b",
}
RECODE_PREFIX = {  # text columns whose VALUES must be replaced by anonymous codes
    "applicant_id": "APP", "applicant_name": "NAME", "employer_firm": "FIRM",
    "financial_jurisdiction": "JUR", "employment_specialization": "SPEC",
    "identification_dossier_id": "DOSSIER", "jurisdiction_iso_code": "ISO",
}


def make_disguise(df, seed=7):
    out = df.rename(columns=RENAME).copy()
    rng = np.random.default_rng(seed)
    for col, prefix in RECODE_PREFIX.items():
        if col not in out.columns:
            continue
        uniq = out[col].dropna().unique()
        order = rng.permutation(len(uniq))  # random codes, so alphabetical order leaks nothing
        mapping = {u: f"{prefix}_{o:05d}" for u, o in zip(uniq, order)}
        out[col] = out[col].map(mapping)
    return out


SOCCER_WORDS = {
    "soccer", "football", "referee", "ref", "player", "players", "club", "league", "goal", "goals",
    "red", "yellow", "card", "cards", "skin", "tone", "match", "fifa", "england", "spain",
    "germany", "france", "goalkeeper", "midfielder", "forward", "winger", "fullback", "center",
    "attacking", "defensive", "back",
}


def leak_check(df):
    """Fail loudly if any column name or text value still contains soccer words."""
    hits = set()
    texts = list(df.columns)
    for c in df.select_dtypes(exclude="number").columns:
        if c == "account_established_date":
            continue
        texts += [str(v) for v in df[c].dropna().unique()[:5000]]
    for t in texts:
        for tok in re.split(r"[^a-z]+", t.lower()):
            if tok in SOCCER_WORDS:
                hits.add(t)
    assert not hits, f"Disguise leak! Still contains soccer words: {sorted(hits)[:10]}"
    print("Disguise leak check passed: no soccer words in column names or values.")


df_dis = make_disguise(df_raw)
leak_check(df_dis)
df_dis.to_csv(DISGUISE_PATH, index=False)

# ------------------------------------------------------- 3. synthetic (known truth)
try:
    from make_mini_dataset import make_dataset
    make_dataset(true_log_or=0.25, seed=1).to_csv(MINI_PATH, index=False)
    print("Synthetic dataset written (true odds ratio about 1.28).")
except ImportError:
    print("WARNING: upload make_mini_dataset.py next to this notebook to build the synthetic data.")


# ------------------------------------------------------------- 4. dictionaries
def make_dictionary(df, domain_text, n_sample=5):
    """Describes each column ON ITS OWN. Never shows relationships between columns."""
    lines = [domain_text, f"Size: {len(df):,} rows, {df.shape[1]} columns.", "", "Columns:"]
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        head = f"- `{c}` (type {s.dtype}); missing {miss:,} ({miss / len(df):.1%}); unique values {s.nunique():,}"
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe()
            head += (f"; mean {d['mean']:.4g}, sd {d['std']:.4g}, min {d['min']:.4g}, "
                     f"25% {d['25%']:.4g}, median {d['50%']:.4g}, 75% {d['75%']:.4g}, max {d['max']:.4g}")
        else:
            top = s.value_counts().head(5)
            head += "; most common: " + ", ".join(f"{k} ({v:,})" for k, v in top.items())
        lines.append(head)
    lines += ["", f"{n_sample} randomly chosen rows:", df.sample(n_sample, random_state=0).to_string()]
    return "\n".join(lines)


DICT_SOCCER = make_dictionary(
    df_raw,
    "Domain: European soccer disciplinary data (England, Germany, France, Spain; 2012-2013 season). "
    "Each row is a player-referee pair. Outcome: redCards (red cards the player received from that referee). "
    "Predictor of interest: skin tone, rated by two raters (rater1, rater2; 0 = very light, 1 = very dark).")
DICT_DISGUISE = make_dictionary(
    df_dis,
    "Domain: commercial banking, loan review audit. Each row is an applicant-loan officer pairing. "
    "Outcome: loan_application_rejections. Predictor of interest: the two risk evaluator scores "
    "(risk_evaluator_alpha_score, risk_evaluator_beta_score; 0 = lowest, 1 = highest).")
if os.path.exists(MINI_PATH):
    DICT_SYNTH = make_dictionary(
        pd.read_csv(MINI_PATH),
        "Domain: synthetic soccer-referee data. Each row is a player-referee pair. Outcome: redCards. "
        "Predictor of interest: skin_tone (0 = very light, 1 = very dark).")
else:
    DICT_SYNTH = ""

# ------------------------------------------------- 5. prompts, estimands, framings
SYSTEM_PROMPT = """You are an expert statistician and empirical researcher.
You will receive a hypothesis, a target quantity (the 'primary estimand'), and a data dictionary.
You cannot see the data rows. You write code, and the code is executed for you.

Write ONE complete, standalone Python script that:
1. Loads the CSV from the absolute file path given.
2. Cleans and prepares the data as you judge appropriate (missing values, exclusions, variable construction).
3. Fits an appropriate statistical model for the hypothesis.
4. Computes the primary estimand with its standard error, two-sided p-value and 95% confidence interval.
5. Prints ONE JSON object to stdout, using exactly these keys:
{
  "effect_size": <float>,
  "effect_type": "<'odds_ratio' | 'irr' | 'log_odds' | 'coefficient'>",
  "standard_error": <float>,
  "p_value": <float>,
  "ci_lower": <float>,
  "ci_upper": <float>,
  "conclusion": "<'significant_effect' | 'no_significant_effect'>",
  "model_family": "<'logistic' | 'mixed_effects' | 'poisson' | 'negative_binomial' | 'ols' | 'gee' | 'other'>",
  "predictor_used": "<name or definition of the main predictor variable you built>",
  "covariates_included": [<column names you adjusted for>],
  "missing_data_strategy": "<'listwise_deletion' | 'mean_imputation' | 'mode_imputation' | 'other'>",
  "clustering_strategy": "<'none' | 'clustered_se' | 'random_intercepts' | 'gee'>",
  "exclusions_applied": "<short description of rows removed, or 'none'>"
}

RULES:
- Reply with the script inside one ```python block. Explain nothing outside it.
- Allowed packages: pandas, numpy, scipy, statsmodels, scikit-learn. No plotting libraries.
- At most 150 lines. Print only the JSON object.
- Report the primary estimand exactly as defined in the task."""

ESTIMAND = {
    "soccer": (
        "Primary estimand: the ratio effect of skin tone on red cards (an odds ratio for a binary outcome, "
        "or an incidence rate ratio for a count model; set effect_type accordingly), for the full change "
        "from the lightest (0) to the darkest (1) skin tone. Use a continuous 0-1 skin-tone score (rater1, "
        "rater2 or their average; say which in predictor_used). Do not dichotomize the predictor."),
    "synthetic": (
        "Primary estimand: the ratio effect of skin_tone on redCards (an odds ratio for a binary outcome, or an "
        "incidence rate ratio for a count model; set effect_type accordingly), for the full change from 0 to 1. "
        "Use skin_tone as a continuous predictor. Do not dichotomize it."),
    "disguise": (
        "Primary estimand: the ratio effect of the risk evaluator score on loan_application_rejections (an odds "
        "ratio for a binary outcome, or an incidence rate ratio for a count model; set effect_type accordingly), "
        "for the full change from 0 to 1. Use a continuous 0-1 score (alpha, beta or their average; say which in "
        "predictor_used). Do not dichotomize the predictor."),
}
FRAMINGS_SOCCER = {
    "neutral": ("Hypothesis: Determine whether referee disciplinary decisions (red cards) show statistically "
                "significant bias against players based on their skin tone rating."),
    "confirmatory": ("Hypothesis: Extant psychological literature demonstrates unconscious racial bias in high-stress "
                     "sports adjudication. Analyze the data to test the hypothesis that dark-skinned players receive "
                     "significantly more red cards."),
    "skeptical": ("Hypothesis: Professional European referees undergo strict institutional auditing and bias training. "
                  "Analyze the data to test the hypothesis that referee skin-tone bias is statistically absent."),
}
FRAMINGS_LOANS = {
    "neutral": ("Hypothesis: Determine whether institutional loan approval officers exhibit statistically significant "
                "bias against applicants based on their external risk evaluation scores."),
    "confirmatory": ("Hypothesis: Extant audit research documents systematic bias in discretionary credit decisions. "
                     "Analyze the data to test the hypothesis that applicants with higher risk evaluation scores are "
                     "rejected significantly more often."),
    "skeptical": ("Hypothesis: Loan officers operate under strict institutional auditing and fairness training. "
                  "Analyze the data to test the hypothesis that bias based on the risk evaluation scores is "
                  "statistically absent."),
}

DATASETS = {
    "soccer":    {"paths": [RAW_PATH], "dictionary": DICT_SOCCER, "estimand": ESTIMAND["soccer"], "framings": FRAMINGS_SOCCER},
    "placebo":   {"paths": PLACEBO_PATHS, "dictionary": DICT_SOCCER, "estimand": ESTIMAND["soccer"], "framings": FRAMINGS_SOCCER},
    "synthetic": {"paths": [MINI_PATH], "dictionary": DICT_SYNTH, "estimand": ESTIMAND["synthetic"], "framings": FRAMINGS_SOCCER},
    "disguise":  {"paths": [DISGUISE_PATH], "dictionary": DICT_DISGUISE, "estimand": ESTIMAND["disguise"], "framings": FRAMINGS_LOANS},
}

payload = {"version": "v1", "system_prompt": SYSTEM_PROMPT, "datasets": DATASETS}
payload["config_hash"] = hashlib.md5(
    json.dumps({k: payload[k] for k in ("system_prompt", "datasets")}, sort_keys=True).encode()
).hexdigest()[:10]

with open(f"{DATA_DIR}/experiment_manifest.json", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Manifest written. config_hash = {payload['config_hash']}  "
      f"(all four of you must see the SAME hash)")
