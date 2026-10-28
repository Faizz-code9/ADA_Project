"""
Mini synthetic dataset for the PILOT (small, fast, no download, no contamination).

Mimics the structure of the soccer-referee data, but WE control the truth:
  - TRUE_LOG_OR = 0.0  -> no real effect (placebo world)
  - TRUE_LOG_OR = 0.25 -> real effect, odds ratio ~ 1.28

Built-in "analytic traps" (so different reasonable choices can diverge):
  - position  : confounder (affects both skin_tone mix and red cards)
  - referee strictness: some referees give many more cards (clustering matters)
  - yellowCards: caused by skin tone AND counts the red cards (red -> yellow)
                 -> it is a COLLIDER (tone -> yellow <- red). Adjusting for it
                    opens a spurious path and biases the estimate ("bad control").
  - games     : exposure; more games = more chances for a card
  - missing skin_tone for ~8% of players (missing-data choice matters)

Usage:
    python make_mini_dataset.py            # writes mini_effect.csv and mini_null.csv
"""
import numpy as np
import pandas as pd


def make_dataset(true_log_or=0.25, n_players=300, n_refs=60,
                 dyads_per_player=12, seed=0):
    rng = np.random.default_rng(seed)

    # ---- players -----------------------------------------------------------
    positions = np.array(["Goalkeeper", "Defender", "Midfielder", "Forward"])
    pos = rng.choice(positions, size=n_players, p=[0.10, 0.35, 0.35, 0.20])

    # skin tone on the 0 / .25 / .5 / .75 / 1 scale; mix depends on position (confounding)
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    base_p = np.array([0.35, 0.30, 0.15, 0.12, 0.08])
    tone = np.empty(n_players)
    for i, p in enumerate(pos):
        w = base_p.copy()
        if p == "Defender":          # more dark-skinned defenders in this fake world
            w = w * np.array([0.8, 0.9, 1.1, 1.4, 1.6])
        w = w / w.sum()
        tone[i] = rng.choice(levels, p=w)

    height = rng.normal(181, 6.5, n_players).round(0)
    weight = rng.normal(76, 7, n_players).round(0)
    league = rng.choice(["England", "Spain", "Germany", "France"], size=n_players)

    pos_effect = {"Goalkeeper": -0.6, "Defender": 0.4, "Midfielder": 0.1, "Forward": -0.1}

    # ---- referees ----------------------------------------------------------
    ref_strict = rng.normal(0.0, 0.5, n_refs)          # random referee strictness

    # ---- player-referee dyads ---------------------------------------------
    rows = []
    for pid in range(n_players):
        refs = rng.choice(n_refs, size=dyads_per_player, replace=False)
        for r in refs:
            games = int(rng.integers(1, 25))
            lin = (-5.0
                   + true_log_or * tone[pid]            # the effect we care about
                   + pos_effect[pos[pid]]
                   + ref_strict[r])
            p_red = 1 / (1 + np.exp(-lin))
            reds = int(rng.binomial(games, p_red))
            # yellow cards: depend on tone, position, games (post-treatment variable)
            lam_y = games * np.exp(-2.0 + 0.3 * tone[pid] + 0.2 * (pos[pid] == "Defender"))
            yellow = int(rng.poisson(lam_y)) + reds
            rows.append((f"P{pid:04d}", f"R{r:03d}", pos[pid], height[pid],
                         weight[pid], league[pid], tone[pid], games, yellow, reds))

    df = pd.DataFrame(rows, columns=[
        "playerID", "refereeID", "position", "height", "weight", "leagueCountry",
        "skin_tone", "games", "yellowCards", "redCards"])

    # missing skin tone for ~8% of players (whole player, like the real data)
    miss_players = rng.choice(df["playerID"].unique(),
                              size=int(0.08 * n_players), replace=False)
    df.loc[df["playerID"].isin(miss_players), "skin_tone"] = np.nan
    return df


if __name__ == "__main__":
    for name, truth in [("mini_effect.csv", 0.25), ("mini_null.csv", 0.0)]:
        d = make_dataset(true_log_or=truth, seed=1)
        d.to_csv(name, index=False)
        print(f"{name}: {len(d):,} rows, true log-OR = {truth}")
        print(d.head(3).to_string(index=False))
        print()
