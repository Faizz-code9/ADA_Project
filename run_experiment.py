"""
run_experiment.py  -- the experiment runner (new notebook, or new cells after data_prep_fixes.py)

HOW IT WORKS (one run)
  build prompt -> call the AI model -> extract the Python script -> run it on a fresh copy
  of the data -> parse the JSON it prints -> (if it crashed: ONE repair attempt) ->
  save the script, the raw reply and one result row to disk immediately.

HOW YOU USE IT
  make_run_plan()                         # once; all 4 of you get the identical plan (fixed seed)
  pilot("Sumit")                          # tiny test, results kept separate from the real runs
  run_batch("Sumit", session=1)           # real runs for your models in session 1
  run_batch("Sumit", session=1)           # run it again after a crash: finished runs are skipped
Use mock=True to test the whole pipeline without calling any AI (costs nothing).
"""
import os, re, sys, json, time, random, shutil, subprocess
from datetime import datetime
import pandas as pd

WORK = os.environ.get("EXP_WORK", "/kaggle/working")
DATA_DIR = f"{WORK}/data"
OUT_DIR = f"{WORK}/outputs"
MANIFEST_PATH = f"{DATA_DIR}/experiment_manifest.json"
PLAN_PATH = f"{DATA_DIR}/run_plan.csv"

N_SESSIONS = 4          # the main runs are spread over 4 sessions (see plan)
MAX_ATTEMPTS = 2        # a failed run may be tried once more, then it is reported as failed
TIMEOUT_S = 180         # a generated script may run at most 3 minutes
EXPERIMENT_VERSION = "v1"

# ------------------------------------------------------------------ the 8 models
# runner = who runs it.  extra = optional provider-specific settings (a failure to apply
# them is handled automatically: the call is retried without them and a note is logged).
MODELS = {
    "gemini-2.5-flash":      dict(provider="gemini", id="gemini-2.5-flash",      runner="Sumit",      extra={"thinking_budget": 1024}),
    "gpt-oss-120b":          dict(provider="groq",   id="openai/gpt-oss-120b",   runner="Sumit",      extra={"reasoning_effort": "medium"}),
    "qwen3.8-27b":           dict(provider="groq",   id="qwen/qwen3.8-27b",      runner="Faizan",     extra={}),
    "gpt-oss-20b":           dict(provider="groq",   id="openai/gpt-oss-20b",    runner="Faizan",     extra={"reasoning_effort": "medium"}),
    "gemma-4-31b":           dict(provider="gemini", id="gemma-4-31b-it",        runner="Samarth",    extra={"no_system": True}),
    "gemini-3.5-flash":      dict(provider="gemini", id="gemini-3.5-flash",      runner="Samarth",    extra={"thinking_budget": 1024}),
    "allam-2-7b":            dict(provider="groq",   id="allam-2-7b",            runner="Thrivikram", extra={}),
    "gemini-2.5-flash-lite": dict(provider="gemini", id="gemini-2.5-flash-lite", runner="Thrivikram", extra={"thinking_budget": 1024}),
}
SLEEP_BETWEEN_CALLS = {"gemini": 6, "groq": 3, "mock": 0}   # seconds; raise these if you hit rate limits

# ----------------------------------------------- how many runs per cell (option B = 93 per model)
TEMPS = [0.0, 0.7, 1.0]
FRAMING_NAMES = ["neutral", "confirmatory", "skeptical"]
RUNS_PER_CELL = {                    # dataset -> {temperature: runs per framing}
    "soccer":    {0.0: 3, 0.7: 5, 1.0: 5},    # 13 x 3 framings = 39
    "placebo":   {0.0: 2, 0.7: 2, 1.0: 2},    #  6 x 3 = 18
    "synthetic": {0.0: 2, 0.7: 2, 1.0: 2},    # 18
    "disguise":  {0.0: 2, 0.7: 2, 1.0: 2},    # 18   -> 93 per model
}


# ====================================================================== 1. RUN PLAN
def make_run_plan(seed=2026, path=PLAN_PATH):
    """Every run, in random order, with a session number.
    Each cell's repeats are spread over the sessions, so every session contains a mix of
    all models, temperatures, framings and datasets (time is then not confused with a treatment)."""
    rng = random.Random(seed)
    rows = []
    for mname, m in MODELS.items():
        for ds, per_t in RUNS_PER_CELL.items():
            for fr in FRAMING_NAMES:
                for T, n in per_t.items():
                    off = rng.randrange(N_SESSIONS)
                    for rep in range(1, n + 1):
                        rows.append(dict(
                            run_id=f"{mname}|{ds}|{fr}|T{T}|r{rep}", model=mname, runner=m["runner"],
                            dataset=ds, variant=((rep + off) % 3 + 1) if ds == "placebo" else 1,
                            framing=fr, temperature=T, rep=rep, session=(rep + off) % N_SESSIONS + 1))
    rng.shuffle(rows)
    plan = pd.DataFrame(rows)
    plan["order"] = range(1, len(plan) + 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plan.to_csv(path, index=False)
    print(f"Run plan saved: {len(plan)} runs -> {path}")
    print(plan.groupby("runner").size().to_string())
    return plan


# ================================================================= 2. MODEL CALLERS
def get_key(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ[name]


def call_gemini(model_id, messages, temperature, system, extra):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=get_key("GEMINI_API_KEY"))
    notes = []
    msgs = list(messages)
    cfg = dict(temperature=temperature, max_output_tokens=8192)
    if extra.get("no_system"):                       # e.g. Gemma: no separate system field
        msgs[0] = ("user", system + "\n\n" + msgs[0][1])
    else:
        cfg["system_instruction"] = system
    contents = [types.Content(role="user" if r == "user" else "model", parts=[types.Part(text=t)]) for r, t in msgs]
    if extra.get("thinking_budget") is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=extra["thinking_budget"])
    try:
        resp = client.models.generate_content(model=model_id, contents=contents,
                                              config=types.GenerateContentConfig(**cfg))
    except Exception as e:
        if "thinking" in str(e).lower() and "thinking_config" in cfg:
            cfg.pop("thinking_config"); notes.append("thinking_budget not accepted; ran without")
            resp = client.models.generate_content(model=model_id, contents=contents,
                                                  config=types.GenerateContentConfig(**cfg))
        else:
            raise
    um = getattr(resp, "usage_metadata", None)
    fin = None
    try:
        fin = str(resp.candidates[0].finish_reason)
    except Exception:
        pass
    return dict(text=resp.text or "", model_returned=getattr(resp, "model_version", None) or model_id,
                tokens_in=getattr(um, "prompt_token_count", None), tokens_out=getattr(um, "candidates_token_count", None),
                finish_reason=fin, notes="; ".join(notes))


def call_groq(model_id, messages, temperature, system, extra):
    import openai
    client = openai.OpenAI(api_key=get_key("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    msgs = [{"role": "system", "content": system}] + [
        {"role": "user" if r == "user" else "assistant", "content": t} for r, t in messages]
    kw = dict(model=model_id, messages=msgs, temperature=temperature, max_tokens=8192)
    notes = []
    if extra.get("reasoning_effort"):
        kw["extra_body"] = {"reasoning_effort": extra["reasoning_effort"]}
    try:
        resp = client.chat.completions.create(**kw)
    except Exception as e:
        if "reasoning" in str(e).lower() and "extra_body" in kw:
            kw.pop("extra_body"); notes.append("reasoning_effort not accepted; ran without")
            resp = client.chat.completions.create(**kw)
        else:
            raise
    u = getattr(resp, "usage", None)
    return dict(text=resp.choices[0].message.content or "", model_returned=getattr(resp, "model", model_id),
                tokens_in=getattr(u, "prompt_tokens", None), tokens_out=getattr(u, "completion_tokens", None),
                finish_reason=resp.choices[0].finish_reason, notes="; ".join(notes))


def call_mock(model_id, messages, temperature, system, extra):
    """Fake 'AI' for testing the pipeline for free. Writes a numpy/pandas-only script."""
    rnd = random.Random(hash((model_id, temperature, len(messages), messages[0][1][-40:])) & 0xFFFF)
    path = re.search(r"Dataset File Path:\n(\S+)", messages[0][1]).group(1)
    if len(messages) == 1 and rnd.random() < 0.25:     # sometimes write a broken script (tests the repair step)
        code = "import pandas as pd\nprint(undefined_variable)"
    else:
        fam = rnd.choice(["logistic", "poisson", "negative_binomial"])
        cov = rnd.choice([[], ["games"], ["games", "position"]])
        code = f'''import pandas as pd, numpy as np, json
df = pd.read_csv(r"{path}")
tone = "skin_tone" if "skin_tone" in df.columns else ("rater1" if "rater1" in df.columns else "risk_evaluator_alpha_score")
red = [c for c in ["redCards", "loan_application_rejections"] if c in df.columns][0]
d = df.dropna(subset=[tone, red])
dark = d[tone] >= 0.5
r1 = (d.loc[dark, red] > 0).mean(); r0 = (d.loc[~dark, red] > 0).mean()
orr = (r1 / (1 - r1)) / (r0 / (1 - r0))
n1, n0 = dark.sum(), (~dark).sum()
se = float(np.sqrt(1 / max(1, (d.loc[dark, red] > 0).sum()) + 1 / max(1, (d.loc[~dark, red] > 0).sum())))
lo, hi = float(np.exp(np.log(orr) - 1.96 * se)), float(np.exp(np.log(orr) + 1.96 * se))
print(json.dumps({{"effect_size": float(orr), "effect_type": "odds_ratio", "standard_error": se,
  "p_value": 0.03 if lo > 1 else 0.4, "ci_lower": lo, "ci_upper": hi,
  "conclusion": "significant_effect" if lo > 1 else "no_significant_effect",
  "model_family": "{fam}", "predictor_used": tone, "covariates_included": {json.dumps(cov)},
  "missing_data_strategy": "listwise_deletion", "clustering_strategy": "none", "exclusions_applied": "none"}}))
'''
    return dict(text=f"```python\n{code}\n```", model_returned=f"mock:{model_id}", tokens_in=0, tokens_out=0,
                finish_reason="stop", notes="")


CALLERS = {"gemini": call_gemini, "groq": call_groq, "mock": call_mock}


class ApiError(Exception):
    pass


def with_backoff(fn, tries=5):
    """Retry rate-limit style errors with growing waits; raise ApiError for anything else."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = any(s in msg for s in ["429", "rate", "quota", "resource_exhausted", "overloaded", "503", "timeout"])
            if transient and k < tries - 1:
                time.sleep(min(60, 5 * 2 ** k))
                continue
            raise ApiError(str(e)[:300])


# ============================================================= 3. SAFE SCRIPT RUNNER
def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.S)
    return blocks[-1].strip() if blocks else None


def extract_json(stdout):
    s = (stdout or "").strip()
    cands = []
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        cands.append(s[i:j + 1])
    lines = [ln for ln in s.splitlines() if ln.strip().startswith("{")]
    if lines:
        cands.append(lines[-1])
    for c in cands:
        try:
            return json.loads(c)
        except Exception:
            pass
    return None


def run_script(code, run_dir):
    script = os.path.join(run_dir, "analysis.py")
    with open(script, "w") as f:
        f.write(code)
    env = {k: v for k, v in os.environ.items() if not any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET"))}
    try:
        p = subprocess.run([sys.executable, script], capture_output=True, text=True,
                           timeout=TIMEOUT_S, cwd=run_dir, env=env)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -9, "", f"Script exceeded {TIMEOUT_S}s and was stopped."


REQUIRED = ["effect_size", "p_value", "ci_lower", "ci_upper", "conclusion"]


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


FIELDS = ["run_id", "model", "provider", "model_returned", "runner", "dataset", "variant", "framing",
          "temperature", "rep", "session", "timestamp", "experiment_version", "config_hash",
          "success", "healed", "attempt", "error", "effect_size", "effect_type", "standard_error",
          "p_value", "ci_lower", "ci_upper", "conclusion", "sig_p05", "sig_ci_above1", "model_family",
          "predictor_used", "covariates_included", "missing_data_strategy", "clustering_strategy",
          "exclusions_applied", "tokens_in", "tokens_out", "finish_reason", "api_notes", "seconds", "script_path"]


def execute_run(row, manifest, mock=False, attempt=1):
    """One full run. Returns one result dict (never raises for model/script problems)."""
    m = MODELS[row["model"]]
    provider = "mock" if mock else m["provider"]
    caller = CALLERS[provider]
    ds = manifest["datasets"][row["dataset"]]
    src = ds["paths"][int(row["variant"]) - 1]
    run_dir = os.path.join(OUT_DIR, "runs", safe(row["run_id"]))
    os.makedirs(run_dir, exist_ok=True)
    data_copy = os.path.join(run_dir, "data.csv")           # fresh private copy: scripts cannot damage the original
    shutil.copy(src, data_copy)
    os.chmod(data_copy, 0o444)
    user = (f"{ds['framings'][row['framing']]}\n\n{ds['estimand']}\n\n"
            f"Data Dictionary:\n{ds['dictionary']}\n\nDataset File Path:\n{data_copy}")
    messages = [("user", user)]
    rec = {k: None for k in FIELDS}
    rec.update(run_id=row["run_id"], model=row["model"], provider=provider, runner=row["runner"],
               dataset=row["dataset"], variant=row["variant"], framing=row["framing"],
               temperature=row["temperature"], rep=row["rep"], session=row["session"],
               timestamp=datetime.now().isoformat(timespec="seconds"), experiment_version=EXPERIMENT_VERSION,
               config_hash=manifest.get("config_hash"), success=False, healed=False, attempt=attempt)
    t0 = time.time()
    notes, tin, tout = [], 0, 0
    result, err, code = None, "", None
    try:
        for step in (1, 2):                                   # step 2 = the single repair attempt
            resp = with_backoff(lambda: caller(m["id"], messages, row["temperature"], manifest["system_prompt"], m["extra"]))
            tin += resp["tokens_in"] or 0
            tout += resp["tokens_out"] or 0
            rec["model_returned"], rec["finish_reason"] = resp["model_returned"], resp["finish_reason"]
            if resp["notes"]:
                notes.append(resp["notes"])
            with open(os.path.join(run_dir, f"response_step{step}.txt"), "w") as f:
                f.write(resp["text"])
            code = extract_code(resp["text"])
            if code is None:
                err = "No ```python block found in the reply."
            else:
                rc, out, serr = run_script(code, run_dir)
                result = extract_json(out)
                if rc == 0 and result and all(k in result for k in REQUIRED):
                    err = ""
                    rec["healed"] = (step == 2)
                    break
                err = (serr or out or "No output.")[-1500:] if not (rc == 0 and result) else \
                      f"JSON missing keys: {[k for k in REQUIRED if k not in result]}"
                result = None
            if step == 1:
                messages += [("assistant", resp["text"]),
                             ("user", f"Your script failed:\n{err}\nReturn a corrected, complete script in one ```python block, "
                                      f"with the same JSON output format.")]
    except ApiError as e:
        err = f"API error: {e}"
    finally:
        try:
            os.chmod(data_copy, 0o644); os.remove(data_copy)      # keep disk usage small
        except Exception:
            pass
    rec.update(seconds=round(time.time() - t0, 1), tokens_in=tin, tokens_out=tout, api_notes="; ".join(notes),
               error=err, script_path=os.path.join(run_dir, "analysis.py") if code else None)
    if result and not err:
        try:
            ci_lo, ci_hi, p = float(result["ci_lower"]), float(result["ci_upper"]), float(result["p_value"])
            rec.update(success=True, effect_size=float(result["effect_size"]), p_value=p, ci_lower=ci_lo, ci_upper=ci_hi,
                       sig_p05=p < 0.05, sig_ci_above1=ci_lo > 1,
                       conclusion=result.get("conclusion"), effect_type=result.get("effect_type"),
                       standard_error=result.get("standard_error"), model_family=result.get("model_family"),
                       predictor_used=result.get("predictor_used"),
                       covariates_included=json.dumps(result.get("covariates_included")),
                       missing_data_strategy=result.get("missing_data_strategy"),
                       clustering_strategy=result.get("clustering_strategy"),
                       exclusions_applied=result.get("exclusions_applied"))
        except Exception as e:
            rec.update(success=False, error=f"Bad numbers in JSON: {e}")
    return rec


def append_row(rec, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([rec], columns=FIELDS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ================================================================ 4. THE RUNNERS
def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _status(path):
    """run_id -> (already succeeded?, number of attempts so far)"""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    out = {}
    for rid, g in d.groupby("run_id"):
        out[rid] = (bool(g["success"].astype(str).str.lower().eq("true").any()), len(g))
    return out


def run_batch(runner, session, only_models=None, limit=None, mock=False):
    """Run everything assigned to `runner` in `session`. Safe to re-run: finished runs are skipped."""
    manifest = load_manifest()
    plan = pd.read_csv(PLAN_PATH)
    sel = plan[(plan.runner == runner) & (plan.session == session)]
    if only_models:
        sel = sel[sel.model.isin(only_models)]
    sel = sel.sort_values("order")
    res_path = f"{OUT_DIR}/results_{runner}.csv"
    status = _status(res_path)
    todo = [r for r in sel.to_dict("records")
            if not status.get(r["run_id"], (False, 0))[0] and status.get(r["run_id"], (False, 0))[1] < MAX_ATTEMPTS]
    if limit:
        todo = todo[:limit]
    print(f"{runner}, session {session}: {len(sel)} planned, {len(todo)} to do now.")
    api_fail_streak = 0
    for i, row in enumerate(todo, 1):
        attempt = status.get(row["run_id"], (False, 0))[1] + 1
        rec = execute_run(row, manifest, mock=mock, attempt=attempt)
        append_row(rec, res_path)
        tag = f"{rec['effect_size']:.3f}" if rec["success"] else f"FAILED ({str(rec['error'])[:60]})"
        print(f"[{i}/{len(todo)}] {row['model']} | {row['dataset']} | {row['framing']} | T={row['temperature']} "
              f"| r{row['rep']} -> {tag}")
        api_fail_streak = api_fail_streak + 1 if str(rec["error"]).startswith("API error") else 0
        if api_fail_streak >= 5:
            print("STOP: 5 API errors in a row (probably a daily quota). Run again later; progress is saved.")
            break
        time.sleep(SLEEP_BETWEEN_CALLS["mock" if mock else MODELS[row["model"]]["provider"]])
    return pd.read_csv(res_path) if os.path.exists(res_path) else None


def pilot(runner, n_main=3, mock=False):
    """Quick check for your models. Writes to results_pilot_<runner>.csv (kept apart from real results).
    For each model: n_main synthetic-data runs at T=0 and at T=1.0 (temperature check) and 2 soccer runs."""
    manifest = load_manifest()
    path = f"{OUT_DIR}/results_pilot_{runner}.csv"
    for mname, m in MODELS.items():
        if m["runner"] != runner:
            continue
        specs = [("synthetic", 0.0)] * n_main + [("synthetic", 1.0)] * n_main + [("soccer", 0.7)] * 2
        for k, (ds, T) in enumerate(specs, 1):
            row = dict(run_id=f"PILOT|{mname}|{ds}|T{T}|{k}", model=mname, runner=runner, dataset=ds, variant=1,
                       framing="neutral", temperature=T, rep=k, session=0)
            rec = execute_run(row, manifest, mock=mock)
            append_row(rec, path)
            print(mname, ds, f"T={T}", "->", f"{rec['effect_size']:.3f}" if rec["success"] else f"FAILED {str(rec['error'])[:70]}")
            time.sleep(SLEEP_BETWEEN_CALLS["mock" if mock else m["provider"]])
    d = pd.read_csv(path)
    print("\nSuccess rate per model:\n", d.groupby("model")["success"].mean().round(2).to_string())
    print("\nSpread of effect_size on the synthetic data (T=1.0 should be wider than T=0):")
    print(d[d.dataset == "synthetic"].groupby(["model", "temperature"])["effect_size"].agg(["count", "std"]).round(3).to_string())
    return d
