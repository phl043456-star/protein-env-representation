#!/usr/bin/env python
"""
Rubisco 4RUB residue-level evaluation script (v3, English-annotated)

- Loads Ep_sim.npy and Ep_manifest.json
- Supports manifest formats:
    * {"residues": [...]}                    (list of dicts with chain/resid info)
    * ["A:9", "A:10", ...]                   (simple list of residue IDs)
    * {"residue_ids": ["A:9", "A:10", ...]}  (4RUB-style format)

- Normalises everything to:
    * res_ids  : numpy array of "CHAIN:RESID" strings
    * chains   : array of chain IDs
    * resnos   : array of residue numbers (as strings, integer-normalised if possible)
    * groups   : integer site-group labels for GroupKFold
                 (role + residue index, e.g. "L:175", "S:43")

- Defines Tier1 / Tier2 label sets for Rubisco 4RUB:
    * L-subunit (chains A/B/C/D)
    * S-subunit (chains S/T/U/V)
    * per-class Tier1/Tier2
    * union sets (L_Tier1_all, global_Tier1_all, etc.)
    * functional-type unions (Type_T1T2)

- Evaluates each subset with:
    * StratifiedKFold (residue-level cross-validation)
    * GroupKFold (site-grouped, based on role+resno)
"""

import json
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. Load feature matrix and manifest
# ============================================================

EP_SIM_PATH = "Ep_sim.npy"
MANIFEST_PATH = "Ep_manifest.json"

print("[INFO] Loading Ep_sim.npy ...")
X = np.load(EP_SIM_PATH)
print(f"[INFO] Ep_sim shape: {X.shape}")

print("[INFO] Loading Ep_manifest.json ...")
with open(MANIFEST_PATH, "r") as f:
    manifest_raw = json.load(f)

# --- normalise manifest into a list of residue entries ---

if isinstance(manifest_raw, dict):
    if "residues" in manifest_raw:
        # Case: {"residues": [ {...}, {...}, ... ]}
        residues_raw = manifest_raw["residues"]
    elif "residue_ids" in manifest_raw:
        # Case: {"residue_ids": ["A:9", "A:10", ...], ...}
        residues_raw = [{"id": rid} for rid in manifest_raw["residue_ids"]]
    else:
        raise ValueError(
            "[ERROR] Could not find 'residues' or 'residue_ids' in Ep_manifest.json dict.\n"
            f"Available keys = {list(manifest_raw.keys())}"
        )
elif isinstance(manifest_raw, list):
    # Case: ["A:9", "A:10", ...] or [{"id": ...}, ...]
    residues_raw = manifest_raw
else:
    raise ValueError(
        "[ERROR] Unrecognised Ep_manifest.json format.\n"
        "Expected one of:\n"
        "  - dict with 'residues' or 'residue_ids'\n"
        "  - list of strings or dicts."
    )

res_ids = []
chains = []
resnos = []


def build_res_id_from_entry(entry):
    """
    Extract a residue ID in the canonical form 'CHAIN:RESNO' from a manifest entry.

    Supported forms:
      - string:          'A:175'
      - dict with 'id':  {'id': 'A:175'}
      - dict with chain / resid:
            {'chain_id': 'A', 'resid': 175}
            {'chain': 'A', 'resseq': 175}
      - other variants with keys from:
            chain_key_candidates = ["chain_id", "chain", "chainID", "chain_name"]
            resid_key_candidates = ["resid", "resseq", "res_id", "resseq_number"]
    """
    # 1) Plain string like "A:175"
    if isinstance(entry, str):
        return entry

    # 2) Dict-like entry
    if isinstance(entry, dict):
        # 2a) Direct 'id' key
        if "id" in entry:
            return str(entry["id"])

        # 2b) Try to reconstruct from chain/resid keys
        chain_key_candidates = ["chain_id", "chain", "chainID", "chain_name"]
        resid_key_candidates = ["resid", "resseq", "res_id", "resseq_number"]

        chain_val = None
        resid_val = None

        for ck in chain_key_candidates:
            if ck in entry:
                chain_val = str(entry[ck]).strip()
                break
        for rk in resid_key_candidates:
            if rk in entry:
                resid_val = str(entry[rk]).strip()
                break

        if chain_val is not None and resid_val is not None:
            return f"{chain_val}:{resid_val}"

    # 3) If all attempts fail, abort with a helpful error
    raise ValueError(
        "[ERROR] Could not extract chain/residue information from manifest entry.\n"
        f"Offending entry: {entry}"
    )


for idx, entry in enumerate(residues_raw):
    rid = build_res_id_from_entry(entry)
    if ":" not in rid:
        raise ValueError(
            f"[ERROR] Residue ID '{rid}' does not match 'CHAIN:RESID' format."
        )
    ch, rn = rid.split(":", 1)
    ch = ch.strip()
    rn = rn.strip()

    # Try to normalise residue index to a plain integer (no insertion codes)
    try:
        rn_int = int(rn)
        rn_str = str(rn_int)
    except ValueError:
        # If it cannot be parsed as an integer, keep the original string
        rn_str = rn

    res_id_norm = f"{ch}:{rn_str}"
    res_ids.append(res_id_norm)
    chains.append(ch)
    resnos.append(rn_str)

res_ids = np.array(res_ids)
chains = np.array(chains)
resnos = np.array(resnos)

N = len(res_ids)
print(f"[INFO] Manifest residues: N = {N}")
print(f"[INFO] Example residue: {res_ids[0]} (chain={chains[0]}, resid={resnos[0]})")

if X.shape[0] != N:
    raise ValueError(
        f"[ERROR] Ep_sim row count ({X.shape[0]}) does not match manifest residue count ({N})."
    )

# ============================================================
# 2. Define site-group labels (role + residue index)
#    GroupKFold: grouping across symmetry-related copies
# ============================================================

L_CHAINS = {"A", "B", "C", "D"}  # large subunit chains
S_CHAINS = {"S", "T", "U", "V"}  # small subunit chains


def role_for_chain(ch):
    """
    Map raw chain ID to a 'role' label:
      - L chains → 'L'
      - S chains → 'S'
      - anything else → use the chain ID itself as the role
    """
    if ch in L_CHAINS:
        return "L"
    if ch in S_CHAINS:
        return "S"
    return ch


group_keys = []
for ch, rn in zip(chains, resnos):
    role = role_for_chain(ch)
    group_keys.append(f"{role}:{rn}")

unique_groups, group_indices = np.unique(group_keys, return_inverse=True)
groups = group_indices  # integer group labels, shape (N,)

print(f"[INFO] Unique site-groups (role+resno): {len(unique_groups)}")

# ============================================================
# 3. Rubisco 4RUB label definitions (per-chain residue index)
#    Tier1 / Tier2 for L- and S-subunits, plus unions.
# ============================================================

rubisco_4RUB_labels = {
    # ----- L-subunit Tier 1 -----
    "L_Tier1_catalytic_core": [
        20, 60, 65, 123,
        175, 177, 201, 203, 204,
        294, 295, 327, 334, 335,
    ],
    "L_Tier1_PTM_core": [201],
    "L_Tier1_gate_latch": [327, 334, 473],
    "L_Tier1_PPI_Rca_core": [1, 2, 3, 4],

    # ----- L-subunit Tier 2 -----
    "L_Tier2_catalytic_shell": [66, 379, 380, 381, 402, 403, 404],
    "L_Tier2_PTM_shell": [104, 256],
    "L_Tier2_PPI_Rca_extension": [5, 6, 7, 8],

    # ----- S-subunit Tier 1 -----
    "S_Tier1_helix8_cluster": [43, 73, 78, 79, 81, 92],
    "S_Tier1_interface_betaAB_core": [16, 18, 32],

    # ----- S-subunit Tier 2 -----
    "S_Tier2_betaAB_regulatory": [59, 67, 68, 69, 71],
    "S_Tier2_helix8_shell": [19, 54, 84],
}

chains_L = ["A", "B", "C", "D"]
chains_S = ["S", "T", "U", "V"]


def expand_chain_res(label_key):
    """
    Take a label key in rubisco_4RUB_labels and expand it to all
    symmetry-related copies in the 4RUB complex.

    Example:
        "L_Tier1_catalytic_core": [20, 60, ...]
    becomes
        {"A:20", "B:20", "C:20", "D:20", ...}
    """
    if label_key.startswith("L_"):
        use_chains = chains_L
    elif label_key.startswith("S_"):
        use_chains = chains_S
    else:
        use_chains = chains_L + chains_S

    res_list = rubisco_4RUB_labels[label_key]
    out = set()
    for ch in use_chains:
        for r in res_list:
            out.add(f"{ch}:{int(r)}")
    return out


subset_pos_sets = {}

for key in rubisco_4RUB_labels.keys():
    subset_pos_sets[key] = expand_chain_res(key)

# ------------------------------------------------------------
# 3.1 Union subsets (Tier1_all / Tier2_all / global)
# ------------------------------------------------------------

subset_pos_sets["L_Tier1_all"] = (
    subset_pos_sets["L_Tier1_catalytic_core"]
    | subset_pos_sets["L_Tier1_PTM_core"]
    | subset_pos_sets["L_Tier1_gate_latch"]
    | subset_pos_sets["L_Tier1_PPI_Rca_core"]
)

subset_pos_sets["L_Tier2_all"] = (
    subset_pos_sets["L_Tier2_catalytic_shell"]
    | subset_pos_sets["L_Tier2_PTM_shell"]
    | subset_pos_sets["L_Tier2_PPI_Rca_extension"]
)

subset_pos_sets["S_Tier1_all"] = (
    subset_pos_sets["S_Tier1_helix8_cluster"]
    | subset_pos_sets["S_Tier1_interface_betaAB_core"]
)

subset_pos_sets["S_Tier2_all"] = (
    subset_pos_sets["S_Tier2_betaAB_regulatory"]
    | subset_pos_sets["S_Tier2_helix8_shell"]
)

subset_pos_sets["global_Tier1_all"] = (
    subset_pos_sets["L_Tier1_all"] | subset_pos_sets["S_Tier1_all"]
)

subset_pos_sets["global_Tier2_all"] = (
    subset_pos_sets["L_Tier2_all"] | subset_pos_sets["S_Tier2_all"]
)

# ------------------------------------------------------------
# 3.2 Functional-type unions of Tier1 + Tier2 (Type_T1T2)
# ------------------------------------------------------------

subset_pos_sets["L_Type_catalytic_T1T2"] = (
    subset_pos_sets["L_Tier1_catalytic_core"]
    | subset_pos_sets["L_Tier2_catalytic_shell"]
)

subset_pos_sets["L_Type_PTM_T1T2"] = (
    subset_pos_sets["L_Tier1_PTM_core"]
    | subset_pos_sets["L_Tier2_PTM_shell"]
)

subset_pos_sets["L_Type_PPI_Rca_T1T2"] = (
    subset_pos_sets["L_Tier1_PPI_Rca_core"]
    | subset_pos_sets["L_Tier2_PPI_Rca_extension"]
)

subset_pos_sets["S_Type_betaAB_T1T2"] = (
    subset_pos_sets["S_Tier1_interface_betaAB_core"]
    | subset_pos_sets["S_Tier2_betaAB_regulatory"]
)

subset_pos_sets["S_Type_helix8_T1T2"] = (
    subset_pos_sets["S_Tier1_helix8_cluster"]
    | subset_pos_sets["S_Tier2_helix8_shell"]
)

print("[INFO] Label subsets defined:")
for name, s in subset_pos_sets.items():
    print(f"  - {name}: {len(s)} positives (chain:res pairs)")

# ============================================================
# 4. Evaluation utilities (aligned with Methods section)
# ============================================================


def evaluate_subset(X, y, groups, subset_name, use_groupk=True):
    """
    Evaluate one subset (e.g. L_Tier1_catalytic_core) using:

      - StratifiedKFold at residue level
      - GroupKFold (optional) with site-grouped folds (role+resno)

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix (Ep_sim).
    y : np.ndarray, shape (N,)
        Binary labels (0/1) for this subset.
    groups : np.ndarray, shape (N,)
        Integer site-group labels for GroupKFold.
    subset_name : str
        Name of the subset (used in logs/results).
    use_groupk : bool
        Whether to run GroupKFold in addition to StratifiedKFold.

    Returns
    -------
    result : dict
        {
          "subset": subset_name,
          "n_pos": n_pos,
          "n_neg": n_neg,
          "auc_strat_mean": float or None,
          "auc_strat_std": float or None,
          "auc_group_mean": float or None,
          "auc_group_std": float or None,
        }
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    N = len(y)

    result = {
        "subset": subset_name,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc_strat_mean": None,
        "auc_strat_std": None,
        "auc_group_mean": None,
        "auc_group_std": None,
    }

    # Degenerate case: all positive or all negative
    if n_pos == 0 or n_pos == N:
        print(f"[WARN] {subset_name}: degenerate labels (n_pos={n_pos}). Skipping.")
        return result

    # --------------------------
    # 4.1 StratifiedKFold
    # --------------------------
    k_strat = min(5, n_pos, n_neg)
    if k_strat < 2:
        print(
            f"[WARN] {subset_name}: StratifiedKFold not feasible "
            f"(n_splits={k_strat} < 2)."
        )
    else:
        skf = StratifiedKFold(
            n_splits=k_strat,
            shuffle=True,
            random_state=42,  # fixed seed for reproducibility
        )
        aucs = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            y_train, y_test = y[train_idx], y[test_idx]

            # Need both classes in train and test
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(
                    f"[WARN] {subset_name}: Stratified fold {fold_idx} has "
                    "a single class in train or test → skipping fold."
                )
                continue

            imputer = SimpleImputer(strategy="mean")
            X_train_imp = imputer.fit_transform(X[train_idx])
            X_test_imp = imputer.transform(X[test_idx])

            clf = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,  # fixed seed
                n_jobs=1,
            )
            clf.fit(X_train_imp, y_train)
            proba = clf.predict_proba(X_test_imp)[:, 1]
            auc = roc_auc_score(y_test, proba)
            aucs.append(auc)

        if len(aucs) > 0:
            result["auc_strat_mean"] = float(np.mean(aucs))
            result["auc_strat_std"] = float(np.std(aucs, ddof=1))
        else:
            print(
                f"[WARN] {subset_name}: all StratifiedKFold folds were skipped → AUC_strat = NA."
            )

    # --------------------------
    # 4.2 GroupKFold (site-grouped, role+resno)
    # --------------------------
    if use_groupk:
        pos_groups = np.unique(groups[y == 1])
        n_pos_groups = len(pos_groups)
        if n_pos_groups < 2:
            print(
                f"[INFO] {subset_name}: positive site-groups = {n_pos_groups}, "
                "skipping GroupKFold (Stratified only)."
            )
        else:
            n_groups_total = len(np.unique(groups))
            k_group = min(5, n_pos_groups, n_groups_total)
            if k_group < 2:
                print(
                    f"[INFO] {subset_name}: GroupKFold n_splits={k_group} < 2, skipping."
                )
            else:
                gkf = GroupKFold(n_splits=k_group)
                aucs_g = []
                for fold_idx, (train_idx, test_idx) in enumerate(
                    gkf.split(X, y, groups), start=1
                ):
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Need both classes in train and test
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                        print(
                            f"[WARN] {subset_name}: GroupKFold fold {fold_idx} has "
                            "a single class in train or test → skipping fold."
                        )
                        continue

                    imputer = SimpleImputer(strategy="mean")
                    X_train_imp = imputer.fit_transform(X[train_idx])
                    X_test_imp = imputer.transform(X[test_idx])

                    clf = RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=42,  # fixed seed
                        n_jobs=1,
                    )
                    clf.fit(X_train_imp, y_train)
                    proba = clf.predict_proba(X_test_imp)[:, 1]
                    auc = roc_auc_score(y_test, proba)
                    aucs_g.append(auc)

                if len(aucs_g) > 0:
                    result["auc_group_mean"] = float(np.mean(aucs_g))
                    result["auc_group_std"] = float(np.std(aucs_g, ddof=1))
                else:
                    print(
                        f"[WARN] {subset_name}: all GroupKFold folds invalid "
                        "→ AUC_group = NA."
                    )

    return result


# ============================================================
# 5. Run evaluation for all subsets
# ============================================================

results = []

print("\n============================================================")
print("[INFO] Start evaluation for all Rubisco 4RUB subsets")
print("============================================================\n")

for subset_name, pos_set in subset_pos_sets.items():
    print("------------------------------------------------------------")
    print(f"[INFO] Evaluating subset: {subset_name}")
    y = np.array([1 if rid in pos_set else 0 for rid in res_ids], dtype=int)
    res = evaluate_subset(X, y, groups, subset_name, use_groupk=True)
    results.append(res)

# ============================================================
# 6. Summary printout
# ============================================================

print("\n############################################################")
print("# Summary (Rubisco 4RUB Tier1/Tier2 + Type_T1T2 subsets)")
print("# subset, n_pos, n_neg, AUC_strat (mean±sd), AUC_group (mean±sd)")
print("############################################################")

for r in results:
    name = r["subset"]
    n_pos = r["n_pos"]
    n_neg = r["n_neg"]

    if r["auc_strat_mean"] is None:
        strat_str = "NA"
    else:
        strat_str = f"{r['auc_strat_mean']:.3f} ± {r['auc_strat_std']:.3f}"

    if r["auc_group_mean"] is None:
        group_str = "NA"
    else:
        group_str = f"{r['auc_group_mean']:.3f} ± {r['auc_group_std']:.3f}"

    print(
        f"{name:24s} | {n_pos:4d} pos, {n_neg:4d} neg | "
        f"Strat: {strat_str:15s} | Group: {group_str:15s}"
    )

print("\n[INFO] Evaluation completed.")