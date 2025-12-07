import json
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# ============================
# 0. File paths
# ============================
EP_SIM_PATH = "Ep_sim.npy"          # Ep_sim.npy file
MANIFEST_PATH = "Ep_manifest.json"  # Ep_manifest.json file


# ============================
# 1. Data loading
#   - X: (N, d) feature matrix
#   - roles: "EL" or "ES" (GroEL vs GroES)
#   - resnos: residue numbers (int)
#   - groups: GroupKFold groups (role:resno)
#   - residue_ids: strings like "A:34"
# ============================
def load_data(ep_sim_path=EP_SIM_PATH, manifest_path=MANIFEST_PATH):
    # Load environment vectors
    X = np.load(ep_sim_path)

    # Load manifest (we only need residue_ids here)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # IMPORTANT: read from 'residue_ids'
    residue_ids = manifest["residue_ids"]

    N = len(residue_ids)
    if X.shape[0] != N:
        raise ValueError(f"X has {X.shape[0]} rows but manifest lists {N} residues.")

    chains = []
    resnos = []
    for rid in residue_ids:
        chain, res = rid.split(":")
        chains.append(chain)
        resnos.append(int(res))

    chains = np.array(chains)
    resnos = np.array(resnos, dtype=int)

    # Infer GroEL vs GroES role per chain:
    #  - GroEL chains have length ≥ 400
    #  - GroES chains have length ~100
    chain_to_max = {}
    for c, r in zip(chains, resnos):
        chain_to_max[c] = max(chain_to_max.get(c, 0), r)

    roles = []
    for c in chains:
        max_res = chain_to_max[c]
        role = "EL" if max_res >= 400 else "ES"
        roles.append(role)
    roles = np.array(roles)

    # GroupKFold groups: (role, resno) as strings "EL:30" / "ES:27"
    groups = np.array([f"{role}:{res}" for role, res in zip(roles, resnos)])

    return X, roles, resnos, groups, residue_ids


# ============================
# 2. Tier1 class definitions
#   - Residues grouped into functional classes for GroEL / GroES
# ============================
GROEL_TIER1_CLASSES = {
    # Class A: ATPase core / allosteric relay
    # As requested: [34, 52, 87, 89, 398] + T30, K51, and the 31–33 loop
    "groel_A_ATP_core": [30, 31, 32, 33, 34, 51, 52, 87, 89, 398],

    # Class B: substrate / GroES hydrophobic patch
    "groel_B_substrate_patch": [199, 201, 203, 204, 234, 237, 263, 264],

    # Class C: hinge / pivot
    "groel_C_hinge_core": [192],
    "groel_C_hinge_support": [374, 375],

    # Class D: inter-ring / inter-subunit contacts
    "groel_D_inter_ring": [105, 197, 386, 434, 452, 461],
}

GROES_TIER1_CLASSES = {
    # Class E: IVL anchor (core)
    "groes_E_IVL_core": [25, 26, 27],

    # Class F: extended loop (support)
    "groes_F_loop_support": [23, 24, 28, 29, 30, 31],
}


# ============================
# 3. Label mask construction
# ============================
def make_label_mask(roles, resnos, class_resnos, role_filter=None):
    """
    Build a boolean mask for a given Tier1 class.

    roles       : (N,)  "EL"/"ES"
    resnos      : (N,)  int residue numbers
    class_resnos: list of residue numbers [30, 31, ...]
    role_filter : "EL" / "ES" / None
                  If not None, restrict positives to that role.
    """
    mask = np.isin(resnos, class_resnos)
    if role_filter is not None:
        mask &= (roles == role_filter)
    return mask


# ============================
# 4. GroupKFold-based AUROC
# ============================
def eval_auc(X, y, groups, n_splits=5, desc=""):
    """
    Evaluate AUROC using GroupKFold.

    X      : (N, d) feature matrix
    y      : (N,) 0/1 labels
    groups : (N,) group labels like "EL:30" or "ES:27"

    Returns:
        mean AUC, std AUC, list of per-fold AUCs
    """
    pos_groups = np.unique(groups[y == 1])

    # Reduce n_splits if there are too few positive groups
    if len(pos_groups) < n_splits:
        n_splits = max(2, len(pos_groups))

    gkf = GroupKFold(n_splits=n_splits)
    imputer = SimpleImputer(strategy="mean")
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    aucs = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        # Both train and test need at least 2 classes to define AUROC
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue

        X_tr = imputer.fit_transform(X[train_idx])
        X_te = imputer.transform(X[test_idx])

        clf.fit(X_tr, y[train_idx])
        proba = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y[test_idx], proba)
        aucs.append(auc)

    if not aucs:
        return np.nan, np.nan, []

    aucs = np.array(aucs)
    return float(aucs.mean()), float(aucs.std()), aucs.tolist()


# ============================
# 5. Main: compute all resolutions
# ============================
def main():
    X, roles, resnos, groups, residue_ids = load_data()
    N = X.shape[0]
    print(f"[INFO] N residues: {N}")
    print(f"[INFO] X shape   : {X.shape}")

    # --------------------------------
    # (1) Global Tier1 (all EL + ES)
    # --------------------------------
    y_global = np.zeros(N, dtype=int)
    all_classes = {}
    all_classes.update(GROEL_TIER1_CLASSES)
    all_classes.update(GROES_TIER1_CLASSES)

    for name, res_list in all_classes.items():
        role = "EL" if name.startswith("groel_") else "ES"
        y_global |= make_label_mask(roles, resnos, res_list, role_filter=role).astype(int)

    print("\n[Global Tier1] #positives:", int(y_global.sum()))
    auc, sd, folds = eval_auc(X, y_global, groups, desc="Global Tier1")
    print("[Global Tier1] AUC =", auc, "+/-", sd, "folds:", folds)

    # --------------------------------
    # (2) GroEL (EL) Tier1 overall
    # --------------------------------
    y_el = np.zeros(N, dtype=int)
    for name, res_list in GROEL_TIER1_CLASSES.items():
        y_el |= make_label_mask(roles, resnos, res_list, role_filter="EL").astype(int)

    mask_el = (roles == "EL")
    auc_el, sd_el, folds_el = eval_auc(X[mask_el], y_el[mask_el], groups[mask_el], desc="GroEL Tier1")
    print("\n[GroEL (EL) Tier1] #positives:", int(y_el.sum()))
    print("[GroEL (EL) Tier1] AUC =", auc_el, "+/-", sd_el, "folds:", folds_el)

    # --------------------------------
    # (3) GroES (ES) Tier1 overall
    # --------------------------------
    y_es = np.zeros(N, dtype=int)
    for name, res_list in GROES_TIER1_CLASSES.items():
        y_es |= make_label_mask(roles, resnos, res_list, role_filter="ES").astype(int)

    mask_es = (roles == "ES")
    auc_es, sd_es, folds_es = eval_auc(X[mask_es], y_es[mask_es], groups[mask_es], desc="GroES Tier1")
    print("\n[GroES (ES) Tier1] #positives:", int(y_es.sum()))
    print("[GroES (ES) Tier1] AUC =", auc_es, "+/-", sd_es, "folds:", folds_es)

    # --------------------------------
    # (4) Per-class AUC (GroEL / EL)
    # --------------------------------
    print("\n[Per-class AUC: GroEL (EL)]")
    for name, res_list in GROEL_TIER1_CLASSES.items():
        y = make_label_mask(roles, resnos, res_list, role_filter="EL").astype(int)
        auc_c, sd_c, folds_c = eval_auc(X[mask_el], y[mask_el], groups[mask_el], desc=name)
        print(
            f"  [{name}] #positives: {int(y.sum())}  "
            f"AUC = {auc_c} +/- {sd_c}   folds: {folds_c}"
        )

    # --------------------------------
    # (5) Per-class AUC (GroES / ES)
    # --------------------------------
    print("\n[Per-class AUC: GroES (ES)]")
    for name, res_list in GROES_TIER1_CLASSES.items():
        y = make_label_mask(roles, resnos, res_list, role_filter="ES").astype(int)
        auc_c, sd_c, folds_c = eval_auc(X[mask_es], y[mask_es], groups[mask_es], desc=name)
        print(
            f"  [{name}] #positives: {int(y.sum())}  "
            f"AUC = {auc_c} +/- {sd_c}   folds: {folds_c}"
        )


if __name__ == "__main__":
    main()