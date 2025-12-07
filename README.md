# protein-env-representation
Environment-vector pipeline and case study scripts for residue-level reactivity analysis.
# Environment-vector case-study scripts (reproducibility note)

This repository contains the minimal Python scripts used to generate the
residue-level case-study results reported in the manuscript. The code is
deliberately lightweight: it assumes that the reader has access to the
precomputed environment-vector features and focuses on making the mapping
from “manuscript section → script → output numbers” explicit.

## 1. File overview

- `4.6.py`  
  Core environment-vector pipeline. Implements the construction of the
  local environment representation \(E_p\) from input structures (e.g.
  PDB/mmCIF files) and writes out feature matrices such as
  `Ep_sim.npy` together with metadata (`Ep_manifest.json`, etc.).
  All case-study scripts below assume that these feature files have
  already been generated.

- `4rub.py`  
  Case-study script for Rubisco (PDB ID: **4RUB**).  
  Loads the Rubisco environment features, constructs Tier-1/Tier-2
  labels for the large and small subunits, and runs the random-forest
  evaluation protocol described in the Methods section (StratifiedKFold
  and GroupKFold). The AUC values reported by this script correspond
  to the Rubisco rows in the main Rubisco case-study table.

- `2c7c.py`  
  Case-study script for the GroEL/GroES chaperonin complex
  (PDB ID: **2C7C**).  
  Loads the 2C7C feature matrix, assigns GroEL/GroES roles, defines the
  Tier-1 classes (ATPase core, hydrophobic patch, hinge support,
  inter-ring, IVL anchor, loop support), and evaluates residue-level
  and class-resolved AUCs under the site-grouped protocol.

- `1TSR.py`  
  Case-study script for the p53–DNA complex (PDB ID: **1TSR**).  
  Loads the 1TSR features, defines protein-side Tier-1/Tier-2 labels
  (DNA-contact core, Zn cluster, fitness/stability core, allosteric/PPI
  pivots, structural shell) and DNA-side core/shell nucleotide labels,
  and reproduces the AUC values reported in the p53 case-study section.

- `1TF5.py`  
  Case-study script for the SecA ATPase motor and preprotein clamp
  (PDB ID: **1TF5**).  
  Loads the 1TF5 features, defines the ATPase, clamp, and C-terminal
  relay/interface modules with Tier-1 core and Tier-2 shell residues,
  and computes the corresponding residue-level AUCs.

Each case-study script is self-contained: given the appropriate
environment-vector files on disk, running the script will print the
numerical results quoted in the corresponding case-study subsection of
the manuscript (global Tier-1/Tier-2 AUCs and, where applicable,
module-wise or class-wise AUCs).

## 2. Dependencies

All scripts use a standard scientific Python stack. At minimum you will
need:

- Python 3.x
- `numpy`
- `scipy`
- `scikit-learn`

The environment-vector pipeline (`4.6.py`) and some case-study scripts
also rely on a structure/trajectory library (e.g. **MDTraj**, **MDAnalysis**
or an equivalent PDB/mmCIF reader) and basic I/O utilities (`json`,
`argparse`, etc.). The exact list of required packages can be read off
from the `import` statements at the top of each script.

No external machine-learning framework (beyond `scikit-learn`) is used.
Random forests are implemented via `sklearn.ensemble.RandomForestClassifier`
with a fixed random seed and deterministic settings, as described in the
Methods section.

## 3. Generating environment features

The scripts do **not** recompute the environment vectors by default; they
assume that the features for each PDB entry have already been generated
by `4.6.py` and written to disk.

A typical workflow is:

1. Prepare a cleaned PDB/mmCIF file for the complex of interest
   (e.g. `4rub_clean.pdb`, `2c7c_clean.pdb`, `1tf5_clean.pdb`,
   `1tsr_clean.pdb`).

2. Use `4.6.py` to compute the environment vectors and save them to a
   directory associated with that structure. The script expects to be
   edited locally so that the input path(s), output directory, grid /
   cutoff parameters, and boolean flags match your environment. After a
   successful run, you should have at least:
   - `Ep_sim.npy` (feature matrix),
   - `Ep_manifest.json` (residue metadata: chain IDs, residue numbers,
     roles, etc.).

3. Confirm that the shapes and residue counts reported by `4.6.py` match
   those in the manuscript for that complex.

The case-study scripts then load these precomputed files by path.

## 4. Reproducing the case-study numbers

Each case-study script follows the same pattern:

1. **Configure paths.**  
   At the top of the script there are typically a few constants that
   specify where to find `Ep_sim.npy`, `Ep_manifest.json` and any
   label-definition files. Adjust these paths to point to your local
   directories.

2. **Run the script.**  
   From a shell:

   ```bash
   python 4rub.py
   python 2c7c.py
   python 1TF5.py
   python 1TSR.py
