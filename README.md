
Current stable version: 4.7

4.7 introduces a Structure-of-Arrays (SoA) memory layout and global atom
indexing, allowing the pipeline to process very large complexes
(e.g. the 70S ribosome, >150k atoms) on a single 16 GB GPU.
Numerical outputs are equivalent to v4.6; only the memory layout and
performance characteristics have changed.

### ⚠️ Note on Reproducibility & Memory Efficiency
To reproduce the results reported in the manuscript (especially for large complexes like **Ribosome 6Q97**), please enable the `--no-pair-contacts` flag.

This optimization was applied to **all benchmarks** in the study to maximize memory efficiency (reducing complexity to $O(N)$) without affecting the accuracy of $E_p$ vectors.

# Example command for large complexes (e.g., Ribosome)
python 4.7.py 6Q97.cif --out-dir results --no-pair-contacts --block-size 64


# Protein Environment Representation

[![BioRxiv](https://img.shields.io/badge/BioRxiv-2025.04.29.651260-b31b1b.svg)](https://doi.org/10.1101/2025.04.29.651260)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Official implementation of "Localized Reactivity on Proteins as Riemannian Manifolds"**



# protein-env-representation
Environment-vector pipeline and case study scripts for residue-level reactivity analysis.
# Environment-vector case-study scripts (reproducibility note)

This repository contains the minimal Python scripts used to generate the
residue-level case-study results reported in the manuscript. The code is
deliberately lightweight: it assumes that the reader has access to the
precomputed environment-vector features and focuses on making the mapping
from “manuscript section → script → output numbers” explicit.

## 1. File overview

- `4.6.py` or `4.7.py`
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

- `6Q97.py`  
  Case-study script for the tmRNA–SmpB–ribosome rescue complex (PDB ID: **6Q97**).  
  Loads the precomputed environment features for all protein and RNA residues, constructs Tier-1/Tier-2 labels for tmRNA and SmpB pockets, 16S rRNA decoding-centre sites, the 23S rRNA peptidyl transferase centre, and helicase-like uS3/uS4/uS5 pockets, and assembles the curated hard-negative panel of 323 buried hydrophobic, electrostatic and stacking decoys.  
  Runs the random-forest evaluation protocol described in the Methods section (StratifiedKFold and GroupKFold) for both global tasks (Tier-1/Tier-2 vs all residues) and adversarial tasks (Tier-1/Tier-1+2 vs hard negatives). The AUC values reported by this script correspond to the 6Q97 rows in the main ribosome case-study table and the hard-negative analysis.

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

All experiments in this repository were run in a Google Colab environment
(December 2025). To make the setup fully reproducible, we distinguish between:

1. The **base Colab image** (before any manual `pip install`),
2. The **additional packages** explicitly installed for this project.

### 2.1 Core runtime (base Colab image)

- **Python**: `3.12.12`  
  - Build: `main, Oct 10 2025, 08:52:57`  
  - Compiler: `GCC 11.4.0`
- **OS**: `Linux-6.6.105+-x86_64-with-glibc2.35`

### 2.2 GPU / CUDA stack

- **GPU**: `Tesla T4`  
  - Driver: `550.54.15`  
  - Memory: `15360 MiB`
- **PyTorch**:
  - `torch`: `2.9.0+cu126`
- **CUDA runtime**: `12.6`
- **cuDNN**: `9.10.2.21`  (reported as `91002`)

### 2.3 Core scientific Python stack (base image)

The following packages are present in the base Colab image and were used
unchanged:

- `numpy`: `2.0.2`
- `scipy`: `1.16.3`
- `pandas`: `2.2.2`
- `matplotlib`: `3.10.0`
- `scikit-learn`: `1.6.1`

No external deep-learning framework is required for the main benchmarks:
random forests are implemented via

- `sklearn.ensemble.RandomForestClassifier`

with a fixed random seed and deterministic settings (see Methods).

### 2.4 Additional biomolecular / structure libraries (installed manually)

On top of the base image described above, we manually installed the
following packages and used them for all environment-vector computations
and structure handling:

- `mdtraj`: `1.11.0`
- `biopython`: `1.86`
- `rdkit`: `2025.09.3`
- `openmm`: `8.4.0.dev-4768436`

Any equivalent PDB/mmCIF reader or MD engine may work in principle, but
**all reported results** in this repository were generated under exactly
the versions listed here.

### 2.5 Full Colab package snapshot

The Colab image contains a large number of additional packages
(JAX, TensorFlow, RAPIDS, geospatial libraries, etc.) that are *not*
used by this codebase. For bit-level reproducibility of the runtime
actually used in the experiments, we store the full `pip freeze` of the
environment *after* installing the packages in §2.4 as:

- `environment_colab_full_2025-12.txt`

This file corresponds exactly to the output of:

```bash
pip freeze > environment_colab_full_2025-12.txt

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
