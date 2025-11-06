# ⚙️ Stokes Wave Solutions

This repository provides **three GUI-based tools** for computing deep-water Stokes waves using hybrid machine learning models. The workflow consists of three stages:

1. **Generate coefficients** \(a_j\) **and Bernoulli constant** \(K\)  
2. **Predict conformal mapping variables** \(\theta\) **and** \(R\)  
3. **Compute velocity field** \((v_x, v_y)\)

All tools are packaged as standalone `.exe` files for easy use **without** a Python environment.

---

## 📦 Executables Overview

| EXE File Name         | Purpose                                     | Input                      | Output                         |
| --------------------- | ------------------------------------------- | -------------------------- | ------------------------------ |
| `cal_coefficient.exe` | Generate Fourier coefficients \(a_j\)     | Steepness value            | `Solution_steepness_XXX.txt`   |
| `inverse_mapping.exe` | Generate coordinates \(\theta, R\)       | Steepness + XY file (x, y) | `thetaR_output.txt`            |
| `velocity_field.exe`  | Compute velocity field                      | `a.txt` + `thetaR.txt`     | `velocity_field.dat` (Tecplot) |

---

## 1️⃣ `cal_coefficient.exe`: Generate Wave Coefficients

### 📅 Input
- A float number: wave steepness \(s \in [0, 0.141]\)

### 📄 Output
- A text file `Solution_steepness_XXX.txt`, containing:
  - \(r\) Fourier coefficients \(a_1, a_2, \dots, a_r\)
  - 1 line: Bernoulli constant \(K\)

### 📝 Format Example
```
0.20841
0.19157
...
1.0332   ← K on last line
```

### ▶️ How to Use
1. Run `cal_coefficient.exe`
2. Enter the desired steepness (e.g., `0.070`)
3. Save the output file

---

## 2️⃣ `inverse_mapping.exe`: Generate coordinates \(\theta, R\)

### 📅 Input
- Steepness value \(steepness\)
- A text file of \(x, y\) pairs:
```
x₁ y₁
x₂ y₂
...
```

### 📄 Output
- A `.txt` file with 4 columns: \(x, y, \theta, R\)

### ▶️ How to Use
1. Run `inverse_mapping.exe`
2. Enter steepness value
3. Select `XY input file`
4. Save the predicted output (e.g., `thetaR_output.txt`)

---

## 3️⃣ `velocity_field.exe`: Compute Velocity Field

### 📅 Input
- `a.txt` file: output from `cal_coefficient.exe`
- `thetaR.txt` file: output from `inverse_mapping.exe`

These files are automatically parsed to obtain:
- \(r =\) number of \(a_j\)
- \(K\)
- Each line: \(\theta, R\)

### 📄 Output
- `velocity_field.dat`: ready for **Tecplot** with 5 columns:
  - \(x, y, v_x, v_y, v = \sqrt{v_x^2 + v_y^2}\)

### 📜 Tecplot Header
```
VARIABLES = "X","Y","VX","VY","V"
ZONE F=POINT N=...
x1    y1    vx1    vy1    v1
x2    y2    vx2    vy2    v2
...
```

### ▶️ How to Use
1. Run `velocity_field.exe`
2. Select the `a-coefficients file`
3. Select the `thetaR.txt` file`
4. Save result as `velocity_field.dat`

---

## 💡 Dependencies
> All `.exe` files are standalone. You do **not** need Python installed.

Internally built using:
- PyTorch
- NumPy
- Tkinter

---

## 📈 Example GUI Workflow
```
1. Run cal_coefficient.exe      → get Solution_steepness_0.070.txt
2. Run inverse_mapping.exe      → input (x, y) → get thetaR_output.txt
3. Run velocity_field.exe       → use above 2 files → get velocity_field.dat
```
Visualize `velocity_field.dat` in Tecplot or ParaView.

---

## 📂 File Naming Convention
- `Solution_steepness_XXX.txt` – Fourier series coefficients for steepness XXX
- `thetaR_output.txt` – θ and R values for input (x, y)
- `velocity_field.dat` – Velocity components in physical space

---

## 💻 Supported Platforms
| Platform             | Supported | Notes                                                        |
| -------------------- | --------- | ------------------------------------------------------------ |
| **Windows (x64)**    | ✅        | Fully supported                                              |

> ⚠️ `.exe` files will **not run** on macOS directly.

---

# 🧪 Python / CLI Usage (Reproducible)

In addition to the GUI tools, this repository provides **four python scripts** for research‑grade training and batch inference. They mirror the GUI functionality and enable reproducible experiments.

| CLI Script                         | Purpose                                               | Mirrors GUI |
|-----------------------------------|-------------------------------------------------------|-------------|
| `ml_cal_coefficient_train.py`     | Train the network   wave steepness → \(a_j\) (+ constant K) | – |
| `ml_cal_coefficient_infer.py`     | Fast prediction of \(a_j\) and K from a column of inputs  | `cal_coefficient.exe` |
| `inverse_map/train.py`            | Train the inverse mapping network to predict \((\theta,R)\) | – |
| `inverse_map/infer.py`            | Batch prediction of \((\theta,R)\), optional evaluation & formatting | `inverse_mapping.exe` |

---

## 📦 Environment
```bash
pytorch numpy 
# Scripts set a fixed random seed (42) and support AMP/Early‑Stopping/CosineLR
```

---

## 4️⃣ `ml_cal_coefficient_train.py`: Train Coefficients Predictor

**Input format** (TXT/CSV):
```
x, a1, a2, ..., aj, K     # x = single input (steepness); a1..K = target coefficients (e.g., number=2001)
```

**Example**
```bash
python ml_cal_coefficient_train.py   --train data/train.txt   --test  data/test.txt   --out   outputs/series_run1   --epochs 2000 --batch-size 64 --lr 1e-3   --hidden 64 --layers 12 --out-dim 2001
```

**Artifacts**
- `checkpoints/{best,last}.pth`, `history.json`, `figs/*`, optional `pred_test_best.csv`, `test_metrics.json`

---

## 5️⃣ `ml_cal_coefficient_infer.py`: Predict Coefficients (CLI)

**Input**
- X‑only: a single column of inputs (steepness)
- X+Y: if ground truth is appended, metrics will be computed

**Example**
```bash
python ml_cal_coefficient_infer.py   --weights outputs/series_run1/checkpoints/best.pth   --input   data/input.txt   --out     outputs/series_infer   --out-dim 2001 --hidden 64 --layers 12
```

**Output**
- `output.csv` (and `metrics_*.json` if targets are present)

---

## 6️⃣ `inverse_map_train.py`: Train Inverse Mapping

**Input format** (CSV/TXT with `in_dim + out_dim` columns; default `2003 + 2`):
- First `in_dim` columns: input features
- Last `out_dim` columns: targets \((\theta, R)\)

**Example**
```bash
python inverse_map_train.py   --train data/train.csv   --test  data/test.csv    --out   outputs/inverse_map_run1   --in-dim 2003 --out-dim 2   --widths 1024,512,256,64,32,8   --epochs 2000 --batch-size 32768 --lr 1e-3 --amp
```

**Artifacts**
- `checkpoints/{best,last}.pth`, `history.json`, `figs/*`

---

## 7️⃣ `inverse_map_infer.py`: Predict \((\theta, R)\) (CLI)

**Input**
- At least `in_dim` columns; if the last `out_dim` columns contain ground truth, add `--eval-if-targets`

**Useful options**
- `--clip i:lo,hi` — clamp output dimension `i` to `[lo, hi]` (e.g., angle ∈ [−π, π], radius ∈ [1e−4, 1])

**Example (batch two files)**
```bash
python inverse_map_infer.py   --weights outputs/inverse_map_run1/checkpoints/best.pth   --input   input.csv   --out     outputs   --in-dim 2003 --out-dim 2 --widths 1024,512,256,64,32,8   --copy-last 4 --prepend-first 2001   --clip 0:-3.141592653589793,3.141592653589793   --clip 1:0.0001,1.0   --eval-if-targets
```

**Output**
- `pred_<input_stem>.csv` (with your chosen formatting), and optional `metrics_<input_stem>.json`

---

## 📬 Contact
Please open issues or pull requests for bug reports and feature requests.
