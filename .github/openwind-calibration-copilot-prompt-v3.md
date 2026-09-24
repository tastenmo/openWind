# GitHub Copilot Instruction Prompt: OpenWInD Mouthpiece Calibration Notebook (v3 - 14.4mm Bore)

Role: Expert Musical Acoustics Engineer & Python Developer
Target Artifact: `openwind_mouthpiece_calibration.ipynb` (Jupyter Notebook)

---

## Objective
Create a fully functional, well-documented Jupyter Notebook (`openwind_mouthpiece_calibration.ipynb`) that uses **OpenWInD** (Open Wind Instrument Design library) to build a digital twin of a High D Soprano Tin Whistle mouthpiece and calibrate its effective acoustic end-correction ($\Delta l_{\text{mouth}}$ / `L_mouth`) against real-world measured fundamental and harmonic frequencies.

---

## Physical Background & Measured Data

- **Instrument Type:** High D Soprano Tin Whistle (Sopran-D Flöte)
- **Tube Material:** DN16 PVC pipe (16.0 mm outer diameter, 0.8 mm wall thickness)
- **Bore Diameter ($D$):** `14.4 mm` (`0.0144 m`)
- **Bore Radius ($r$):** `0.0072 m` (`7.2 mm`)
- **Physical Tube Length ($L_{\text{tube}}$):** `0.26361 m` (263.61 mm from labium edge to tube end)
- **Measured Fundamental Frequency ($D_5$ / Bell Note):** `587.33 Hz`
- **Measured 2nd Octave Frequency ($D_6$):** `1174.66 Hz`
- **Calibration Goal:** Adjust the virtual mouthpiece extension parameter `L_mouth` in OpenWInD so that the simulated first impedance peak $f_1$ matches $587.33\text{ Hz}$ exactly, and verify that the second impedance peak $f_2$ aligns with $1174.66\text{ Hz}$.

---

## Required Jupyter Notebook Structure

The generated notebook MUST contain the following 8 structured cells, complete with Markdown explanations and Python code:

### 1. Header & Dependencies (Markdown + Code)
- Explanatory Markdown introducing the acoustic calibration concept in OpenWInD.
- Imports: `numpy`, `matplotlib.pyplot`, `scipy.optimize`, `scipy.signal` (or `numpy.argmax`), and `openwind` (`Instrument`, `Player`, `Simu`).

### 2. Configuration & Measured Parameters (Code)
- Define clean Python constants:
  ```python
  R_BORE = 0.0072          # Inner radius in meters (14.4mm diameter: 16mm OD - 2*0.8mm wall)
  L_PHYSICAL = 0.26361     # Physical tube length from labium to tube end in meters
  TARGET_F1_D5 = 587.33    # Measured D5 fundamental frequency in Hz
  TARGET_F2_D6 = 1174.66   # Measured D6 second octave frequency in Hz
  TEMPERATURE = 20.0       # Air temperature in Celsius
  ```

### 3. OpenWInD Instrument Generator Function (Code)
- Define a function `create_whistle_model(L_mouth)`:
  - Construct a bore profile combining the virtual mouthpiece acoustic length `L_mouth` and the main body bore `[(0.0, R_BORE), (L_PHYSICAL + L_mouth, R_BORE)]`.
  - Instantiate `Instrument(bore=..., radiation='unflanged')`.
  - Return the `Instrument` object.

### 4. Impedance Spectrum & Peak Detection Helper (Code)
- Define `get_impedance_peaks(instrument, f_min=300, f_max=2000, n_points=2000)`:
  - Calculate input impedance $Z(f)$ using OpenWInD's `instrument.get_impedance()`.
  - Identify $f_1$ (first peak near D5) and $f_2$ (second peak near D6).
  - Return `(f1, f2, frequencies, Z)`.

### 5. Automated Calibration / Optimization Loop (Code)
- Formulate an objective cost function `objective(L_mouth)` that calculates $|f_1(L_{\text{mouth}}) - f_{\text{target\_D5}}|$.
- Use `scipy.optimize.minimize_scalar` or `scipy.optimize.root_scalar` (search interval `[0.005, 0.040]` m) to find the calibrated `L_mouth_opt`.
- Print formatted calibration results:
  - Calibrated $L_{\text{mouth}}$ in mm (Expected: ~24.2 mm).
  - Simulated $f_1$ vs Target $D_5$ ($587.33\text{ Hz}$) and error in Cents.

### 6. Validation & Octave Cent Offset (Code)
- Calculate the simulated 2nd harmonic $f_2$ with the calibrated $L_{\text{mouth\_opt}}$.
- Compute the octave ratio $\frac{f_2}{f_1}$ and the deviation from a pure octave in Cents:
  $$\text{Cent Deviation} = 1200 \times \log_2\left(\frac{f_2 / f_1}{2}\right)$$
- Print a summary confirming whether the octave interval matches the real-world measured $1175\text{ Hz}$ ($D_6$).

### 7. Publication-Quality Visualization (Code)
- Plot $|Z(f)|$ (Impedance magnitude) from $300\text{ Hz}$ to $2000\text{ Hz}$.
- Add vertical dashed lines for $f_1$ ($587.33\text{ Hz}$) and $f_2$ ($1174.66\text{ Hz}$) with peak annotations.
- Include a secondary subplot showing the phase $\arg(Z)$.
- Add titles, axis labels in German/English, grid, and legend.

### 8. Export Calibrated Model for Tonehole Optimization (Code)
- Save the calibrated parameters to a JSON configuration file `calibrated_whistle_parameters.json`.
- Output a ready-to-use Python snippet showing how to load this calibrated instrument model into OpenWInD's `OptimProblem` for downstream tonehole placement optimization.

---

## Code Quality & Formatting Guidelines

1. **Robust Error Handling:** Include fallback bounds if OpenWInD peak searching fails].
2. **Clear Markdown Annotations:** Explain the acoustic physics (e.g., end correction, acoustic impedance, tonehole lattice theory) above each code cell.
3. **Type Hints & Docstrings:** Write clean Python 3.10+ code with PEP 8 standards.
4. **Reproducibility:** Set fixed temperature ($20^\circ\text{C}$) and explicit solver settings.

---
*Generate the complete Jupyter Notebook code and text following these exact steps.*
