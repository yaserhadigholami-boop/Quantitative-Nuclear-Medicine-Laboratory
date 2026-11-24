# Quantitative Nuclear Medicine Laboratory
### Version 3.3.4

An interactive Streamlit teaching platform for quantitative nuclear medicine, gamma-counter physics, radiometric measurements, detector calibration, normalisation, sample preparation, biodistribution, gamma spectroscopy and PET/SPECT cross-calibration.

The application combines scientific teaching material with interactive simulations and quantitative analysis tools. Students can explore the underlying measurement physics, perform virtual laboratory experiments, generate synthetic measurement datasets, and analyse their results within a single environment.

---

## Overview

The laboratory is organised into two main learning stages.

**Stage 1 – Experiment and Acquisition**

Students explore the scientific principles and perform simulated experiments using the virtual measurement environment. Experiments include calibration, detector normalisation, geometry and matrix effects, tissue counting, gamma spectroscopy and PET/SPECT comparisons.

**Stage 2 – Analysis and Visualisation**

Measurement datasets generated in Stage 1 can be selected for further analysis, including regression, calibration assessment, detector comparisons, quantitative plotting and data export.

The application is designed to support practical teaching in quantitative nuclear medicine and radiopharmaceutical research.

## Key Features

### 1. Gamma-counter physics and instrumentation
- Interactive explanation of gamma-counter detection mechanisms.
- Scintillation detection using NaI(Tl).
- Photomultiplier tube operation and signal processing.
- Pulse-height analysis and multichannel analyser principles.
- Isotope-informed gamma spectra with photopeaks, Compton features and other illustrative spectral components.
- Interactive instrument fault-finding and troubleshooting.

### 2. Live acquisition simulation
- Progressive spectrum generation using a 2048-channel model.
- Cumulative counts versus acquisition time.
- Interval count-rate measurements (counts per second).
- Poisson counting-statistical uncertainty.
- Radioactive decay during acquisition.
- Background and dead-time effects.
- Interactive acquisition scenarios.

### 3. Calibration and detector normalisation
- Multi-point calibration using radioactive reference standards.
- Linear regression, sensitivity estimation and intercept calculation.
- Assessment of calibration linearity and goodness of fit.
- Detector-to-detector sensitivity comparisons.
- Relative detector normalisation.
- Geometry- and matrix-dependent response corrections.

### 4. Sample preparation and biodistribution
- Sample geometry and vial-volume simulations.
- Investigation of positioning and matrix effects.
- Simulated tissue counting and biodistribution measurements.
- Activity concentration, percentage injected dose (%ID) and %ID/g calculations.
- Hypothetical tumour-enriched and tumour-reduced uptake scenarios.

### 5. Gamma spectroscopy
- Interactive energy spectra.
- Energy calibration and peak identification.
- Photopeak and region-of-interest (ROI) analysis.
- Spectral resolution and peak broadening.
- Background and continuum contributions.
- Illustrative isotope-specific spectral features.

### 6. PET/SPECT cross-calibration
- Import of PET/SPECT and gamma-counter measurement datasets.
- Activity and activity-concentration comparisons.
- Decay correction and reference-time alignment.
- Recovery-coefficient considerations.
- Scatter plots and Bland–Altman agreement analysis.

### 7. Quantitative data analysis
- Interactive measurement tables.
- Calibration regression and residual assessment.
- Comparison of experimental scenarios.
- Customisable plots and visualisation.
- Export of generated datasets for further analysis.

---

## Version 3.3.4 – Final Refinements

This release includes improvements to the live acquisition visualisation, measurement equations and quantitative analysis.

### Live cumulative-count visualisation

The cumulative-count plot now dynamically adjusts its elapsed-time and count axes as the acquisition progresses.

Previously, the graph displayed the full eventual acquisition range from the beginning, making early changes difficult to observe.

The revised display progressively expands the visible time and count ranges as measurements accumulate.

### Interval count-rate visualisation

The interval-CPS plot now calculates the count rate independently for each acquisition time bin:

\[
R_i = \frac{N_i}{\Delta t}
\]

where:

- \(R_i\) is the measured count rate in interval \(i\), in counts per second (cps).
- \(N_i\) is the number of recorded events in interval \(i\).
- \(\Delta t\) is the duration of the interval, in seconds.

Approximate Poisson counting uncertainty is displayed using:

\[
\sigma_{R_i} = \frac{\sqrt{N_i}}{\Delta t}
\]

The graph distinguishes between:

- Observed interval count rates.
- The model-predicted count-rate trend.
- Approximate Poisson uncertainty.

The vertical axis uses an intentionally zoomed scale to make changes in interval count rates easier to inspect. Its limits should be considered when comparing measurements between experiments.

### Radioactive decay during acquisition

Longer acquisition durations are available to demonstrate how radioactive decay affects the measured count rate.

The expected count-rate behaviour depends on the radionuclide half-life and acquisition duration.

For example, a short acquisition using \(^{89}\mathrm{Zr}\) should show very little physical decay, whereas a sufficiently long acquisition using \(^{18}\mathrm{F}\) produces a more noticeable decline.

This distinction is important when interpreting the interval-CPS graph.

### Dead-time modelling

The expected interval-count equation now explicitly incorporates the acquisition-average dead-time response factor used by the simulator.

For the non-paralysable model:

\[
R_{\mathrm{observed}} =
\frac{R_{\mathrm{true}}}
{1 + R_{\mathrm{true}}\tau}
\]

The simulator applies a simplified acquisition-average dead-time correction. It should not be interpreted as a full time-dependent electronics model.

### Mathematical notation and equation consistency

The measurement equations have been reviewed to improve mathematical formatting and consistency.

Examples include:

\[
R_{\mathrm{net}}
\]

\[
R_{\mathrm{gross}}
\]

\[
S_{\mathrm{cal}}
\]

\[
A_{\mathrm{tissue}}
\]

The revised material also clarifies net-rate uncertainty, calibration regression, geometry uncertainty propagation and PET/gamma agreement.

Duplicated interval-rate and uncertainty equations have been removed from the teaching material.

---

## Installation and Setup

### Requirements

- Python 3.10 or later.
- pip package manager.
- macOS, Windows or Linux.
- A modern web browser.

Python dependencies are listed in `requirements.txt`.

### Installation on macOS

Clone the repository or download and extract the project.

Open Terminal in the project directory and run:

```bash
chmod +x run_mac.command
./run_mac.command
```

The launcher:

1. Creates a local Python virtual environment if required.
2. Installs the dependencies listed in `requirements.txt`.
3. Starts the Streamlit application.

The application should then be accessible through the local address displayed in the Terminal.

### Manual installation

Alternatively, create the environment manually:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

### Updating an existing installation

Stop the currently running Streamlit application using `Ctrl+C`.

Replace the existing application source file with the updated `app.py`, then restart:

```bash
cd ~/Documents/GC_App
streamlit run app.py
```

A dependency reinstall is not required for a code-only update, provided the existing environment satisfies `requirements.txt`.

---

## Interpreting the Live Acquisition Plots

### Cumulative counts

Cumulative counts represent the total number of recorded events since the beginning of acquisition.

They should increase or remain constant as acquisition progresses.

The displayed x- and y-axis ranges expand dynamically with elapsed acquisition time and accumulated counts.

### Interval count rate

Each interval-CPS point is calculated independently using the number of events recorded during that interval divided by its duration.

The measured values fluctuate due to counting statistics.

The expected count-rate trend is displayed separately from the observed measurements.

### Poisson uncertainty

For independent radioactive counting events, the approximate standard uncertainty in the number of recorded counts is:

\[
\sigma_N = \sqrt{N}
\]

The corresponding interval count-rate uncertainty is:

\[
\sigma_R = \frac{\sqrt{N}}{\Delta t}
\]

These expressions assume Poisson counting statistics and do not, by themselves, include systematic uncertainties from calibration, geometry, detector response or background-rate estimation.

### Radionuclide dependence

The expected interval count rate changes according to the radionuclide decay constant:

\[
\lambda = \frac{\ln 2}{T_{1/2}}
\]

For a long-lived radionuclide such as \(^{89}\mathrm{Zr}\), decay may be negligible over a short counting interval.

For shorter-lived radionuclides such as \(^{18}\mathrm{F}\), decay becomes more apparent when the acquisition duration is a meaningful fraction of the physical half-life.

---

## Project Structure

A typical project installation contains:

```text
Quantitative-Nuclear-Medicine-Laboratory/
│
├── app.py
├── requirements.txt
├── README.md
├── run_mac.command
│
└── .streamlit/
    └── config.toml
```

The main application is implemented in Python using Streamlit, with interactive data visualisations and plots.

---

## Scientific Model and Limitations

Quantitative Nuclear Medicine Laboratory is an educational simulation platform.

Its purpose is to demonstrate measurement principles, counting statistics, detector response, calibration, normalisation, gamma spectroscopy and quantitative imaging workflows.

The simulated data and instrument responses are not intended to replace experimental measurements or instrument-specific calibration.

In particular:

- Isotope-informed spectra use simplified spectral-response models.
- Detector efficiencies and response functions are illustrative.
- Dead-time is represented using a simplified acquisition-average non-paralysable model.
- Geometry and matrix corrections are teaching approximations.
- Coincidence summing, Compton contributions and other spectral features are illustrative rather than validated Monte Carlo detector simulations.
- PET/SPECT recovery and cross-calibration examples use hypothetical or user-supplied measurements.

For experimental applications, use traceable radioactive reference standards, validated calibration procedures and instrument-specific quality-control measurements.

The application should not be used as a substitute for approved radiation-safety procedures, manufacturer operating instructions or validated quantitative measurement protocols.

**Data and privacy:** The application is designed for synthetic teaching data and user-controlled experimental datasets. Do not upload identifiable patient information or other confidential clinical data to a publicly hosted deployment.

---

## Educational Purpose

The laboratory is intended to support teaching and practical training in:

- Nuclear medicine physics.
- Radiopharmaceutical measurements.
- Gamma-counter operation and quality control.
- Quantitative biodistribution studies.
- Gamma spectroscopy.
- Preclinical PET/SPECT imaging.
- Cross-modality calibration and quantitative validation.

The central learning objective is to connect the physical principles of radiation detection with practical measurements, calibration procedures and quantitative interpretation.

Students are encouraged to investigate experimental variables, identify measurement limitations, troubleshoot simulated faults and critically evaluate their generated results.# Physics_Lectures
