# Root D2O Inverse Model

Inverse modeling of axial and radial fluxes in roots based on D2O concentration profiles from neutron radiography.

This script reads processed root-wise D2O profiles, fits a simple transport model to each root, and exports fitted parameters, simulated profiles, and overview plots.

## What it does

For each sample and compartment, the script:

- reads processed D2O profiles from `OutPuts.xlsx`
- smooths noisy profiles in time and space
- selects representative observation times
- solves a 1D transport model with:
  - axial transport in xylem
  - exchange between xylem and surrounding root tissue
  - radial uptake in xylem
- fits model parameters by minimizing the mismatch between observed and simulated profiles
- saves fitted parameters and figures

## Fitted parameters

The model fits three parameters for each root:

- `jx_base`: axial flux in xylem
- `a`: radial flux into xylem
- `gamma`: exchange coefficient between xylem and tissue

## Input folder structure

```text
day_D2O/
├─ mean_r.xlsx
├─ s_d_4/
│  ├─ Root_Length_Info.xlsx
│  ├─ D2O_compartment4/
│  │  └─ FinalResults/
│  │     └─ OutPuts.xlsx
│  ├─ D2O_compartment3/
│  │  └─ FinalResults/
│  │     └─ OutPuts.xlsx
│  └─ D2O_compartment2/
│     └─ FinalResults/
│        └─ OutPuts.xlsx
```

A diffusion-fit file is also expected, e.g.:
```text
night_D2O/
└─ fitting_parameters.xlsx
```
## Outputs

The script writes:

- fitted_params_RadialFlux.xlsx in the sample folder
- OutPuts_Fittings.xlsx in each compartment folder
- <compartment>_Overview.png overview plots

## Numerical setup

We used following solving system:
- solve_ivp(..., method="BDF")
- atol = 1e-9
- rtol = 1e-8
- max_step = 1

Optimization is performed in two stages:

1. Differential evolution
2. TNC local refinement

## Notes
- Missing compartment folders are skipped automatically.
- The script assumes the same column structure as the Excel files produced by the image-processing workflow.
