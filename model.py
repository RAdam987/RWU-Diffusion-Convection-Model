from __future__ import annotations

import argparse
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline, interp1d
from scipy.ndimage import median_filter
from scipy.optimize import differential_evolution, minimize
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.close("all")

LOGGER = logging.getLogger(__name__)
BEST_OBJ = np.inf


def remove_noise(
    df: pd.DataFrame,
    fig_name: str = "D2O_Cleaning",
    time_window: int = 2,
    spatial_median_window: int = 1,
    smooth_factor: float = 1,
    n_iter: int = 2,
) -> pd.DataFrame:
    """
    Smooth observed D2O profiles in time and space.

    This function:
    1. sorts the profile by distance,
    2. fills NaNs temporarily by interpolation,
    3. smooths along time using isotonic regression,
    4. smooths along space using isotonic regression + spline,
    5. restores the original NaN positions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing one root profile with a distance column
        ('DistanceFromDistalPart') and multiple time columns
        ('CD2O_Root_t...').
    fig_name : str, optional
        Name of the figure, currently only kept for compatibility and
        potential plotting/debugging.
    time_window : int, optional
        Window size for temporal median filtering before isotonic regression.
    spatial_median_window : int, optional
        Window size for spatial median filtering before isotonic regression.
    smooth_factor : float, optional
        Scaling factor for spline smoothing in space.
    n_iter : int, optional
        Number of alternating time/space smoothing iterations.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with the same structure as the input.
    """
    if df["DistanceFromDistalPart"].isnull().any():
        LOGGER.warning("Dropping rows with NaN in 'DistanceFromDistalPart'.")
        df = df.dropna(subset=["DistanceFromDistalPart"]).reset_index(drop=True)

    df_original = df.copy()
    df = df.sort_values("DistanceFromDistalPart").reset_index(drop=True)

    distance = df["DistanceFromDistalPart"].values
    time_cols = [c for c in df.columns if c.startswith("CD2O_Root_t")]

    data = df[time_cols].values.astype(float)
    nan_mask = np.isnan(data)

    def fill_nans(matrix: np.ndarray) -> np.ndarray:
        """
        Fill missing values temporarily by linear interpolation.

        First interpolation is done row-wise (time direction),
        then column-wise (space direction).
        
        Parameters
        ----------
        matrix : np.ndarray
            2D array of concentration values with possible NaNs.

        Returns
        -------
        np.ndarray
            Array where NaNs are temporarily filled for smoothing.
        """
        mat = matrix.copy()

        for i in range(mat.shape[0]):
            s = pd.Series(mat[i, :])
            s.interpolate(method="linear", limit_direction="both", inplace=True)
            mat[i, :] = s.values

        for j in range(mat.shape[1]):
            s = pd.Series(mat[:, j])
            s.interpolate(method="linear", limit_direction="both", inplace=True)
            mat[:, j] = s.values

        return mat

    working = fill_nans(data)

    def smooth_time(matrix: np.ndarray) -> np.ndarray:
        """
        Smooth each spatial position across time.

        Median filtering removes spikes, then isotonic regression enforces
        monotonic increase in time.

        Parameters
        ----------
        matrix : np.ndarray
            2D array with space along rows and time along columns.

        Returns
        -------
        np.ndarray
            Time-smoothed array.
        """
        smoothed = np.zeros_like(matrix)

        for i in range(matrix.shape[0]):
            y = matrix[i, :]
            y_med = median_filter(y, size=time_window, mode="reflect")
            iso = IsotonicRegression(increasing=True)
            y_iso = iso.fit_transform(np.arange(len(y_med)), y_med)
            smoothed[i, :] = y_iso

        return smoothed

    def smooth_space(matrix: np.ndarray) -> np.ndarray:
        """
        Smooth each time profile across space.

        Median filtering removes local spikes,
        isotonic regression enforces monotone non-increasing shape,
        and a spline provides a smoother final profile.

        Parameters
        ----------
        matrix : np.ndarray
            2D array with space along rows and time along columns.

        Returns
        -------
        np.ndarray
            Space-smoothed array.
        """
        smoothed = np.zeros_like(matrix)

        for j in range(matrix.shape[1]):
            y = matrix[:, j]
            y_med = median_filter(y, size=spatial_median_window, mode="reflect")
            iso = IsotonicRegression(increasing=False)
            y_iso = iso.fit_transform(distance, y_med)
            s_val = smooth_factor * len(distance) * np.var(y_iso)
            spline = UnivariateSpline(distance, y_iso, s=s_val)
            y_final = spline(distance)
            smoothed[:, j] = y_final

        return smoothed

    cleaned = working.copy()

    for _ in range(n_iter):
        cleaned = smooth_time(cleaned)
        cleaned = smooth_space(cleaned)

    cleaned[nan_mask] = np.nan
    cleaned[cleaned < 0] = 0

    for i, col in enumerate(time_cols):
        df[col] = cleaned[:, i]

    df_original[time_cols] = df[time_cols]
    df_original["DistanceFromDistalPart"] = df["DistanceFromDistalPart"]

    return df_original


def read_obs(
    sample: str,
    base_dir: Path,
    compartments: list[str],
    nr_obs_space_sel: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Read processed observation data for one sample.

    This function:
    - opens the Excel outputs created by image processing,
    - smooths the raw root profiles,
    - defines time zero based on D2O arrival,
    - selects spatial and temporal subsets,
    - stores everything in a nested dictionary.

    Parameters
    ----------
    sample : str
        Sample name, for example 's_d_4'.
    base_dir : Path
        Base directory containing sample folders and shared input files
        such as 'mean_r.xlsx'.
    compartments : list[str]
        List of compartments to process, e.g.
        ['compartment4', 'compartment3', 'compartment2'].
    nr_obs_space_sel : int
        Number of spatial positions to retain for fitting after subsampling.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Nested dictionary:
        obs[compartment][root_id] = root-specific observation data.
    """
    obs: dict[str, dict[str, dict[str, Any]]] = {}

    foldername = base_dir / sample
    avg_root_radius = pd.read_excel(base_dir / "mean_r.xlsx", index_col="treatment")
    root_length_info = pd.read_excel(base_dir / sample / "Root_Length_Info.xlsx")

    foldername.mkdir(parents=True, exist_ok=True)
    excel_file_path = foldername / "fitted_params_RadialFlux.xlsx"

    if not excel_file_path.exists():
        pd.DataFrame().to_excel(excel_file_path, engine="openpyxl", index=False)

    for comp_nr in compartments:
        com_path = foldername / f"D2O_{comp_nr}"
        outputs_path = com_path / "FinalResults" / "OutPuts.xlsx"

        if not com_path.is_dir():
            continue

        if not outputs_path.exists():
            continue

        file = pd.ExcelFile(outputs_path)
        sheets = file.sheet_names

        obs[comp_nr] = {}

        for root_nr in sheets:
            df = pd.read_excel(outputs_path, sheet_name=root_nr)

            root_r = df["Root Diameter [cm]"][0] / 2
            root_type = df.loc[0, "Root Type"]

            if "CR" in root_type:
                rt = "mean_r_cr[cm]"
            elif "SR" in root_type:
                rt = "mean_r_sr[cm]"
            else:
                rt = None

            if rt is not None:
                _mean_radius_treat = avg_root_radius.loc[sample[0:3], rt]

            fig_name = comp_nr + "_" + root_nr
            df = remove_noise(df, fig_name)

            t_obs = df["tObs [min]"].sort_values().dropna(how="all").values
            t_obs = t_obs - t_obs[0]

            c_d2o_obs_tip = np.nanmedian(df[df.columns[5:]].values[:20, :], axis=0)
            distance = df["DistanceFromDistalPart"].values

            c_d2o_obs = df[df.columns[5:]].values
            c_d2o_obs = c_d2o_obs[10:-10, :]
            distance = distance[10:-10]

            t0_select = np.argmax(
                np.nanmean(c_d2o_obs[0:int(c_d2o_obs.shape[1] / 4), :], axis=0) > 0.015
            ) - 1

            if t0_select == -1 or c_d2o_obs.shape[1] - t0_select < 7:
                continue

            t_obs = t_obs[t0_select:]
            t_obs = t_obs - t_obs[0]

            c_d2o_obs = c_d2o_obs[:, t0_select:]
            c_d2o_obs_tip = c_d2o_obs_tip[t0_select:]

            c_d2o_obs[:, 0] = 0
            c_d2o_obs_tip[0] = 0

            idx_valid = np.isfinite(c_d2o_obs_tip)
            c_d2o_obs_tip_fun = interp1d(
                t_obs[idx_valid],
                c_d2o_obs_tip[idx_valid],
                kind="slinear",
                fill_value="extrapolate",
            )

            c_d2o_obs_tip_x = c_d2o_obs_tip.copy()
            c_d2o_obs_tip_x[0:] = 0.9
            c_d2o_obs_tip_xylem_fun = interp1d(
                t_obs[idx_valid],
                c_d2o_obs_tip_x[idx_valid],
                kind="slinear",
                fill_value="extrapolate",
            )

            idx_space_select = np.arange(
                0,
                c_d2o_obs.shape[0],
                int(c_d2o_obs.shape[0] / (nr_obs_space_sel - 1)),
            ).tolist()

            distance = distance[idx_space_select]
            c_d2o_obs = c_d2o_obs[idx_space_select, :]

            def select_profiles_by_cumulative_change(
                c_d2o_obs_local: np.ndarray,
                t_obs_local: np.ndarray,
                distance_local: np.ndarray,
                fig_name_local: str,
                n_select: int,
            ) -> np.ndarray:
                """
                Select a subset of time profiles that best represents total profile evolution.

                The selection is based on cumulative change between consecutive profiles.

                Parameters
                ----------
                c_d2o_obs_local : np.ndarray
                    2D observed concentration matrix for one root.
                t_obs_local : np.ndarray
                    Observation times corresponding to the columns.
                distance_local : np.ndarray
                    Distances corresponding to the rows.
                fig_name_local : str
                    Figure name placeholder, currently only kept for compatibility.
                n_select : int
                    Number of representative time profiles to select.

                Returns
                -------
                np.ndarray
                    Array of selected time indices.
                """
                data = c_d2o_obs_local
                n_dist, n_time = data.shape

                min_valid_points = max(2, int(0.1 * n_dist))

                first_valid = None
                for t in range(1, n_time):
                    if np.sum(~np.isnan(data[:, t])) > min_valid_points:
                        first_valid = t
                        break

                delta = []
                valid_time_indices = []

                for t in range(first_valid + 1, n_time):
                    y_prev = data[:, t - 1]
                    y_curr = data[:, t]

                    valid = (~np.isnan(y_prev)) & (~np.isnan(y_curr))

                    if np.sum(valid) > min_valid_points:
                        diff = np.abs(y_curr[valid] - y_prev[valid])
                        area = np.trapezoid(diff, distance_local[valid])
                        delta.append(area)
                        valid_time_indices.append(t)
                    else:
                        delta.append(0)
                        valid_time_indices.append(t)

                delta = np.array(delta)
                cum_change = np.cumsum(delta)

                if cum_change.max() > 0:
                    cum_change = cum_change / cum_change.max()

                targets = np.linspace(0, 1, n_select)

                if first_valid > 5:
                    selected_times = [0, first_valid]
                else:
                    selected_times = [0]

                for val in targets:
                    idx = np.argmin(np.abs(cum_change - val))
                    selected_times.append(valid_time_indices[idx])

                selected_times.append(n_time - 1)
                selected_times = np.unique(selected_times)

                return selected_times

            selected_times = select_profiles_by_cumulative_change(
                c_d2o_obs,
                t_obs,
                distance,
                fig_name,
                n_select=10,
            )
            selected_times = selected_times[:-4]

            t_obs = np.array(t_obs[selected_times])
            t_obs = t_obs - t_obs[0]
            c_d2o_obs = c_d2o_obs[:, selected_times]

            c_d2o_obs_t0 = c_d2o_obs[:, 0]
            c_d2o_obs_t0[np.isnan(c_d2o_obs_t0)] = 0
            c_d2o_obs[:, 0] = c_d2o_obs_t0

            c_mean = np.nanmean(c_d2o_obs, axis=0)
            if (c_mean[-1] - c_mean[0]) > 0.02:
                obs[comp_nr][root_nr] = {}
                obs[comp_nr][root_nr]["RootID"] = root_type
                obs[comp_nr][root_nr]["root_r"] = root_r
                obs[comp_nr][root_nr]["t_Obs"] = t_obs
                obs[comp_nr][root_nr]["distance"] = distance
                obs[comp_nr][root_nr]["cD2O_Obs"] = c_d2o_obs
                obs[comp_nr][root_nr]["cD2O_Obs_tip_fun"] = c_d2o_obs_tip_fun
                obs[comp_nr][root_nr]["cD2O_Obs_tip_xylem_fun"] = c_d2o_obs_tip_xylem_fun

                filtered_values = root_length_info.loc[
                    (root_length_info["compartment"] == comp_nr)
                    & (root_length_info["root"] == root_nr),
                    "DistanceFromRootTip",
                ]
                obs[comp_nr][root_nr]["RootLengthInfo"] = filtered_values.tolist()

    return obs


def computing_obj_func(obs: np.ndarray, fit: np.ndarray) -> float:
    """
    Compute the scalar objective function used for optimization.

    Parameters
    ----------
    obs : np.ndarray
        Observed concentration matrix.
    fit : np.ndarray
        Simulated concentration matrix on the same grid as `obs`.

    Returns
    -------
    float
        Scalar objective value based on variance-normalized SSE.
    """
    obs = np.array(obs)
    fit = np.array(fit)

    mask = np.isfinite(obs)
    _ = mask

    var = np.nanvar(obs)
    sse = np.nansum((fit - obs) ** 2)
    obj_fun = sse / (var + 1e-12)

    return obj_fun


def save_data(base_dir: Path, sample: str, fits: dict, obs: dict) -> None:
    """
    Save observed and fitted profiles to Excel files.

    One output Excel file is written per compartment.
    Each root gets two sheets:
    - observed profiles
    - fitted profiles
    
    Parameters
    ----------
    base_dir : Path
        Base data directory.
    sample : str
        Sample name.
    fits : dict
        Nested dictionary containing simulated profiles.
    obs : dict
        Nested dictionary containing observed profiles.
    """
    for comp_nr in fits.keys():
        foldername = base_dir / sample / f"D2O_{comp_nr}"
        foldername.mkdir(parents=True, exist_ok=True)

        excel_file_path = foldername / "OutPuts_Fittings.xlsx"

        if not excel_file_path.exists():
            with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
                pd.DataFrame().to_excel(writer, sheet_name="Init", index=False)

        for root_nr in fits[comp_nr].keys():
            z = obs[comp_nr][root_nr]["distance"]
            df1 = pd.DataFrame(
                fits[comp_nr][root_nr]["cD2O_sim"],
                index=z,
                columns=obs[comp_nr][root_nr]["t_Obs"],
            )
            df2 = pd.DataFrame(
                obs[comp_nr][root_nr]["cD2O_Obs"],
                index=z,
                columns=obs[comp_nr][root_nr]["t_Obs"],
            )

            with pd.ExcelWriter(excel_file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df1.to_excel(writer, sheet_name=f"{root_nr}_Obs", header=True, startrow=0)
                df2.to_excel(writer, sheet_name=f"{root_nr}_Fit", header=True, startrow=0)


def create_diffusion_profile(z: np.ndarray, diffusion_fit_file: Path, sample: str) -> np.ndarray:
    """
    Create a linear diffusion profile from treatment-specific fit parameters.
    
    Parameters
    ----------
    z : np.ndarray
        Axial coordinate array.
    diffusion_fit_file : Path
        Path to the Excel file containing fitted diffusion coefficients.
    sample : str
        Sample name; the treatment is inferred from `sample[:3]`.

    Returns
    -------
    np.ndarray
        Diffusion profile evaluated at `z`.
    """
    def linear_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b

    fitted_params = pd.read_excel(diffusion_fit_file)
    a = fitted_params.loc[(fitted_params["Treatment"] == sample[:3]), "a_fit"].iloc[0]
    b = fitted_params.loc[(fitted_params["Treatment"] == sample[:3]), "b_fit"].iloc[0]
    d_profile = linear_model(z, a, b)
    return d_profile


def interp_new(
    c_d2o_sim: np.ndarray,
    z_sim: np.ndarray,
    t_sim: np.ndarray,
    new_z: np.ndarray,
    new_t: np.ndarray,
) -> np.ndarray:
    """
    Interpolate a simulated concentration field onto a new grid.

    Parameters
    ----------
    c_d2o_sim : np.ndarray
        Simulated concentration field defined on (`z_sim`, `t_sim`).
    z_sim : np.ndarray
        Original spatial grid.
    t_sim : np.ndarray
        Original time grid.
    new_z : np.ndarray
        Target spatial grid.
    new_t : np.ndarray
        Target time grid.

    Returns
    -------
    np.ndarray
        Interpolated concentration field on (`new_z`, `new_t`).
    """
    interp_func = RegularGridInterpolator(
        (z_sim, t_sim),
        c_d2o_sim,
        bounds_error=False,
        fill_value=np.nan,
    )
    z_new, t_new = np.meshgrid(new_z, new_t, indexing="ij")
    points = np.stack([z_new.ravel(), t_new.ravel()], axis=-1)
    c_d2o_sim_interp = interp_func(points)
    c_d2o_sim_interp = c_d2o_sim_interp.reshape(z_new.shape)
    return c_d2o_sim_interp


def create_axial_profile(z: np.ndarray, x: dict[str, float]) -> np.ndarray:
    """
    Build the axial velocity profile along the root.

    Velocity is modeled as:
    jx_base + a * z

    Parameters
    ----------
    z : np.ndarray
        Axial coordinate array.
    x : dict[str, float]
        Parameter dictionary containing 'jx_base' and 'a'.

    Returns
    -------
    np.ndarray
        Axial velocity profile evaluated at `z`.
    """
    return x["jx_base"] + x["a"] * z


def precompute_z_map(z: np.ndarray, new_z: np.ndarray):
    """
    Precompute indices and interpolation weights for fast 1D interpolation in z.

    Parameters
    ----------
    z : np.ndarray
        Original regular spatial grid.
    new_z : np.ndarray
        Target spatial coordinates.

    Returns
    -------
    tuple
        (i0, i1, w) where i0 and i1 are neighboring indices in `z`
        and w is the interpolation weight.
    """
    dz = z[1] - z[0]
    s = (new_z - z[0]) / dz
    i0 = np.floor(s).astype(int)
    i0 = np.clip(i0, 0, len(z) - 2)
    w = s - i0
    i1 = i0 + 1
    return i0, i1, w


def calc_obj_fct(
    x: np.ndarray,
    comp_nr: str,
    root_nr: str,
    obs: dict,
    fits: dict,
    obj_func_scalar: Any,
    dz0: float,
    cf_xylem: float,
    solver_max_step: float,
) -> float:
    """
    Solve the forward model for one root and compute the objective function.

    This function:
    - builds geometry,
    - defines transport coefficients,
    - solves the PDE system in semi-discrete form,
    - interpolates the solution back to observed positions,
    - stores the simulated profiles,
    - returns the scalar misfit.

    Parameters
    ----------
    x : np.ndarray
        Parameter vector in the order [jx_base, a, gamma].
    comp_nr : str
        Compartment identifier.
    root_nr : str
        Root identifier within the compartment.
    obs : dict
        Nested observation dictionary.
    fits : dict
        Nested dictionary where simulated profiles are stored.
    obj_func_scalar : Any
        Compatibility flag preserved from the original script.
    dz0 : float
        Axial discretization size in cm.
    cf_xylem : float
        Cross-sectional fraction of the root that is assigned to xylem.
    solver_max_step : float
        Maximum internal time step for `solve_ivp`.

    Returns
    -------
    float
        Scalar objective value for the current parameter set.
    """
    global BEST_OBJ

    x = {
        "jx_base": x[0],
        "a": x[1],
        "gamma": x[2],
    }

    r_root = obs[comp_nr][root_nr]["root_r"]
    r_x = np.sqrt(cf_xylem) * r_root

    a_t = np.pi * r_root**2
    a_x = np.pi * r_x**2
    a_tissue = a_t - a_x

    z_offset = 1
    distance = obs[comp_nr][root_nr]["distance"]
    nz = min(int(np.max(distance) / dz0), 50)
    z = np.linspace(0, np.max(distance) + z_offset, nz)
    dz = z[1] - z[0]
    seg_posi = obs[comp_nr][root_nr]["RootLengthInfo"][0]
    _ = seg_posi

    c_d2o_obs = obs[comp_nr][root_nr]["cD2O_Obs"]
    cin_func = obs[comp_nr][root_nr]["cD2O_Obs_tip_fun"]
    cin_func_xylme = obs[comp_nr][root_nr]["cD2O_Obs_tip_xylem_fun"]

    t_obs = obs[comp_nr][root_nr]["t_Obs"]
    t_final = t_obs[-1]

    idx_valid = np.isfinite(c_d2o_obs[:, 0])
    c0_fun = interp1d(distance[idx_valid], c_d2o_obs[idx_valid, 0], fill_value="extrapolate")
    _ = c0_fun

    ct0 = np.zeros(z.shape)
    cx0 = np.zeros(z.shape)
    y0 = np.concatenate([cx0, ct0])

    # diffusion 
    d0 = 1.2 * 10**-3
    d_nodes = (d0 / 6) * np.ones_like(z)

    d_face = np.zeros(nz + 1)
    d_face[1:-1] = 2 * d_nodes[:-1] * d_nodes[1:] / (d_nodes[:-1] + d_nodes[1:] + 1e-20)
    d_face[0] = d_nodes[0]
    d_face[-1] = d_nodes[-1]

    u_x = create_axial_profile(z, x)

    u_face = np.zeros(nz + 1)
    u_face[1:-1] = 0.5 * (u_x[:-1] + u_x[1:])
    u_face[0] = u_x[0]
    u_face[-1] = u_x[-1]

    qx = u_x * a_x
    dqxdz = np.zeros(qx.shape)
    dqxdz[1:-1] = (qx[2:] - qx[:-2]) / (2 * dz)
    dqxdz[0] = (qx[1] - qx[0]) / dz
    dqxdz[-1] = (qx[-1] - qx[-2]) / dz

    jr_surface = dqxdz / (2 * np.pi * r_root)
    qx_in = qx[0]

    cx_ext = np.zeros(nz + 2)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the semi-discrete transport system.

        The system includes:
        - axial diffusion in xylem,
        - axial advection in xylem,
        - exchange between xylem and tissue.
        
        Parameters
        ----------
        t : float
            Current simulation time.
        y : np.ndarray
            Current state vector containing xylem and tissue concentrations.

        Returns
        -------
        np.ndarray
            Time derivative of the state vector.
        """
        cx = y[:nz]
        ct = y[nz:]

        cavg_in = cin_func(t)

        cx_in = cin_func_xylme(t)
        ct_in = (a_t * cavg_in - a_x * cx_in) / a_tissue

        if ct_in < 0:
            ct_in = 0
            cx_in = a_t * cavg_in / a_x

        cx_ext[1:-1] = cx
        cx_ext[0] = cx_in
        cx_ext[-1] = 2.0 * cx[-1] - cx[-2]

        grad_cx = (cx_ext[1:] - cx_ext[:-1]) / dz
        f_diff_x = d_face * grad_cx

        c_up = np.where(u_face > 0, cx_ext[:-1], cx_ext[1:])
        f_adv_x = u_face * c_up

        f_total_x = f_diff_x - f_adv_x

        d_cxdt = (f_total_x[1:] - f_total_x[:-1]) / dz + x["gamma"] * (ct - cx)
        d_ctdt = -x["gamma"] * (ct - cx)

        return np.concatenate([d_cxdt, d_ctdt])

    # Keep solve_ivp exactly as in the original script
    sol = solve_ivp(
        rhs,
        [0, t_final],
        y0,
        method="BDF",
        t_eval=t_obs,
        atol=1e-9,
        rtol=1e-8,
        max_step=solver_max_step,
    )

    if not sol.success:
        return 1e5

    cx_sim = sol.y[:nz, :]
    ct_sim = sol.y[nz:, :]

    c_d2o_sim = (a_x * cx_sim + a_tissue * ct_sim) / a_t
    t_sim = sol.t
    _ = t_sim

    new_z = distance + z_offset
    i0, i1, w = precompute_z_map(z, new_z)
    c_d2o_sim = (1 - w)[:, None] * c_d2o_sim[i0, :] + w[:, None] * c_d2o_sim[i1, :]

    fits[comp_nr][root_nr] = {
        "z": z[z > z_offset],
        "cD2O_sim": c_d2o_sim,
        "t_sim": sol.t,
        "jr_surface": jr_surface[z > z_offset],
        "Qx_in": qx_in,
    }

    obj_func = computing_obj_func(c_d2o_obs, c_d2o_sim)
    obj_fct = np.mean(obj_func)

    if obj_fct < BEST_OBJ:
        BEST_OBJ = obj_fct
        LOGGER.info("New best objective: %.6e", BEST_OBJ)

    if obj_func_scalar == [True]:
        return obj_fct
    else:
        return obj_fct


def plot_results(obs: dict, fits: dict, sample: str, outpath: Path | None = None, ncols: int = 4) -> None:
    """
    Plot observed and simulated profiles for all roots in all compartments.

    Parameters
    ----------
    obs : dict
        Nested observation dictionary.
    fits : dict
        Nested dictionary containing simulated profiles.
    sample : str
        Sample name, currently only kept for compatibility.
    outpath : Path | None, optional
        Output path for saving the figure. If None, the figure is not saved.
    ncols : int, optional
        Number of subplot columns.
    """
    n_plots = sum(len(fits[comp_nr]) for comp_nr in fits.keys())
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()

    count = 0
    last_comp = None

    for comp_nr in fits.keys():
        last_comp = comp_nr
        for root_nr in fits[comp_nr].keys():
            ax = axes[count]
            count += 1

            z = obs[comp_nr][root_nr]["distance"]
            c_d2o_obs = obs[comp_nr][root_nr]["cD2O_Obs"]
            c_d2o_sim = fits[comp_nr][root_nr]["cD2O_sim"]

            ax.set_title(f"{comp_nr}, {root_nr}")

            n = min(c_d2o_obs.shape[1], c_d2o_sim.shape[1])
            cmap = plt.cm.get_cmap("tab20", n)

            for k in range(n):
                ax.plot(z, c_d2o_obs[:, k], "o", markersize=5, color=cmap(k))
                ax.plot(z, c_d2o_sim[:, k], linewidth=2, color=cmap(k))

            if c_d2o_obs.shape[1] != c_d2o_sim.shape[1]:
                ax.text(
                    0.02,
                    0.98,
                    f"Obs={c_d2o_obs.shape[1]} Sim={c_d2o_sim.shape[1]}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )

            ax.set_xlabel("z [cm]")
            ax.set_ylabel("conc. of D$_2$O")

    for j in range(count, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    plt.show()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches="tight", dpi=300)


def minimize_inverse_model(
    x: dict[str, float],
    bound: dict[str, tuple[float, float]],
    comp_nr: str,
    root_nr: str,
    obs: dict,
    fits: dict,
    dz0: float,
    cf_xylem: float,
    solver_max_step: float,
) -> dict[str, float]:
    """
    Fit model parameters in two stages:

    1. Differential evolution for global search
    2. TNC refinement for local polishing

    Parameters
    ----------
    x : dict[str, float]
        Initial parameter guess dictionary with keys 'jx_base', 'a', and 'gamma'.
    bound : dict[str, tuple[float, float]]
        Parameter bounds dictionary.
    comp_nr : str
        Compartment identifier.
    root_nr : str
        Root identifier.
    obs : dict
        Nested observation dictionary.
    fits : dict
        Nested dictionary where simulated profiles are stored.
    dz0 : float
        Axial discretization size in cm.
    cf_xylem : float
        Xylem cross-sectional fraction.
    solver_max_step : float
        Maximum internal time step for the ODE solver.

    Returns
    -------
    dict[str, float]
        Optimized parameter dictionary.
    """
    param_names = ["jx_base", "a", "gamma"]
    bounds_list = [bound["jx_base"], bound["a"], bound["gamma"]]

    x_best = np.array([x["jx_base"], x["a"], x["gamma"]])

    LOGGER.info("STAGE 1: GLOBAL SEARCH (Differential Evolution)")
    result = differential_evolution(
        calc_obj_fct,
        x0=x_best,
        bounds=bounds_list,
        args=(comp_nr, root_nr, obs, fits, True, dz0, cf_xylem, solver_max_step),
        strategy="rand1bin",
        init="latinhypercube",
        popsize=10,
        maxiter=300,
        polish=True,
        seed=30,
        updating="deferred",
        tol=1e-4,
        atol=1e-4,
        disp=False,
    )

    LOGGER.info("Stage 1 complete: objective=%.6e", result.fun)
    for i, param_name in enumerate(param_names):
        LOGGER.info("%s = %.6e", param_name, result.x[i])

    LOGGER.info("STAGE 2: FINAL POLISH (TNC Refinement)")
    obj_func_scalar = False
    _ = obj_func_scalar

    result = minimize(
        calc_obj_fct,
        x0=result.x,
        args=(comp_nr, root_nr, obs, fits, True, dz0, cf_xylem, solver_max_step),
        bounds=bounds_list,
        method="TNC",
        options={
            "maxfun": 5000,
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-10,
            "eps": 1e-9,
        },
    )

    x_best = result.x
    x_final = {}
    for i, param_name in enumerate(param_names):
        x_final[param_name] = x_best[i]

    return x_final


def run_sample(
    sample: str,
    base_dir: Path,
    diffusion_fit_file: Path,
    compartments: list[str],
    dz0: float,
    solver_max_step: float,
    nr_obs_space_sel: int,
    cf_xylem: float,
) -> None:
    """
    Run the full workflow for one sample.

    This function:
    - reads observations,
    - fits each root in each compartment,
    - stores fitted parameters,
    - writes plots and Excel outputs.

    Parameters
    ----------
    sample : str
        Sample name.
    base_dir : Path
        Base directory containing all input and output folders.
    diffusion_fit_file : Path
        Path to the diffusion-fit file; currently passed through for compatibility.
    compartments : list[str]
        Compartments to process.
    dz0 : float
        Axial discretization size in cm.
    solver_max_step : float
        Maximum time step used in the ODE solver.
    nr_obs_space_sel : int
        Number of spatial observation points retained for fitting.
    cf_xylem : float
        Xylem cross-sectional fraction.
    """
    global BEST_OBJ

    _ = diffusion_fit_file  
    fits: dict[str, dict[str, dict[str, Any]]] = {}
    obs = read_obs(
        sample=sample,
        base_dir=base_dir,
        compartments=compartments,
        nr_obs_space_sel=nr_obs_space_sel,
    )

    x = {
        "jx_base": 0.05,
        "a": 0.06,
        "gamma": 7e-3,
    }

    for comp_nr in obs.keys():
        fits[comp_nr] = {}
        x_all = pd.DataFrame()
        jr_prof = pd.DataFrame()
        data_root: dict[str, dict[str, Any]] = {}

        for root_nr in obs[comp_nr].keys():
            LOGGER.info("Sample=%s | %s | Root=%s", sample, comp_nr, root_nr)

            bound = {}
            bound["jx_base"] = (1e-5, 150)
            bound["a"] = (0, 2)
            bound["gamma"] = (1e-5, 5e-1)

            root_count = obs[comp_nr][root_nr]["RootID"]
            _ = root_count

            BEST_OBJ = np.inf
            x = minimize_inverse_model(
                x=x,
                bound=bound,
                comp_nr=comp_nr,
                root_nr=root_nr,
                obs=obs,
                fits=fits,
                dz0=dz0,
                cf_xylem=cf_xylem,
                solver_max_step=solver_max_step,
            )

            obj_func_scalar = True
            x_array = np.array([x["jx_base"], x["a"], x["gamma"]])
            obj_fct = calc_obj_fct(
                x_array,
                comp_nr,
                root_nr,
                obs,
                fits,
                obj_func_scalar,
                dz0,
                cf_xylem,
                solver_max_step,
            )
            _ = obj_fct

            z_sim = fits[comp_nr][root_nr]["z"]
            z_sim = z_sim - z_sim[0]
            posi = obs[comp_nr][root_nr]["RootLengthInfo"] + z_sim

            jr = fits[comp_nr][root_nr]["jr_surface"]
            qx_in = fits[comp_nr][root_nr]["Qx_in"]

            x_store = {}
            x_store["RootType"] = obs[comp_nr][root_nr]["RootID"][0]
            x_store["Dist. From Tip till barrier [cm]"] = obs[comp_nr][root_nr]["RootLengthInfo"][0]
            x_store["jr_z=0 [cm/min]"] = jr[1]
            x_store["jx_base [cm/min]"] = x["jx_base"]
            x_store["jx/jr [-]"] = x["jx_base"] / jr[1]
            x_store["gamma"] = x["gamma"]
            x_store["Qx_in [cm^3/min]"] = qx_in
            x_store["Dist. From Tip - middel [cm]"] = np.mean(posi)
            x_store["jr_middel [cm/min]"] = jr[int(len(jr) / 2)]
            x_store["R_root[cm]"] = obs[comp_nr][root_nr]["root_r"]

            data_root[comp_nr] = {}
            data_root[comp_nr][root_count] = {}
            data_root[comp_nr][root_count]["jx_base"] = x["jx_base"]

            result = pd.DataFrame.from_dict(x_store, orient="index", columns=[f"{root_nr}"])
            x_all = pd.concat([x_all, result], axis=1)

            dist_df = pd.DataFrame(posi, columns=[f"Dist_{root_nr}"])
            jr_df = pd.DataFrame(jr, columns=[f"jr_{root_nr}"])
            jr_prof = pd.concat([jr_prof, dist_df, jr_df], axis=1)

        full_col = pd.concat([x_all, jr_prof], axis=1)
        out_param_file = base_dir / sample / "fitted_params_RadialFlux.xlsx"
        with pd.ExcelWriter(out_param_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            full_col.to_excel(writer, sheet_name="FittingParam_" + comp_nr, header=True)

        fig_path = base_dir / sample / f"{comp_nr}_Overview.png"
        plot_results(obs, fits, sample=sample, outpath=fig_path)

        save_data(base_dir, sample, fits, obs)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running the script.
    """
    parser = argparse.ArgumentParser(
        description="Numerical Model to Solve Diffusion Convection Equation in 1D."
    )
    parser.add_argument("--base-dir", type=Path, required=True, help="Base directory containing sample folders.")
    parser.add_argument(
        "--diffusion-fit-file",
        type=Path,
        required=True,
        help="Path to fitting_parameters.xlsx.",
    )
    parser.add_argument("--samples", nargs="+", required=True, help="Sample names, e.g. s_d_4")
    parser.add_argument(
        "--compartments",
        nargs="+",
        default=["compartment4", "compartment3", "compartment2"],
        help="Compartments to process.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point of the script.

    Reads command-line arguments, sets constants, and runs the full workflow
    for all requested samples.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    dz0 = 0.02
    solver_max_step = 1
    nr_obs_space_sel = 15
    cf_xylem = 0.04

    for sample in args.samples:
        run_sample(
            sample=sample,
            base_dir=args.base_dir,
            diffusion_fit_file=args.diffusion_fit_file,
            compartments=args.compartments,
            dz0=dz0,
            solver_max_step=solver_max_step,
            nr_obs_space_sel=nr_obs_space_sel,
            cf_xylem=cf_xylem,
        )

if __name__ == "__main__":
    main()
