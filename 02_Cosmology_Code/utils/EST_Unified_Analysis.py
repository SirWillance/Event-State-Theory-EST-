"""
EST_Unified_Analysis.py
-----------------------

Unified analysis tool for Event-State Theory cosmology simulations.

- GUI-based (Tkinter folder picker).
- Reads EST simulation outputs (final_universe_state.npy, timeseries .npy).
- Computes matter power spectrum using the SAME logic as compute_pk.py
  (reproduces n_s ≈ -3.03 for the 64^3 test universe from Paper IV).
- Additionally computes:
    * Dimensionless power spectrum Δ^2(k)
    * Two-point correlation function ξ(r)
    * Simple halo catalog via connected components on a downsampled grid
    * 3D export data for external visualization

Creates:
    * PowerSpectrum/   : P(k), Δ^2(k), plots, metadata
    * Correlation/     : ξ(r) data + plot
    * HaloCatalog/     : halo_catalog.csv + halo_labels_downsampled.npy
    * Visualizations/  : projections, slices, histogram (+ optional GIF)
    * TimeSeries/      : plots, GIF, combined timeseries_data.csv
    * 3D/              : occupied_voxels.csv, universe_delta.npy

Usage:
    python EST_Unified_Analysis.py

Then select an EST experiment folder (the one containing final_universe_state.npy).
"""

import os
from pathlib import Path
import json
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import tkinter as tk
from tkinter import filedialog, messagebox


# ----------------------------------------------------------
# Structure metric (simple filament/void contrast measure)
# ----------------------------------------------------------

def structure_metric_from_grid(universe_grid: np.ndarray) -> float:
    """
    Estimate a simple 'structure metric' from a 3D field:
    - project along one axis
    - compute variance / mean
    - renormalize and cap at 1.0
    """
    if universe_grid is None or universe_grid.size == 0:
        return 0.0
    if universe_grid.sum() == 0:
        return 0.0

    proj = universe_grid.sum(axis=0)
    var = float(np.var(proj))
    mean = float(np.mean(proj))
    if mean == 0.0:
        return 0.0

    metric = var / (mean + 1e-10)
    return float(min(metric / 10.0, 1.0))


# ----------------------------------------------------------
# POWER SPECTRUM CORE (matching compute_pk.py)
# ----------------------------------------------------------

def compute_power_spectrum_est(field: np.ndarray, source_file: str = None):
    """
    Compute power spectrum from 3D field using the same logic
    as your compute_pk.py:

    - δ = ρ / <ρ> - 1
    - 3D FFT of δ
    - |δ_k|^2 power
    - log-spaced k-bins from 10^0.01 to k_Nyquist
    - spectral index n_s fitted over k in [8, 50] (with dynamic fallback)

    This ensures that values like n_s ≈ -3.03 for a 64^3 grid
    are reproduced when using the same input field.
    """
    density = field.astype(float)
    mean_density = density.mean()
    delta = density / mean_density - 1.0

    # 3D FFT and raw power
    f = np.fft.fftn(delta)
    pk3d = np.abs(f) ** 2

    nx, ny, nz = field.shape
    knyq = nx // 2

    # Wave numbers in grid units
    kx = np.fft.fftfreq(nx, d=1) * nx
    ky = np.fft.fftfreq(ny, d=1) * ny
    kz = np.fft.fftfreq(nz, d=1) * nz
    kgrid = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kgrid[0] ** 2 + kgrid[1] ** 2 + kgrid[2] ** 2)

    # Logarithmic radial binning
    k = k.ravel()
    pk = pk3d.ravel()
    bins = np.logspace(0.01, np.log10(knyq), 35)
    k_centers = (bins[:-1] + bins[1:]) / 2

    pk_binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (k >= lo) & (k < hi)
        if mask.any():
            pk_binned.append(pk[mask].mean())
        else:
            pk_binned.append(np.nan)

    pk_binned = np.array(pk_binned)

    # Remove NaNs
    valid_mask = ~np.isnan(pk_binned)
    k_centers = k_centers[valid_mask]
    pk_binned = pk_binned[valid_mask]

    # Fit spectral index
    mask = (k_centers > 8) & (k_centers < 50)
    if mask.sum() < 2:
        # Fallback: dynamic range based on grid size
        low_k = max(2, nx // 64)
        high_k = min(50, nx // 4)
        mask = (k_centers > low_k) & (k_centers < high_k)

    if mask.sum() < 2:
        raise ValueError(
            f"Not enough k-bins in fit range for spectral index (found {mask.sum()})"
        )

    logk = np.log(k_centers[mask])
    logpk = np.log(pk_binned[mask])
    coeffs = np.polyfit(logk, logpk, 1)
    n_s = coeffs[0]

    return {
        "k_centers": k_centers,
        "pk_binned": pk_binned,
        "n_s": n_s,
        "bins": bins,
        "field_shape": field.shape,
        "mean_density": mean_density,
        "knyq": knyq,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "source_file": source_file,
        "fit_range": [8, 50],
    }


# ----------------------------------------------------------
# Dimensionless power Δ^2(k)
# ----------------------------------------------------------

def compute_and_save_dimensionless_power(ps_results: dict, ps_dir: Path):
    k = ps_results["k_centers"]
    Pk = ps_results["pk_binned"]

    # Δ^2(k) = k^3 P(k) / (2π^2)
    delta2 = (k ** 3) * Pk / (2.0 * np.pi ** 2)
    np.save(ps_dir / "delta2_k.npy", delta2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(k, delta2, "o-", lw=2)
    ax.set_xlabel("k [grid units]")
    ax.set_ylabel(r"$\Delta^2(k)$")
    ax.set_title("Dimensionless Power Spectrum")
    ax.grid(which="both", alpha=0.3)
    plt.tight_layout()
    out = ps_dir / "dimensionless_power_spectrum.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ----------------------------------------------------------
# Save power-spectrum results (same formats as compute_pk.py)
# ----------------------------------------------------------

def save_power_spectrum_results(results: dict, ps_dir: Path):
    ps_dir.mkdir(exist_ok=True, parents=True)

    k_centers = results["k_centers"]
    pk_binned = results["pk_binned"]
    n_s = results["n_s"]
    field_shape = results["field_shape"]
    mean_density = results["mean_density"]
    source_file = results["source_file"]
    knyq = results["knyq"]
    fit_range = results["fit_range"]

    # NPY
    np.save(ps_dir / "k_centers.npy", k_centers)
    np.save(ps_dir / "pk_binned.npy", pk_binned)

    # NPZ
    np.savez(
        ps_dir / "power_spectrum.npz",
        k_centers=k_centers,
        pk_binned=pk_binned,
        n_s=n_s,
        bins=results["bins"],
        field_shape=field_shape,
        mean_density=mean_density,
    )

    # TXT
    np.savetxt(
        ps_dir / "power_spectrum.txt",
        np.column_stack([k_centers, pk_binned]),
        header=(
            "Power Spectrum Data\n"
            f"Grid: {field_shape}\n"
            f"n_s = {n_s:.4f}\n"
            "k [grid units]\tP(k)"
        ),
        fmt="%.6e",
    )

    # CSV
    np.savetxt(
        ps_dir / "power_spectrum.csv",
        np.column_stack([k_centers, pk_binned]),
        header=f"k,P(k)\n# Grid:{field_shape}\n# n_s={n_s:.4f}",
        delimiter=",",
        fmt="%.6e",
        comments="",
    )

    # JSON metadata
    meta = {
        "analysis": "EST Power Spectrum",
        "grid_size": list(map(int, field_shape)),
        "mean_density": float(mean_density),
        "spectral_index_ns": float(n_s),
        "source_file": str(source_file) if source_file else None,
        "timestamp": datetime.now().isoformat(),
        "fit_range_k_grid": fit_range,
        "k_bins": int(len(k_centers)),
        "nyquist_frequency": int(knyq),
    }
    with (ps_dir / "analysis_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Simple metadata text
    with (ps_dir / "metadata.txt").open("w", encoding="utf-8") as f:
        f.write("Power Spectrum Analysis\n")
        f.write("======================\n")
        f.write(f"Field dimensions: {field_shape}\n")
        f.write(f"Number of k-bins: {len(k_centers)}\n")
        f.write(f"Spectral index n_s: {n_s:.6f}\n")
        f.write(f"Fit range: k = [{fit_range[0]}, {fit_range[1]}] grid units\n")
        f.write(f"Mean density: {mean_density:.6f}\n")
        f.write(f"Nyquist frequency: {knyq}\n")
        if source_file:
            f.write(f"Source file: {source_file}\n")

    return meta


def create_power_spectrum_plot(results: dict, ps_dir: Path):
    k_centers = results["k_centers"]
    pk_binned = results["pk_binned"]
    n_s = results["n_s"]
    nx = results["nx"]

    plt.figure(figsize=(10, 6), facecolor="black")
    plt.rcParams["text.color"] = "white"
    plt.loglog(
        k_centers,
        pk_binned,
        "o-",
        color="#00ffff",
        lw=2.5,
        label=f"EST {nx}³ (n_s = {n_s:.3f})",
    )
    plt.axvline(8, color="gray", alpha=0.4, ls="--", label="Fit range start")
    plt.axvline(50, color="gray", alpha=0.4, ls="--", label="Fit range end")
    plt.xlabel("k  [grid units]", fontsize=14, color="white")
    plt.ylabel("P(k)", fontsize=14, color="white")
    plt.title(f"EST Matter Power Spectrum ({nx}³ Grid)", color="white", fontsize=16)
    plt.grid(alpha=0.3, color="gray")
    plt.legend()
    plt.tight_layout()

    plot_name = f"EST_Power_Spectrum_{nx}.png"
    plot_path = ps_dir / plot_name
    plt.savefig(plot_path, dpi=300, facecolor="black")
    plt.close()
    return plot_path


# ----------------------------------------------------------
# Two-point correlation function ξ(r) via FFT
# ----------------------------------------------------------

def compute_correlation_function(universe: np.ndarray, corr_dir: Path):
    """
    Compute the isotropic two-point correlation function ξ(r) using
    the Wiener-Khinchin theorem:

        ξ = FFT^{-1}(|δ_k|^2) / N

    Then radially average ξ over spherical shells in real space.
    """
    corr_dir.mkdir(exist_ok=True, parents=True)

    density = universe.astype(float)
    mean_density = density.mean()
    delta = density / mean_density - 1.0

    # Auto-correlation via FFT
    f = np.fft.fftn(delta)
    power = np.abs(f) ** 2
    xi_grid = np.fft.ifftn(power).real / delta.size
    xi_grid = np.fft.fftshift(xi_grid)

    nx, ny, nz = universe.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    # Radial distances in grid units
    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    z = np.arange(nz) - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    r_flat = r.ravel()
    xi_flat = xi_grid.ravel()

    r_max = min(nx, ny, nz) / 2.0
    n_bins = min(50, int(r_max))
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    xi_binned = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for rv, xv in zip(r_flat, xi_flat):
        if rv < r_max:
            idx = int(rv / r_max * n_bins)
            if idx >= n_bins:
                idx = n_bins - 1
            xi_binned[idx] += xv
            counts[idx] += 1

    mask = counts > 0
    xi_binned[mask] /= counts[mask]

    # Save data
    np.save(corr_dir / "r_bins.npy", r_centers)
    np.save(corr_dir / "xi_r.npy", xi_binned)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_centers[mask], xi_binned[mask], "-o", lw=2)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("r [grid units]")
    ax.set_ylabel(r"$\xi(r)$")
    ax.set_title("Two-point Correlation Function")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = corr_dir / "correlation_function.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)

    # Simple metadata
    meta = {
        "analysis": "EST Correlation Function",
        "grid_size": list(map(int, universe.shape)),
        "r_max_grid_units": float(r_max),
        "bins": int(n_bins),
        "timestamp": datetime.now().isoformat(),
    }
    with (corr_dir / "correlation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out


# ----------------------------------------------------------
# Halo finder (downsampled, connected components)
# ----------------------------------------------------------

def downsample_max(grid: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample 3D grid by integer factor using max pooling.
    """
    if factor <= 1:
        return grid
    nx, ny, nz = grid.shape
    nx2 = nx // factor
    ny2 = ny // factor
    nz2 = nz // factor

    g = grid[:nx2 * factor, :ny2 * factor, :nz2 * factor]
    g = g.reshape(nx2, factor, ny2, factor, nz2, factor)
    g = g.max(axis=(1, 3, 5))
    return g


def find_halos(universe: np.ndarray, halo_dir: Path,
               threshold: float = 0.5,
               target_size: int = 128):
    """
    Very simple halo finder:

    - Downsample the grid to approx target_size^3 using max-pooling.
    - Threshold > 'threshold' to define occupied voxels.
    - Run a 6-connected component labeling on the downsampled boolean grid.
    - Build a halo catalog with:
        halo_id, voxel_count, approximate_radius_grid_units, center_of_mass_(x,y,z)

    NOTE:
        This is intentionally simple but should be good enough for
        first-generation EST halo statistics.
    """
    halo_dir.mkdir(exist_ok=True, parents=True)

    nx = universe.shape[0]
    factor = max(1, nx // target_size)
    ds_grid = downsample_max(universe.astype(float), factor)
    occ = ds_grid > threshold

    labels = -np.ones_like(ds_grid, dtype=np.int32)
    halo_props = []
    halo_id = 0

    # Precompute neighbor offsets (6-connectivity)
    neighbors = [(1, 0, 0), (-1, 0, 0),
                 (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1)]

    nx2, ny2, nz2 = ds_grid.shape

    # We only iterate over occupied voxels
    occ_indices = np.argwhere(occ)

    visited = np.zeros_like(occ, dtype=bool)

    for idx in occ_indices:
        x, y, z = idx
        if visited[x, y, z]:
            continue

        # BFS / flood-fill
        stack = [(x, y, z)]
        visited[x, y, z] = True
        labels[x, y, z] = halo_id

        voxels = []
        while stack:
            cx, cy, cz = stack.pop()
            voxels.append((cx, cy, cz))
            for dx, dy, dz in neighbors:
                nxp, nyp, nzp = cx + dx, cy + dy, cz + dz
                if 0 <= nxp < nx2 and 0 <= nyp < ny2 and 0 <= nzp < nz2:
                    if occ[nxp, nyp, nzp] and not visited[nxp, nyp, nzp]:
                        visited[nxp, nyp, nzp] = True
                        labels[nxp, nyp, nzp] = halo_id
                        stack.append((nxp, nyp, nzp))

        if voxels:
            vox_arr = np.array(voxels, dtype=float)
            count = len(voxels)

            # Approximate radius in downsampled grid units
            # Assume spherical: V = count * (factor^3)
            # In grid units, radius^3 ~ 3*count / (4π)
            radius_grid = (3.0 * count / (4.0 * np.pi)) ** (1.0 / 3.0)

            # Center of mass (downsampled indices)
            com = vox_arr.mean(axis=0)  # (x, y, z)

            halo_props.append({
                "halo_id": int(halo_id),
                "voxel_count": int(count),
                "radius_grid_units_downsampled": float(radius_grid),
                "center_of_mass_x": float(com[0]),
                "center_of_mass_y": float(com[1]),
                "center_of_mass_z": float(com[2]),
            })

            halo_id += 1

    # Save labels and catalog
    np.save(halo_dir / "halo_labels_downsampled.npy", labels)

    catalog_path = halo_dir / "halo_catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "halo_id",
                "voxel_count",
                "radius_grid_units_downsampled",
                "center_of_mass_x",
                "center_of_mass_y",
                "center_of_mass_z",
            ],
        )
        writer.writeheader()
        for h in halo_props:
            writer.writerow(h)

    meta = {
        "analysis": "EST Halo Catalog (downsampled)",
        "original_grid_size": list(map(int, universe.shape)),
        "downsample_factor": int(factor),
        "downsampled_grid_size": [int(nx2), int(ny2), int(nz2)],
        "threshold": float(threshold),
        "halo_count": int(len(halo_props)),
        "timestamp": datetime.now().isoformat(),
    }
    with (halo_dir / "halo_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return catalog_path, meta


# ----------------------------------------------------------
# Extra visualizations from the grid
# ----------------------------------------------------------

def make_universe_projections(universe: np.ndarray, vis_dir: Path):
    vis_dir.mkdir(exist_ok=True, parents=True)
    nx, ny, nz = universe.shape

    # Sum projections
    proj_xy = universe.sum(axis=2)  # along z
    proj_xz = universe.sum(axis=1)  # along y
    proj_yz = universe.sum(axis=0)  # along x

    def save_proj(img, name, xlabel, ylabel, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img, cmap="inferno", origin="lower", interpolation="nearest")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Integrated density")
        out = vis_dir / name
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    p_xy = save_proj(proj_xy, "projection_xy.png", "x", "y", "Projection Σ_z (x-y)")
    p_xz = save_proj(proj_xz, "projection_xz.png", "x", "z", "Projection Σ_y (x-z)")
    p_yz = save_proj(proj_yz, "projection_yz.png", "y", "z", "Projection Σ_x (y-z)")

    # Central slices
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    slice_xy = universe[:, :, cz]
    slice_xz = universe[:, cy, :]
    slice_yz = universe[cx, :, :]

    def save_slice(img, name, xlabel, ylabel, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(img.T, cmap="inferno", origin="lower", interpolation="nearest")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Density")
        out = vis_dir / name
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close(fig)
        return out

    s_xy = save_slice(slice_xy, "slice_center_xy.png", "x", "y", "Central Slice z = mid")
    s_xz = save_slice(slice_xz, "slice_center_xz.png", "x", "z", "Central Slice y = mid")
    s_yz = save_slice(slice_yz, "slice_center_yz.png", "y", "z", "Central Slice x = mid")

    return [p_xy, p_xz, p_yz, s_xy, s_xz, s_yz]


def make_density_histogram(universe: np.ndarray, vis_dir: Path):
    vis_dir.mkdir(exist_ok=True, parents=True)
    values = universe.ravel()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=np.arange(-0.5, 1.6, 0.5), log=True)
    ax.set_xlabel("Voxel value")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Voxel Value Distribution")
    plt.tight_layout()
    out = vis_dir / "density_histogram.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ----------------------------------------------------------
# Time-series plots + GIFs
# ----------------------------------------------------------

def make_timeseries_plots_and_csv(exp_dir: Path, analysis_dir: Path,
                                  density_ts, phys_ts, z_ts, t_ts):
    if density_ts is None:
        return None, None, None, None

    ts_dir = analysis_dir / "TimeSeries"
    ts_dir.mkdir(exist_ok=True, parents=True)

    frames = np.arange(len(density_ts))

    # Density vs frame
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, density_ts, lw=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Global density (simulation units)")
    ax.set_title("Density Evolution")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    density_plot = ts_dir / "density_timeseries.png"
    plt.savefig(density_plot, dpi=300)
    plt.close(fig)

    # Physical Ω_m vs frame, z, t
    phys_plot = None
    z_plot = None
    t_plot = None

    if phys_ts is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(frames, phys_ts, lw=2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Ω_m")
        ax.set_title("Physical Matter Density Ω_m vs Frame")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        phys_plot = ts_dir / "physical_density_timeseries.png"
        plt.savefig(phys_plot, dpi=300)
        plt.close(fig)

    if (phys_ts is not None) and (z_ts is not None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z_ts, phys_ts, lw=2)
        ax.set_xlabel("Redshift z")
        ax.set_ylabel("Ω_m")
        ax.set_title("Ω_m vs Redshift")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        z_plot = ts_dir / "omega_vs_redshift.png"
        plt.savefig(z_plot, dpi=300)
        plt.close(fig)

    if (phys_ts is not None) and (t_ts is not None):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_ts, phys_ts, lw=2)
        ax.set_xlabel("Time [Gyr]")
        ax.set_ylabel("Ω_m")
        ax.set_title("Ω_m vs Cosmic Time")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        t_plot = ts_dir / "omega_vs_time.png"
        plt.savefig(t_plot, dpi=300)
        plt.close(fig)

    # Combined CSV dataset
    csv_path = ts_dir / "timeseries_data.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame,density,omega,z,time_gyr\n")
        for i in range(len(density_ts)):
            d = density_ts[i]
            o = phys_ts[i] if phys_ts is not None and i < len(phys_ts) else ""
            z = z_ts[i] if z_ts is not None and i < len(z_ts) else ""
            t = t_ts[i] if t_ts is not None and i < len(t_ts) else ""
            f.write(f"{i},{d},{o},{z},{t}\n")

    return density_plot, phys_plot, z_plot, t_plot


def make_density_timeseries_gif(density_ts, gif_path: Path):
    """
    Make a simple GIF that gradually draws the density time series.
    """
    if density_ts is None or len(density_ts) == 0:
        return None

    gif_frames = []
    frames = np.arange(len(density_ts))

    # Choose at most 100 frames for GIF to keep file size reasonable
    step = max(1, len(frames) // 100)

    for end in range(1, len(frames) + 1, step):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(frames[:end], density_ts[:end], lw=2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Global density")
        ax.set_title("Density Evolution")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        import io
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = imageio.imread(buf)
        gif_frames.append(img)

    imageio.mimsave(gif_path, gif_frames, duration=0.08)
    return gif_path


def try_projection_gif_from_snapshots(exp_dir: Path, vis_dir: Path):
    """
    OPTIONAL: If multiple 3D snapshots are saved as universe_frame_*.npy,
    build a projection GIF across frames.
    """
    pattern = "universe_frame_"
    snapshots = sorted(
        p for p in exp_dir.glob("*.npy") if p.name.startswith(pattern)
    )
    if not snapshots:
        return None

    gif_frames = []
    for p in snapshots:
        grid = np.load(p)
        proj = grid.sum(axis=2)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(
            proj, cmap="inferno", origin="lower", interpolation="nearest"
        )
        ax.set_title(p.name)
        ax.axis("off")
        plt.tight_layout()

        import io
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = imageio.imread(buf)
        gif_frames.append(img)

    gif_path = vis_dir / "projection_evolution.gif"
    imageio.mimsave(gif_path, gif_frames, duration=0.08)
    return gif_path


# ----------------------------------------------------------
# 3D export data
# ----------------------------------------------------------

def make_3d_exports(universe: np.ndarray, export_dir: Path):
    export_dir.mkdir(exist_ok=True, parents=True)

    # Save overdensity field for external tools
    density = universe.astype(float)
    mean_density = density.mean()
    delta = density / mean_density - 1.0
    np.save(export_dir / "universe_delta.npy", delta)

    # Save occupied voxel coordinates (value > 0)
    coords = np.argwhere(universe > 0)
    csv_path = export_dir / "occupied_voxels.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("x,y,z\n")
        for x, y, z in coords:
            f.write(f"{x},{y},{z}\n")

    meta = {
        "analysis": "EST 3D export",
        "grid_size": list(map(int, universe.shape)),
        "occupied_voxel_count": int(coords.shape[0]),
        "timestamp": datetime.now().isoformat(),
    }
    with (export_dir / "export_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return csv_path, meta


# ----------------------------------------------------------
# Unified analysis around EST simulation outputs
# ----------------------------------------------------------

def load_optional_npy(exp_dir: Path, name: str):
    p = exp_dir / name
    return np.load(p) if p.exists() else None


def analyze_experiment_folder(exp_dir: Path):
    """
    Perform unified analysis on an EST experiment folder.

    Expected files (some optional):
        final_universe_state.npy  (required for P(k))
        density_timeseries.npy
        physical_density_timeseries.npy
        redshift_timeseries.npy
        time_gyr_timeseries.npy
    """
    print(f"\n=== EST Unified Analysis ===")
    print(f"Experiment folder: {exp_dir}")

    final_path = exp_dir / "final_universe_state.npy"
    if not final_path.exists():
        raise FileNotFoundError(
            f"final_universe_state.npy not found in {exp_dir}"
        )

    # Load main 3D field
    universe = np.load(final_path)
    if universe.ndim != 3:
        raise ValueError(
            f"final_universe_state.npy must be 3D, got shape {universe.shape}"
        )

    analysis_dir = exp_dir / "Analysis"
    ps_dir = analysis_dir / "PowerSpectrum"
    vis_dir = analysis_dir / "Visualizations"
    corr_dir = analysis_dir / "Correlation"
    halo_dir = analysis_dir / "HaloCatalog"
    export_dir = analysis_dir / "3D"

    # Power spectrum (canonical)
    ps_results = compute_power_spectrum_est(universe, source_file=str(final_path))
    ps_meta = save_power_spectrum_results(ps_results, ps_dir)
    ps_plot = create_power_spectrum_plot(ps_results, ps_dir)
    delta2_plot = compute_and_save_dimensionless_power(ps_results, ps_dir)

    # Correlation function
    corr_plot = compute_correlation_function(universe, corr_dir)

    # Halo catalog
    halo_catalog_path, halo_meta = find_halos(universe, halo_dir)

    # Time series and structure
    density_ts = load_optional_npy(exp_dir, "density_timeseries.npy")
    phys_ts = load_optional_npy(exp_dir, "physical_density_timeseries.npy")
    z_ts = load_optional_npy(exp_dir, "redshift_timeseries.npy")
    t_ts = load_optional_npy(exp_dir, "time_gyr_timeseries.npy")

    struct_metric = structure_metric_from_grid(universe)

    # Extra visualizations
    proj_paths = make_universe_projections(universe, vis_dir)
    hist_path = make_density_histogram(universe, vis_dir)

    # Time series plots + CSV + GIF
    density_plot, phys_plot, z_plot, t_plot = make_timeseries_plots_and_csv(
        exp_dir, analysis_dir, density_ts, phys_ts, z_ts, t_ts
    )
    gif_path = None
    if density_ts is not None:
        ts_dir = analysis_dir / "TimeSeries"
        gif_path = ts_dir / "density_evolution.gif"
        gif_path = make_density_timeseries_gif(density_ts, gif_path)

    # Optional projection GIF from snapshots (if present)
    proj_gif_path = try_projection_gif_from_snapshots(exp_dir, vis_dir)

    # 3D exports
    vox_csv, export_meta = make_3d_exports(universe, export_dir)

    # Build textual summary
    summary_lines = []
    summary_lines.append("EST COSMOLOGY UNIFIED ANALYSIS")
    summary_lines.append("================================")
    summary_lines.append("")
    summary_lines.append(f"Experiment folder: {exp_dir.name}")
    summary_lines.append(
        f"Grid size: {universe.shape[0]} x {universe.shape[1]} x {universe.shape[2]}"
    )
    summary_lines.append(
        f"Mean occupancy (simulation units): {float(universe.mean()):.6f}"
    )
    summary_lines.append(
        f"Structure metric (projection variance/mean, capped): {struct_metric:.4f}"
    )
    summary_lines.append("")

    summary_lines.append("Power Spectrum (canonical, compute_pk-equivalent):")
    summary_lines.append(f"  Spectral index n_s: {ps_results['n_s']:.6f}")
    summary_lines.append(
        f"  Fit range: k = [{ps_results['fit_range'][0]}, "
        f"{ps_results['fit_range'][1]}] grid units"
    )
    summary_lines.append(f"  Number of k-bins: {len(ps_results['k_centers'])}")
    summary_lines.append(f"  Nyquist frequency: {ps_results['knyq']}")
    summary_lines.append(f"  P(k) plot:        {ps_plot}")
    summary_lines.append(f"  Δ^2(k) plot:      {delta2_plot}")
    summary_lines.append(f"  Data directory:   {ps_dir}")
    summary_lines.append("")

    summary_lines.append("Correlation function:")
    summary_lines.append(f"  ξ(r) plot:        {corr_plot}")
    summary_lines.append(f"  Data directory:   {corr_dir}")
    summary_lines.append("")

    summary_lines.append("Halo catalog (downsampled):")
    summary_lines.append(f"  Halos:            {halo_meta['halo_count']}")
    summary_lines.append(f"  Catalog:          {halo_catalog_path}")
    summary_lines.append(f"  Labels:           {halo_dir / 'halo_labels_downsampled.npy'}")
    summary_lines.append("")

    if density_ts is not None:
        d = density_ts
        summary_lines.append("Density timeseries:")
        summary_lines.append(f"  Frames:           {len(d)}")
        summary_lines.append(f"  Initial density:  {d[0]:.6f}")
        summary_lines.append(f"  Final density:    {d[-1]:.6f}")
        if d[0] > 0:
            pct = (d[-1] / d[0] - 1.0) * 100.0
            summary_lines.append(f"  Relative change:  {pct:.2f} %")
        summary_lines.append(f"  Min / Max:        {d.min():.6f} / {d.max():.6f}")
        summary_lines.append(f"  Plot:             {density_plot}")
        if gif_path is not None:
            summary_lines.append(f"  GIF:              {gif_path}")
        summary_lines.append("")

    if phys_ts is not None:
        Ω = phys_ts
        summary_lines.append("Physical Ω_m timeseries:")
        summary_lines.append(f"  Final Ω_m:        {Ω[-1]:.6f}")
        summary_lines.append(f"  Min / Max Ω_m:    {Ω.min():.6f} / {Ω.max():.6f}")
        if phys_plot is not None:
            summary_lines.append(f"  Ω_m vs frame:     {phys_plot}")
        if z_plot is not None:
            summary_lines.append(f"  Ω_m vs z:         {z_plot}")
        if t_plot is not None:
            summary_lines.append(f"  Ω_m vs time:      {t_plot}")
        summary_lines.append("")

    if z_ts is not None:
        z = z_ts
        summary_lines.append("Redshift range:")
        summary_lines.append(f"  z ∈ [{z.min():.3f}, {z.max():.3f}]")
        summary_lines.append("")

    if t_ts is not None:
        t = t_ts
        summary_lines.append("Cosmic time range:")
        summary_lines.append(f"  t ∈ [{t.min():.3f}, {t.max():.3f}] Gyr")
        summary_lines.append("")

    summary_lines.append("Universe visualizations:")
    summary_lines.append(f"  Projections/slices: {vis_dir}")
    summary_lines.append(f"  Histogram:          {hist_path}")
    if proj_gif_path is not None:
        summary_lines.append(f"  Projection GIF:     {proj_gif_path}")
    summary_lines.append("")

    summary_lines.append("3D export data:")
    summary_lines.append(f"  Overdensity field:  {export_dir / 'universe_delta.npy'}")
    summary_lines.append(f"  Occupied voxels:    {vox_csv}")
    summary_lines.append(f"  Export directory:   {export_dir}")
    summary_lines.append("")

    summary = "\n".join(summary_lines)

    # Save summary
    analysis_dir.mkdir(exist_ok=True, parents=True)
    summary_path = analysis_dir / "est_analysis_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)

    print("\n" + summary)
    print(f"\nSummary written to: {summary_path}")
    print(f"Analysis outputs written under: {analysis_dir}")

    return summary_path, analysis_dir


# ----------------------------------------------------------
# Simple Tkinter GUI wrapper (folder picker + run)
# ----------------------------------------------------------

def main():
    root = tk.Tk()
    root.withdraw()  # hide main window

    messagebox.showinfo(
        "EST Unified Analysis",
        "Select an EST experiment folder (the one containing final_universe_state.npy)."
    )

    folder = filedialog.askdirectory(
        title="Select EST experiment folder",
        initialdir=os.getcwd()
    )

    if not folder:
        messagebox.showwarning("EST Unified Analysis", "No folder selected. Exiting.")
        return

    exp_dir = Path(folder)

    try:
        summary_path, analysis_dir = analyze_experiment_folder(exp_dir)
        messagebox.showinfo(
            "EST Unified Analysis",
            f"Analysis complete.\n\nSummary:\n  {summary_path}\n\nAll outputs in:\n  {analysis_dir}"
        )
    except Exception as e:
        messagebox.showerror(
            "EST Unified Analysis - Error",
            f"Analysis failed:\n{type(e).__name__}: {e}"
        )
        raise


if __name__ == "__main__":
    main()
