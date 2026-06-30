"""
Utility functions for MCMC diagnostics and visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Callable, Union


def compute_effective_sample_size(samples: np.ndarray) -> float:
    """
    Compute the effective sample size (ESS) for a chain.
    
    ESS measures the number of independent samples equivalent to the
    correlated samples from MCMC.
    
    Parameters
    ----------
    samples : np.ndarray
        MCMC samples (n_samples x n_dim)
    
    Returns
    -------
    float
        Effective sample size
    """
    n = len(samples)
    if n < 2:
        return float(n)
    
    # Compute autocorrelation for each dimension
    ess_dims = []
    for dim in range(samples.shape[1]):
        series = samples[:, dim]
        # Center the series
        series = series - np.mean(series)
        
        # Compute autocorrelation up to lag n//2
        max_lag = min(n // 2, 1000)
        autocorr = np.correlate(series, series, mode='full')
        autocorr = autocorr[n-1:] / autocorr[n-1]  # Normalize
        
        # Use Geyer's initial positive sequence estimator
        tau_sum = 0.0
        
        for lag in range(1, max_lag):
            if lag == 1:
                tau_sum = 1 + 2 * autocorr[lag]
            else:
                # Check if next pair is positive
                if autocorr[2*lag-1] + autocorr[2*lag] > 0:
                    tau_sum += 2 * (autocorr[2*lag-1] + autocorr[2*lag])
                else:
                    break
        
        if tau_sum > 0:
            ess = n / tau_sum
        else:
            ess = float(n)
        ess_dims.append(ess)
    
    # Return minimum ESS across dimensions (conservative estimate)
    return float(np.min(ess_dims))


def gelman_rubin_diagnostic(chains: List[np.ndarray]) -> float:
    """
    Compute the Gelman-Rubin R-hat diagnostic for multiple chains.
    
    R-hat measures convergence of MCMC chains. Values < 1.1 indicate
    convergence.
    
    Parameters
    ----------
    chains : List[np.ndarray]
        List of chains (each is n_samples x n_dim)
    
    Returns
    -------
    float
        R-hat statistic (maximum over dimensions)
    """
    n_chains = len(chains)
    n_samples = min(len(chain) for chain in chains)
    
    # Trim all chains to same length
    chains = [chain[:n_samples] for chain in chains]
    
    # Compute for each dimension
    r_hat_dims = []
    for dim in range(chains[0].shape[1]):
        # Get samples for this dimension from all chains
        dim_samples = np.array([chain[:, dim] for chain in chains])
        
        # Between-chain variance
        chain_means = np.mean(dim_samples, axis=1)
        overall_mean = np.mean(chain_means)
        B = n_samples * np.var(chain_means, ddof=1)
        
        # Within-chain variance
        chain_vars = np.var(dim_samples, axis=1, ddof=1)
        W = np.mean(chain_vars)
        
        # Estimate of variance
        var_hat = (n_samples - 1) / n_samples * W + B / n_samples
        
        # R-hat
        r_hat = np.sqrt(var_hat / W)
        r_hat_dims.append(float(r_hat))
    
    return float(np.max(r_hat_dims))  # Conservative: use max over dimensions


def plot_contour(
    target_log_pdf: Callable,
    samples: np.ndarray,
    x_range: Tuple[float, float] = (-6, 6),
    y_range: Tuple[float, float] = (-6, 6),
    n_points: int = 100,
    title: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot contour of target distribution with sampled points overlay.
    
    Parameters
    ----------
    target_log_pdf : callable
        Target log probability density function
    samples : np.ndarray
        MCMC samples (n_samples x 2)
    x_range : tuple
        Range for x-axis
    y_range : tuple
        Range for y-axis
    n_points : int
        Number of grid points per dimension
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    if samples.shape[1] != 2:
        raise ValueError("This function only works for 2D distributions")
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Compute target density on grid
    log_pdf_values = np.array([target_log_pdf(p) for p in grid_points])
    pdf_values = np.exp(log_pdf_values - np.max(log_pdf_values))
    pdf_grid = pdf_values.reshape(X.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot target distribution contours
    contour = ax.contourf(X, Y, pdf_grid, levels=30, cmap='viridis', alpha=0.6)
    ax.contour(X, Y, pdf_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot sampled points
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=5, c='red', label='Samples')
    
    # Plot first 100 samples as a path
    if len(samples) > 100:
        ax.plot(samples[:100, 0], samples[:100, 1], 'b-', alpha=0.3,
                linewidth=0.5, label='Chain path (first 100)')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title if title else 'Target Distribution with MCMC Samples')
    ax.legend()
    plt.colorbar(contour, label='Density')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_figure(
    mh_samples: np.ndarray,
    hmc_samples: np.ndarray,
    target_log_pdf: Callable,
    mh_ess: Optional[float] = None,
    hmc_ess: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a side-by-side comparison of MH and HMC samples.
    
    Parameters
    ----------
    mh_samples : np.ndarray
        Samples from Metropolis-Hastings
    hmc_samples : np.ndarray
        Samples from HMC
    target_log_pdf : callable
        Target log probability density function
    mh_ess : float, optional
        ESS for MH
    hmc_ess : float, optional
        ESS for HMC
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create grid for target distribution
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    log_pdf_values = np.array([target_log_pdf(p) for p in grid_points])
    pdf_values = np.exp(log_pdf_values - np.max(log_pdf_values))
    pdf_grid = pdf_values.reshape(X.shape)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, samples, title, ess in zip(
        axes,
        [mh_samples, hmc_samples],
        ['Metropolis-Hastings', 'Hamiltonian Monte Carlo (Mici)'],
        [mh_ess, hmc_ess]
    ):
        # Target contours
        ax.contour(X, Y, pdf_grid, levels=15, alpha=0.3, colors='black')
        ax.contourf(X, Y, pdf_grid, levels=15, alpha=0.2, cmap='viridis')
        
        # Samples
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=5, c='red')
        
        # ESS text
        if ess is not None:
            ax.text(0.05, 0.95, f'ESS = {ess:.0f}',
                    transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_autocorrelation(samples: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute autocorrelation for each dimension of the samples.
    
    Parameters
    ----------
    samples : np.ndarray
        MCMC samples (n_samples x n_dim)
    max_lag : int, optional
        Maximum lag to compute. Defaults to min(n//2, 1000)
    
    Returns
    -------
    np.ndarray
        Autocorrelation for each dimension (n_lags x n_dim)
    """
    n = len(samples)
    if max_lag is None:
        max_lag = min(n // 2, 1000)
    
    autocorrs = []
    for dim in range(samples.shape[1]):
        series = samples[:, dim]
        series = series - np.mean(series)
        
        autocorr = np.correlate(series, series, mode='full')
        autocorr = autocorr[n-1:] / autocorr[n-1]  # Normalize
        autocorrs.append(autocorr[:max_lag])
    
    return np.column_stack(autocorrs)


def plot_autocorrelation(samples: np.ndarray, max_lag: Optional[int] = None) -> plt.Figure:
    """
    Plot autocorrelation for each dimension.
    
    Parameters
    ----------
    samples : np.ndarray
        MCMC samples (n_samples x n_dim)
    max_lag : int, optional
        Maximum lag to display
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    n = len(samples)
    if max_lag is None:
        max_lag = min(n // 4, 100)
    
    autocorrs = compute_autocorrelation(samples, max_lag)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dim in range(autocorrs.shape[1]):
        ax.plot(autocorrs[:, dim], label=f'Dimension {dim+1}')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation by Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig