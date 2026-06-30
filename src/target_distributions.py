"""
Target distributions for MCMC sampling experiments.
"""
import numpy as np
from scipy.stats import multivariate_normal


def mixture_of_gaussians_log_pdf(x, means=None, covs=None, weights=None):
    """
    Log probability density for a mixture of Gaussians.
    
    A classic multi-modal distribution that's challenging for MCMC.
    
    Parameters
    ----------
    x : array-like
        Point at which to evaluate the density
    means : list of arrays, optional
        Means of the Gaussian components
    covs : list of arrays, optional
        Covariance matrices of the Gaussian components
    weights : list of floats, optional
        Mixture weights (must sum to 1)
    
    Returns
    -------
    float
        Log probability density
    """
    if means is None:
        # Default: two modes at (-3,-3) and (3,3)
        means = [
            np.array([-3.0, -3.0]),
            np.array([3.0, 3.0])
        ]
    
    if covs is None:
        # Spherical covariances with different widths
        covs = [
            np.eye(2) * 0.8,
            np.eye(2) * 1.2
        ]
    
    if weights is None:
        weights = [0.4, 0.6]  # Unequal mixture weights
    
    # Calculate log pdf for each component
    log_densities = []
    for mean, cov, weight in zip(means, covs, weights):
        log_densities.append(
            np.log(weight) + multivariate_normal.logpdf(x, mean, cov)
        )
    
    # Log-sum-exp trick for numerical stability
    max_log = np.max(log_densities)
    log_pdf = max_log + np.log(np.sum(np.exp(np.array(log_densities) - max_log)))
    
    return log_pdf


def banana_shaped_log_pdf(x):
    """
    A non-linear distribution that's challenging for MH but easy for HMC.
    
    Creates a banana-shaped distribution via transformation:
    y1 = x1
    y2 = x2 - 0.1 * x1^2
    
    Parameters
    ----------
    x : array-like
        Point at which to evaluate the density
    
    Returns
    -------
    float
        Log probability density
    """
    # Transform to banana shape
    u = np.array([x[0], x[1] - 0.1 * x[0]**2])
    return multivariate_normal.logpdf(u, mean=np.zeros(2), cov=np.eye(2))


def create_ring_distribution():
    """
    Create a ring-shaped distribution (another challenging multi-modal).
    
    Returns
    -------
    callable
        Log probability density function
    """
    def log_pdf(x):
        radius = np.sqrt(x[0]**2 + x[1]**2)
        
        # Ring potential
        ring_energy = -0.5 * ((radius - 3.0) / 0.5)**2
        
        # Add a small Gaussian prior at center to prevent divergence
        prior_energy = -0.5 * (radius / 2.0)**2
        
        # Log-sum-exp for mixture of ring and center
        energies = [ring_energy, prior_energy - 5.0]
        max_energy = np.max(energies)
        log_pdf = max_energy + np.log(np.sum(np.exp(np.array(energies) - max_energy)))
        
        return log_pdf
    
    return log_pdf


def correlated_3d_log_pdf(x):
    """
    A 3D distribution with correlation to test higher-dimensional sampling.
    
    Parameters
    ----------
    x : array-like
        Point at which to evaluate the density (length 3)
    
    Returns
    -------
    float
        Log probability density
    """
    mean = np.zeros(3)
    # Create a covariance with correlation
    cov = np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    return multivariate_normal.logpdf(x, mean, cov)


def get_distribution(name):
    """
    Factory function to get a target distribution by name.
    
    Parameters
    ----------
    name : str
        Name of the distribution ('mixture_gaussian', 'banana', 'ring', 'correlated_3d')
    
    Returns
    -------
    callable
        Log probability density function
    """
    distributions = {
        'mixture_gaussian': mixture_of_gaussians_log_pdf,
        'banana': banana_shaped_log_pdf,
        'ring': create_ring_distribution(),
        'correlated_3d': correlated_3d_log_pdf
    }
    return distributions[name]