"""
Unit tests for MCMC samplers.
"""
import pytest
import numpy as np
from src.metropolis_hastings import MetropolisHastings
from src.target_distributions import mixture_of_gaussians_log_pdf
from src.gibbs_sampler import GibbsSampler


def test_metropolis_hastings_initialization():
    """Test that MH sampler initializes correctly."""
    sampler = MetropolisHastings(
        target_log_pdf=mixture_of_gaussians_log_pdf,
        proposal_scale=0.5,
        n_dim=2
    )
    assert sampler.proposal_scale == 0.5
    assert sampler.n_dim == 2
    assert sampler.adapt_scale is True


def test_metropolis_hastings_sampling():
    """Test that MH sampler runs and produces samples."""
    sampler = MetropolisHastings(
        target_log_pdf=mixture_of_gaussians_log_pdf,
        proposal_scale=0.5,
        n_dim=2
    )
    
    samples, diagnostics = sampler.sample(
        n_samples=100,
        burn_in=50,
        thin=1,
        progress_bar=False
    )
    
    assert len(samples) == 100
    assert samples.shape[1] == 2
    assert 'acceptance_rate' in diagnostics
    assert 'proposal_scale_final' in diagnostics
    assert len(diagnostics['acceptance_history']) > 0
    assert 0 < diagnostics['acceptance_rate'] <= 1


def test_metropolis_hastings_with_different_dimensions():
    """Test MH sampler with different dimensionalities."""
    # Test 1D
    sampler_1d = MetropolisHastings(
        target_log_pdf=lambda x: -0.5 * x[0]**2,  # Standard normal
        n_dim=1
    )
    samples, _ = sampler_1d.sample(n_samples=50, burn_in=20, progress_bar=False)
    assert samples.shape[1] == 1
    
    # Test 3D
    sampler_3d = MetropolisHastings(
        target_log_pdf=lambda x: -0.5 * np.sum(x**2),  # Multivariate normal
        n_dim=3
    )
    samples, _ = sampler_3d.sample(n_samples=50, burn_in=20, progress_bar=False)
    assert samples.shape[1] == 3


def test_summary_statistics():
    """Test summary statistics computation."""
    sampler = MetropolisHastings(
        target_log_pdf=mixture_of_gaussians_log_pdf,
        proposal_scale=0.5,
        n_dim=2
    )
    
    samples, _ = sampler.sample(
        n_samples=100,
        burn_in=50,
        thin=1,
        progress_bar=False
    )
    
    stats = sampler.get_summary_statistics()
    assert 'dim_1' in stats
    assert 'dim_2' in stats
    assert 'mean' in stats['dim_1']
    assert 'std' in stats['dim_1']
    assert 'quantiles' in stats['dim_1']
    assert '2.5%' in stats['dim_1']['quantiles']
    assert '97.5%' in stats['dim_1']['quantiles']


def test_plot_chain():
    """Test that plot_chain runs without errors."""
    sampler = MetropolisHastings(
        target_log_pdf=mixture_of_gaussians_log_pdf,
        proposal_scale=0.5,
        n_dim=2
    )
    
    samples, _ = sampler.sample(
        n_samples=50,
        burn_in=20,
        thin=1,
        progress_bar=False
    )
    
    # Should not raise an exception
    fig = sampler.plot_chain()
    assert fig is not None

def test_gibbs_sampler_initialization():
    """Test that Gibbs sampler initializes correctly."""
    from src.gibbs_sampler import GibbsSampler
    from src.target_distributions import mixture_of_gaussians_log_pdf
    
    sampler = GibbsSampler(
        target_log_pdf=mixture_of_gaussians_log_pdf,
        n_dim=2
    )
    assert sampler.n_dim == 2
    assert sampler.samples is None
    assert sampler.all_samples is None


def test_gibbs_sampler_sampling():
    """Test that Gibbs sampler runs and produces samples."""
    from src.gibbs_sampler import GibbsSampler
    from src.target_distributions import mixture_of_gaussians_log_pdf
    
    sampler = GibbsSampler(
        target_log_pdf=mixture_of_gaussians_log_pdf,
        n_dim=2
    )
    
    samples, diagnostics = sampler.sample(
        n_samples=100,
        burn_in=50,
        thin=1,
        progress_bar=False
    )
    
    assert len(samples) == 100
    assert samples.shape[1] == 2
    assert 'acceptance_rate' in diagnostics
    assert len(diagnostics['acceptance_history']) > 0
    assert 0 < diagnostics['acceptance_rate'] <= 1


def test_gibbs_sampler_different_dimensions():
    """Test Gibbs sampler with different dimensionalities."""
    from src.gibbs_sampler import GibbsSampler
    
    # Test 1D
    sampler_1d = GibbsSampler(
        target_log_pdf=lambda x: -0.5 * x[0]**2,
        n_dim=1
    )
    samples, _ = sampler_1d.sample(n_samples=50, burn_in=20, progress_bar=False)
    assert samples.shape[1] == 1
    
    # Test 3D
    sampler_3d = GibbsSampler(
        target_log_pdf=lambda x: -0.5 * np.sum(x**2),
        n_dim=3
    )
    samples, _ = sampler_3d.sample(n_samples=50, burn_in=20, progress_bar=False)
    assert samples.shape[1] == 3


def test_gibbs_summary_statistics():
    """Test Gibbs summary statistics computation."""
    from src.gibbs_sampler import GibbsSampler
    from src.target_distributions import mixture_of_gaussians_log_pdf
    
    sampler = GibbsSampler(
        target_log_pdf=mixture_of_gaussians_log_pdf,
        n_dim=2
    )
    
    samples, _ = sampler.sample(
        n_samples=100,
        burn_in=50,
        thin=1,
        progress_bar=False
    )
    
    stats = sampler.get_summary_statistics()
    assert 'dim_1' in stats
    assert 'dim_2' in stats
    assert 'mean' in stats['dim_1']
    assert 'std' in stats['dim_1']
    assert 'quantiles' in stats['dim_1']
    assert '2.5%' in stats['dim_1']['quantiles']
    assert '97.5%' in stats['dim_1']['quantiles']


def test_gibbs_plot_chain():
    """Test that Gibbs plot_chain runs without errors."""
    from src.gibbs_sampler import GibbsSampler
    from src.target_distributions import mixture_of_gaussians_log_pdf
    
    sampler = GibbsSampler(
        target_log_pdf=mixture_of_gaussians_log_pdf,
        n_dim=2
    )
    
    samples, _ = sampler.sample(
        n_samples=50,
        burn_in=20,
        thin=1,
        progress_bar=False
    )
    
    fig = sampler.plot_chain()
    assert fig is not None


def test_gibbs_on_gaussian():
    """Test Gibbs sampler on a simple Gaussian."""
    from src.gibbs_sampler import GibbsSampler
    
    # Target: 2D independent Gaussian
    target = lambda x: -0.5 * np.sum(x**2)
    sampler = GibbsSampler(target_log_pdf=target, n_dim=2)
    samples, _ = sampler.sample(n_samples=500, burn_in=200, progress_bar=False)
    
    # Mean should be close to 0
    mean = np.mean(samples, axis=0)
    assert np.abs(mean[0]) < 0.5, f"Gibbs dim1 mean: {mean[0]}"
    assert np.abs(mean[1]) < 0.5, f"Gibbs dim2 mean: {mean[1]}"
    
    # Variance should be close to 1 (relaxed tolerance for 500 samples)
    var = np.var(samples, axis=0)
    assert 0.7 < var[0] < 1.3, f"Gibbs dim1 var: {var[0]}"
    assert 0.7 < var[1] < 1.3, f"Gibbs dim2 var: {var[1]}"

def test_metropolis_on_gaussian():
    """Test Metropolis-Hastings on a simple Gaussian."""
    from src.metropolis_hastings import MetropolisHastings
    
    # Target: 2D independent Gaussian
    target = lambda x: -0.5 * np.sum(x**2)
    sampler = MetropolisHastings(target_log_pdf=target, n_dim=2, proposal_scale=0.5)
    samples, _ = sampler.sample(n_samples=500, burn_in=200, progress_bar=False)
    
    # Mean should be close to 0
    mean = np.mean(samples, axis=0)
    assert np.abs(mean[0]) < 0.5, f"MH dim1 mean: {mean[0]}"
    assert np.abs(mean[1]) < 0.5, f"MH dim2 mean: {mean[1]}"
    
    # Variance should be close to 1 (relaxed tolerance for 500 samples)
    var = np.var(samples, axis=0)
    assert 0.7 < var[0] < 1.3, f"MH dim1 var: {var[0]}"
    assert 0.7 < var[1] < 1.3, f"MH dim2 var: {var[1]}"

def test_compute_ess():
    """Test ESS computation."""
    from src.utils import compute_effective_sample_size
    
    # Independent samples should have ESS close to n
    n = 1000
    independent_samples = np.random.randn(n, 2)
    ess = compute_effective_sample_size(independent_samples)
    assert ess > 0.8 * n
    assert ess <= n

def test_gelman_rubin():
    """Test Gelman-Rubin diagnostic."""
    from src.utils import gelman_rubin_diagnostic
    
    # Create multiple chains from a standard normal
    n_chains = 4
    n_samples = 500
    chains = [np.random.randn(n_samples, 2) for _ in range(n_chains)]
    
    r_hat = gelman_rubin_diagnostic(chains)
    assert r_hat < 1.1

def test_plot_autocorrelation():
    """Test autocorrelation plotting."""
    from src.utils import plot_autocorrelation
    
    samples = np.random.randn(1000, 2)
    fig = plot_autocorrelation(samples)
    assert fig is not None