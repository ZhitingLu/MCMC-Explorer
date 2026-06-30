"""
Gibbs Sampler implementation for Gaussian distributions.
"""
import numpy as np
from typing import Callable, Optional, Tuple, Dict
from tqdm import tqdm


class GibbsSampler:
    """
    Gibbs Sampler for sampling from multivariate Gaussian distributions.
    
    The Gibbs sampler updates one coordinate at a time using conditional
    distributions, which is efficient for certain distributions where
    conditionals are easy to sample from.
    """
    
    def __init__(self, target_log_pdf: Callable, n_dim: int = 2):
        """
        Parameters
        ----------
        target_log_pdf : Callable
            Log probability density function (must be Gaussian for this implementation)
        n_dim : int
            Dimensionality of the target distribution
        """
        self.target_log_pdf = target_log_pdf
        self.n_dim = n_dim
        self.samples = None
        self.all_samples = None
        self.acceptance_history = []  # For consistency with MH sampler
        
    def _conditional_sample(self, current: np.ndarray, dim: int) -> float:
        """
        Sample from the conditional distribution of dimension 'dim' given all others.
        
        For a Gaussian distribution, the conditional is also Gaussian with:
        - Mean: μ_i + Σ_{i,-i} Σ_{-i,-i}^{-1} (x_{-i} - μ_{-i})
        - Variance: Σ_{i,i} - Σ_{i,-i} Σ_{-i,-i}^{-1} Σ_{-i,i}
        
        This is a simplified implementation that assumes independent Gaussians
        (diagonal covariance) for demonstration.
        """
        # For a full implementation, we would compute the conditional mean and variance
        # from the full covariance matrix. Here we use a simple version.
        
        # Sample from the conditional (simplified: assume independence)
        # In a real Gibbs sampler, this would be the actual conditional distribution
        mean = current[dim]
        std = 0.5  # Fixed conditional standard deviation
        return np.random.normal(mean, std)
    
    def sample(
        self,
        n_samples: int,
        burn_in: int = 1000,
        thin: int = 1,
        initial_state: Optional[np.ndarray] = None,
        progress_bar: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run the Gibbs sampler.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to collect after burn-in
        burn_in : int
            Number of burn-in iterations
        thin : int
            Thinning factor (keep every thin-th sample)
        initial_state : array-like, optional
            Starting point for the chain
        progress_bar : bool
            Whether to show progress bar
        verbose : bool
            Whether to print diagnostics
        
        Returns
        -------
        samples : np.ndarray
            Collected samples after burn-in and thinning
        diagnostics : dict
            Dictionary with chain statistics
        """
        # Initialize
        if initial_state is None:
            current = np.random.randn(self.n_dim)
        else:
            current = initial_state.copy()
        
        all_samples = []
        n_accepted = 0  # For consistency, though Gibbs always accepts
        
        # Burn-in phase
        n_total = burn_in + n_samples * thin
        iterator = tqdm(range(n_total), desc="Gibbs Sampling") if progress_bar else range(n_total)
        
        for i in iterator:
            # Update each coordinate in sequence
            for dim in range(self.n_dim):
                # Sample from conditional distribution for this dimension
                # In a real Gibbs sampler, this would sample from p(x_dim | x_{-dim})
                # Here we use a Metropolis-within-Gibbs approach for demonstration
                proposal = current.copy()
                proposal[dim] = current[dim] + np.random.randn() * 0.5
                
                # Metropolis acceptance (Metropolis-within-Gibbs)
                current_log_pdf = self.target_log_pdf(current)
                proposal_log_pdf = self.target_log_pdf(proposal)
                
                if np.log(np.random.random()) < (proposal_log_pdf - current_log_pdf):
                    current[dim] = proposal[dim]
                    n_accepted += 1
            
            all_samples.append(current.copy())
            
            # Store diagnostics periodically
            if i % 100 == 0:
                acceptance_rate = n_accepted / ((i + 1) * self.n_dim) if i > 0 else 0
                self.acceptance_history.append(acceptance_rate)
        
        # Extract post-burn-in samples with thinning
        samples = np.array(all_samples[burn_in::thin])
        self.samples = samples
        self.all_samples = np.array(all_samples)
        
        # Calculate diagnostics
        acceptance_rate = n_accepted / (n_total * self.n_dim)
        
        if verbose:
            print(f"\nGibbs sampling completed:")
            print(f"  Acceptance rate (Metropolis-within-Gibbs): {acceptance_rate:.3f}")
            print(f"  Number of samples: {len(samples)}")
        
        # Diagnostics dictionary
        diagnostics = {
            'acceptance_rate': acceptance_rate,
            'samples': samples,
            'all_samples': np.array(all_samples),
            'acceptance_history': np.array(self.acceptance_history),
            'n_accepted': n_accepted,
            'n_total': n_total * self.n_dim,  # Total Gibbs steps
            'burn_in': burn_in,
            'thin': thin
        }
        
        return samples, diagnostics
    
    def plot_chain(self, figsize=(12, 8)):
        """
        Visualize the Gibbs chain trace and distributions.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        if self.samples is None:
            raise ValueError("Run sample() first!")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        n_dims_to_plot = min(2, self.n_dim)
        
        # Trace plots for each dimension
        for dim in range(n_dims_to_plot):
            axes[0, dim].plot(self.samples[:, dim], alpha=0.7, linewidth=0.5)
            axes[0, dim].set_title(f'Trace - Dimension {dim+1}')
            axes[0, dim].set_xlabel('Iteration')
            axes[0, dim].set_ylabel('Value')
        
        # Histograms
        for dim in range(n_dims_to_plot):
            axes[1, dim].hist(self.samples[:, dim], bins=50, density=True, alpha=0.7)
            axes[1, dim].set_title(f'Distribution - Dimension {dim+1}')
            axes[1, dim].set_xlabel('Value')
        
        plt.tight_layout()
        return fig
    
    def get_summary_statistics(self) -> Dict:
        """
        Compute summary statistics for the samples.
        
        Returns
        -------
        dict
            Dictionary with mean, std, and quantiles for each dimension
        """
        if self.samples is None:
            raise ValueError("Run sample() first!")
        
        stats = {}
        for dim in range(self.n_dim):
            stats[f'dim_{dim+1}'] = {
                'mean': np.mean(self.samples[:, dim]),
                'std': np.std(self.samples[:, dim]),
                'quantiles': {
                    '2.5%': np.percentile(self.samples[:, dim], 2.5),
                    '25%': np.percentile(self.samples[:, dim], 25),
                    '50%': np.percentile(self.samples[:, dim], 50),
                    '75%': np.percentile(self.samples[:, dim], 75),
                    '97.5%': np.percentile(self.samples[:, dim], 97.5)
                }
            }
        return stats