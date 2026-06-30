"""
Metropolis-Hastings MCMC Sampler with adaptive proposal.
"""
import numpy as np
from typing import Callable, Optional, Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm


class MetropolisHastings:
    """
    Metropolis-Hastings MCMC Sampler with adaptive proposal.
    
    This implementation includes:
    - Automatic proposal tuning via Robbins-Monro adaptation
    - Convergence diagnostics
    - Chain history for visualization
    """
    
    def __init__(
        self,
        target_log_pdf: Callable,
        proposal_scale: float = 1.0,
        n_dim: int = 2,
        adapt_scale: bool = True,
        target_acceptance: float = 0.234
    ):
        """
        Parameters
        ----------
        target_log_pdf : Callable
            Log of the target probability density function (unnormalized is fine)
        proposal_scale : float
            Initial proposal distribution scale (standard deviation)
        n_dim : int
            Dimensionality of the target distribution
        adapt_scale : bool
            Whether to adapt proposal scale during burn-in
        target_acceptance : float
            Optimal acceptance rate for adaptation (0.234 for high dims)
        """
        self.target_log_pdf = target_log_pdf
        self.proposal_scale = proposal_scale
        self.n_dim = n_dim
        self.adapt_scale = adapt_scale
        self.target_acceptance = target_acceptance
        
        # Chain history
        self.samples = None
        self.log_pdf_values = None
        self.acceptance_history = []
        self.proposal_scale_history = []
        self.all_samples = None
        
    def _proposal(self, current: np.ndarray) -> np.ndarray:
        """Generate proposal using Gaussian random walk."""
        return current + np.random.normal(0, self.proposal_scale, size=self.n_dim)
    
    def _log_acceptance_ratio(
        self, 
        current: np.ndarray, 
        proposed: np.ndarray,
        current_log_pdf: float,
        proposed_log_pdf: float
    ) -> float:
        """
        Calculate log acceptance ratio for Metropolis-Hastings.
        
        For symmetric proposal (Gaussian), proposal ratio = 1, so it cancels.
        """
        return proposed_log_pdf - current_log_pdf
    
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
        Run the MCMC sampler.
        
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
            Dictionary with chain statistics and history
        """
        # Initialize
        if initial_state is None:
            current = np.random.randn(self.n_dim) * 2
        else:
            current = initial_state.copy()
            
        current_log_pdf = self.target_log_pdf(current)
        
        # Storage
        all_samples = []
        all_log_pdfs = []
        n_accepted = 0
        
        # Burn-in phase
        n_total = burn_in + n_samples * thin
        iterator = tqdm(range(n_total), desc="Sampling") if progress_bar else range(n_total)
        
        for i in iterator:
            # Generate proposal
            proposed = self._proposal(current)
            proposed_log_pdf = self.target_log_pdf(proposed)
            
            # Acceptance probability
            log_accept = self._log_acceptance_ratio(
                current, proposed, current_log_pdf, proposed_log_pdf
            )
            
            # Accept or reject
            if np.log(np.random.random()) < log_accept:
                current = proposed
                current_log_pdf = proposed_log_pdf
                n_accepted += 1
            
            # Store (even during burn-in for diagnostics)
            all_samples.append(current.copy())
            all_log_pdfs.append(current_log_pdf)
            
            # Adapt proposal scale during burn-in
            if self.adapt_scale and i < burn_in:
                acceptance_rate = n_accepted / (i + 1) if i > 0 else 0
                if i > 100:  # Start adapting after initial phase
                    # Robbins-Monro style adaptation
                    learning_rate = np.min([0.01, 1.0 / np.sqrt(i + 1)])
                    self.proposal_scale *= np.exp(
                        learning_rate * (acceptance_rate - self.target_acceptance)
                    )
            
            # Store diagnostics periodically
            if i % 100 == 0:
                self.acceptance_history.append(n_accepted / (i + 1) if i > 0 else 0)
                self.proposal_scale_history.append(self.proposal_scale)
        
        # Extract post-burn-in samples with thinning
        samples = np.array(all_samples[burn_in::thin])
        log_pdfs = np.array(all_log_pdfs[burn_in::thin])
        
        self.samples = samples
        self.log_pdf_values = log_pdfs
        self.all_samples = np.array(all_samples)
        
        # Calculate diagnostics
        acceptance_rate = n_accepted / n_total
        
        if verbose:
            print(f"\nSampling completed:")
            print(f"  Acceptance rate: {acceptance_rate:.3f}")
            print(f"  Final proposal scale: {self.proposal_scale:.3f}")
            print(f"  Number of samples: {len(samples)}")
        
        # Diagnostics dictionary
        diagnostics = {
            'acceptance_rate': acceptance_rate,
            'proposal_scale_final': self.proposal_scale,
            'samples': samples,
            'log_pdf_values': log_pdfs,
            'all_samples': np.array(all_samples),
            'all_log_pdfs': np.array(all_log_pdfs),
            'acceptance_history': np.array(self.acceptance_history),
            'proposal_scale_history': np.array(self.proposal_scale_history),
            'n_accepted': n_accepted,
            'n_total': n_total,
            'burn_in': burn_in,
            'thin': thin
        }
        
        return samples, diagnostics
    
    def plot_chain(self, figsize=(12, 8)):
        """
        Visualize the MCMC chain trace and distributions.
        
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
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
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
        
        # Acceptance rate over time
        if len(self.acceptance_history) > 0:
            axes[0, 2].plot(self.acceptance_history)
            axes[0, 2].axhline(y=0.234, color='r', linestyle='--', label='Target (0.234)')
            axes[0, 2].set_title('Acceptance Rate')
            axes[0, 2].set_xlabel('Iteration (x100)')
            axes[0, 2].legend()
        
        # Proposal scale over time
        if len(self.proposal_scale_history) > 0:
            axes[1, 2].plot(self.proposal_scale_history)
            axes[1, 2].set_title('Proposal Scale')
            axes[1, 2].set_xlabel('Iteration (x100)')
        
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