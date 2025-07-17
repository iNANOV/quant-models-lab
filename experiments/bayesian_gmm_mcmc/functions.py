import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from scipy.stats import norm, beta, halfnorm
from sklearn.mixture import GaussianMixture


def bayesian_gmm_mcmc(
    data,
    n_components=3,
    random_state=42,
    prior_mu_std=0.02,
    prior_sigma_scale=0.02,
    prior_pi_alpha=5,
    prior_pi_beta=5,
    sigma_bounds=(0.005, 0.5),
    penalty_weight=1000,
    n_samples=60000,
    step_size=0.0005,
    burn_in=10000,
    thin=10,
    plot_traces=True,
    plot_densities=True
):
    """
    Bayesian Gaussian Mixture Model with Metropolis-Hastings Sampling.

    Parameters
    ----------
    data : array-like
        Input data array.
    n_components : int
        Number of Gaussian components.
    random_state : int
        Random seed for reproducibility.
    prior_mu_std : float
        Std deviation for normal prior on means.
    prior_sigma_scale : float
        Scale for half-normal prior on standard deviations.
    prior_pi_alpha : float
        Alpha parameter for Beta prior on weights.
    prior_pi_beta : float
        Beta parameter for Beta prior on weights.
    sigma_bounds : tuple
        (min_sigma, max_sigma) bounds for standard deviations.
    penalty_weight : float
        Penalty weight for enforcing sum(pi) ≈ 1.
    n_samples : int
        Total MCMC samples to draw.
    step_size : float
        Proposal step size.
    burn_in : int
        Number of initial samples to discard.
    thin : int
        Thinning factor for MCMC samples.
    plot_traces : bool
        Plot parameter trace plots.
    plot_densities : bool
        Plot posterior density estimates.

    Returns
    -------
    samples : ndarray
        Posterior samples for all parameters.
    """

    x = np.asarray(data)

    # === GMM Initialization ===
    gmm = GaussianMixture(n_components=n_components, random_state=random_state).fit(x.reshape(-1, 1))
    mu_init = gmm.means_.flatten()
    log_sigma_init = np.log(np.sqrt(gmm.covariances_).flatten())
    pi_init = gmm.weights_

    # Sort components by ascending means
    sorted_idx = np.argsort(mu_init)
    mu_init = mu_init[sorted_idx]
    log_sigma_init = log_sigma_init[sorted_idx]
    pi_init = pi_init[sorted_idx]

    print(f"📊 GMM Initialization:\nμ: {mu_init}\nlog(σ): {log_sigma_init}\nπ: {pi_init}")

    # --- Helper functions ---
    def log_likelihood(x, mu, log_sigma, pi):
        sigma = np.exp(log_sigma)
        pdf = sum(pi[k] * norm.pdf(x, mu[k], sigma[k]) for k in range(n_components))
        return np.sum(np.log(pdf + 1e-9))  # avoid log(0)

    def log_prior(mu, log_sigma, pi):
        sigma = np.exp(log_sigma)

        # Reject extreme sigma values
        if np.any(sigma < sigma_bounds[0]) or np.any(sigma > sigma_bounds[1]):
            return -np.inf

        # Normal prior on means
        mu_prior = sum(norm.logpdf(mu[k], mu_init[k], prior_mu_std) for k in range(n_components))

        # Half-normal prior on sigmas
        sigma_prior = sum(halfnorm.logpdf(sigma[k], scale=prior_sigma_scale) for k in range(n_components))

        # Beta prior on pi
        pi_prior = sum(beta.logpdf(pi[k], prior_pi_alpha, prior_pi_beta) for k in range(n_components))

        # Penalty to enforce sum(pi) = 1
        pi_sum_penalty = -penalty_weight * abs(np.sum(pi) - 1)

        return mu_prior + sigma_prior + pi_prior + pi_sum_penalty

    def log_posterior(x, mu, log_sigma, pi):
        return log_likelihood(x, mu, log_sigma, pi) + log_prior(mu, log_sigma, pi)

    # --- Metropolis-Hastings ---
    samples = []
    mu, log_sigma, pi = mu_init.copy(), log_sigma_init.copy(), pi_init.copy()
    current_log_post = log_posterior(x, mu, log_sigma, pi)
    accept_count = 0

    for i in range(n_samples):
        # Propose new parameters
        mu_prop = mu + np.random.normal(0, step_size, n_components)
        log_sigma_prop = log_sigma + np.random.normal(0, step_size, n_components)
        pi_prop = pi + np.random.normal(0, step_size, n_components)

        # Clip and normalize pi
        pi_prop = np.clip(pi_prop, 1e-6, 1 - 1e-6)
        pi_prop /= np.sum(pi_prop)

        # Enforce ordering
        order_idx = np.argsort(mu_prop)
        mu_prop = mu_prop[order_idx]
        log_sigma_prop = log_sigma_prop[order_idx]
        pi_prop = pi_prop[order_idx]

        # Accept/reject
        sigma_prop = np.exp(log_sigma_prop)
        if np.any(sigma_prop < sigma_bounds[0]) or np.any(sigma_prop > sigma_bounds[1]):
            accept = False
        else:
            prop_log_post = log_posterior(x, mu_prop, log_sigma_prop, pi_prop)
            alpha = np.exp(prop_log_post - current_log_post)
            accept = np.random.rand() < min(1, alpha)

        if accept:
            mu, log_sigma, pi = mu_prop, log_sigma_prop, pi_prop
            current_log_post = prop_log_post
            accept_count += 1

        samples.append(np.concatenate([mu, np.exp(log_sigma), pi]))

    samples = np.array(samples)
    print(f"✅ Acceptance rate: {accept_count / n_samples:.2%}")

    # --- Postprocessing ---
    thinned_samples = samples[burn_in::thin]

    mu_post = np.mean(thinned_samples[:, :n_components], axis=0)
    sigma_post = np.mean(thinned_samples[:, n_components:2*n_components], axis=0)
    pi_post = np.mean(thinned_samples[:, 2*n_components:], axis=0)

    print(f"Posterior μ: {mu_post}")
    print(f"Posterior σ: {sigma_post}")
    print(f"Posterior π: {pi_post}")

    # --- Plotting ---
    if plot_traces:
        labels = [rf'$\mu_{k}$' for k in range(n_components)] + \
                 [rf'$\sigma_{k}$' for k in range(n_components)] + \
                 [rf'$\pi_{k}$' for k in range(n_components)]
        fig, axs = plt.subplots(n_components, 3, figsize=(12, 3 * n_components))
        axs = axs.flat
        for i in range(3 * n_components):
            axs[i].plot(samples[:, i], alpha=0.6)
            axs[i].set_title(labels[i])
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()

    if plot_densities:
        fig, axs = plt.subplots(n_components, 3, figsize=(12, 3 * n_components))
        axs = axs.flat
        for i in range(3 * n_components):
            sns.kdeplot(thinned_samples[:, i], fill=True, ax=axs[i])
            axs[i].set_title(labels[i])
            axs[i].grid(True)
        plt.tight_layout()
        plt.show()

    return thinned_samples, mu_init, log_sigma_init, pi_init, mu_post, sigma_post, pi_post


def plot_gmm_vs_posterior(
    mu_init, log_sigma_init, pi_init,
    mu_post, sigma_post, pi_post,
    x_range=None,
    n_points=1000
):
    """
    Plot initial GMM components and posterior Gaussian mixture components.

    Parameters
    ----------
    mu_init : array-like
        Initial means μ_k from GMM initialization.
    log_sigma_init : array-like
        Log standard deviations (log σ_k) from GMM initialization.
    pi_init : array-like
        Initial mixing weights π_k from GMM initialization.
    mu_post : array-like
        Posterior means μ_k from MCMC.
    sigma_post : array-like
        Posterior standard deviations σ_k from MCMC.
    pi_post : array-like
        Posterior mixing weights π_k from MCMC.
    x_range : tuple, optional
        (min, max) range for x-axis values. If None, calculated automatically.
    n_points : int, optional
        Number of points in x grid for plotting.

    Returns
    -------
    None
    """
    sigma_init = np.exp(log_sigma_init)

    # Define x range for plotting if not provided
    if x_range is None:
        x_min = min(np.min(mu_init - 4*sigma_init), np.min(mu_post - 4*sigma_post))
        x_max = max(np.max(mu_init + 4*sigma_init), np.max(mu_post + 4*sigma_post))
        x_range = (x_min, x_max)

    x = np.linspace(x_range[0], x_range[1], n_points)

    # Compute PDFs for initial GMM components
    pdf_init = np.zeros_like(x)
    for k in range(len(mu_init)):
        pdf_init += pi_init[k] * norm.pdf(x, mu_init[k], sigma_init[k])

    # Compute PDFs for posterior components
    pdf_post = np.zeros_like(x)
    for k in range(len(mu_post)):
        pdf_post += pi_post[k] * norm.pdf(x, mu_post[k], sigma_post[k])

    plt.figure(figsize=(10, 6))

    # Plot initial components separately
    for k in range(len(mu_init)):
        plt.plot(x, pi_init[k]*norm.pdf(x, mu_init[k], sigma_init[k]),
                 linestyle='--', label=f'Init Comp {k+1}', alpha=0.7)
    # Plot initial mixture (sum)
    plt.plot(x, pdf_init, linestyle='--', color='black', label='Init GMM Mixture')

    # Plot posterior components separately
    for k in range(len(mu_post)):
        plt.plot(x, pi_post[k]*norm.pdf(x, mu_post[k], sigma_post[k]),
                 linestyle='-', label=f'Posterior Comp {k+1}', alpha=0.9)
    # Plot posterior mixture (sum)
    plt.plot(x, pdf_post, linestyle='-', color='red', linewidth=2, label='Posterior Mixture')

    plt.title('Initial GMM vs Posterior Gaussian Components')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_gamma_t(x, mu, sigma, pi):
    """
    Compute posterior regime probabilities (γₜ) for any number of regimes.
    
    Parameters:
        x      : 1D array of data points
        mu     : array of means for each regime
        sigma  : array of std deviations for each regime
        pi     : array of weights for each regime
        
    Returns:
        gamma_t : 2D array of shape (len(x), n_regimes)
    """
    n_regimes = len(mu)
    likelihoods = np.array([pi[k] * norm.pdf(x, mu[k], sigma[k]) for k in range(n_regimes)])
    denom = np.sum(likelihoods, axis=0) + 1e-12  # Avoid division by zero
    gamma_t = likelihoods / denom
    return gamma_t.T  # shape: (T, n_regimes)

def plot_regime_probabilities(x, mu, sigma, pi, dates=None, labels=None, colors=None):
    """
    Plot posterior probabilities for multiple regimes with optional datetime x-axis.

    Parameters:
        x       : 1D array of data (returns)
        mu      : regime means
        sigma   : regime stds
        pi      : regime weights
        dates   : (optional) pandas Series or array-like of datetime objects
        labels  : list of labels for regimes (optional)
        colors  : list of colors for plotting (optional)
    """
    gamma_t = compute_gamma_t(x, mu, sigma, pi)
    n_regimes = gamma_t.shape[1]

    x_vals = dates if dates is not None else np.arange(len(x))

    plt.figure(figsize=(14, 6))
    for k in range(n_regimes):
        label = labels[k] if labels else f'Regime {k+1}'
        color = colors[k] if colors else None
        plt.plot(x_vals, gamma_t[:, k], label=label, color=color)
    
    plt.title(f'Posterior Regime Probabilities ({n_regimes} Regimes)')
    plt.xlabel('Date' if dates is not None else 'Time Index')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_upward_crossings(df, gamma_probs, threshold=0.99, marker_type='v',
                          marker_size=20, marker_color='red', type = 'line',
                          title='Upward Crossings', figscale=3):
    """
    Plots an OHLC chart with markers for upward crossings of a threshold in regime probabilities.
    """
    df = df.copy()

    # Force 'date' column to be datetime
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isnull().any():
        raise ValueError("Invalid dates found in 'date' column.")

    df = df.set_index('date')

    # Detect upward crossings
    upward_crossings_idx = np.where(
        (gamma_probs[:-1] < threshold) & (gamma_probs[1:] >= threshold)
    )[0] + 1

    # Create marker positions
    marker_positions = np.full(len(df), np.nan)
    marker_positions[upward_crossings_idx] = df['high'].iloc[upward_crossings_idx] * 1.1

    # Build mplfinance addplot
    apdict = mpf.make_addplot(
        marker_positions,
        type='scatter',
        markersize=marker_size,
        marker=marker_type,
        color=marker_color
    )

    # Plot OHLC with markers
    mpf.plot(
        df,
        type=type,
        style='yahoo',
        addplot=apdict,
        title=title,
        ylabel='Price',
        volume=False,
        figscale=figscale,
        datetime_format='%Y-%m-%d'
    )
