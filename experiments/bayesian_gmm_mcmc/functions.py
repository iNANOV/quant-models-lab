import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    return thinned_samples, mu_init, log_sigma_init, pi_init
