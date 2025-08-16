import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



class GaussianHMM:
    def __init__(self, n_states, n_iter=100, tol=1e-6, seed=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)

    def _initialize_params(self, X):
        N = self.n_states
        self.pi = np.ones(N) / N                            # Initial state distribution
        self.A = np.ones((N, N)) / N                        # Transition matrix
        self.mu = self.rng.normal(loc=0, scale=0.02, size=N)   # Means of Gaussians
        self.sigma = np.ones(N) * 0.01                     # Standard deviations

    def _gaussian_prob(self, x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    def _forward(self, X):
        T = len(X)
        N = self.n_states
        alpha = np.zeros((T, N))
        scale_factors = np.zeros(T)

        # Initialization
        for j in range(N):
            alpha[0, j] = self.pi[j] * self._gaussian_prob(X[0], self.mu[j], self.sigma[j])
        scale_factors[0] = alpha[0].sum()
        alpha[0] /= scale_factors[0]

        # Induction
        for t in range(1, T):
            for j in range(N):
                prob = self._gaussian_prob(X[t], self.mu[j], self.sigma[j])
                alpha[t, j] = prob * np.sum(alpha[t - 1] * self.A[:, j])
            scale_factors[t] = alpha[t].sum()
            alpha[t] /= scale_factors[t]

        return alpha, scale_factors

    def _backward(self, X, scale_factors):
        T = len(X)
        N = self.n_states
        beta = np.zeros((T, N))

        # Initialization
        beta[-1] = 1 / scale_factors[-1]

        # Induction
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum([
                    self.A[i, j] * self._gaussian_prob(X[t + 1], self.mu[j], self.sigma[j]) * beta[t + 1, j]
                    for j in range(N)
                ])
            beta[t] /= scale_factors[t]

        return beta

    def _e_step(self, X):
        T = len(X)
        N = self.n_states

        alpha, scale_factors = self._forward(X)
        beta = self._backward(X, scale_factors)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denom = 0
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (
                        alpha[t, i] *
                        self.A[i, j] *
                        self._gaussian_prob(X[t + 1], self.mu[j], self.sigma[j]) *
                        beta[t + 1, j]
                    )
                    denom += xi[t, i, j]
            xi[t] /= denom

        return gamma, xi

    def _m_step(self, X, gamma, xi):
        T = len(X)
        N = self.n_states

        self.pi = gamma[0]

        # Transition probabilities
        self.A = xi.sum(axis=0) / gamma[:-1].sum(axis=0, keepdims=True).T

        # Gaussian parameters
        for j in range(N):
            weights = gamma[:, j]
            total = weights.sum()
            self.mu[j] = np.sum(weights * X) / total
            self.sigma[j] = np.sqrt(np.sum(weights * (X - self.mu[j])**2) / total)

    def fit(self, X):
        X = np.asarray(X).flatten()
        self._initialize_params(X)

        log_likelihoods = []

        for iteration in range(self.n_iter):
            gamma, xi = self._e_step(X)
            self._m_step(X, gamma, xi)

            # Compute log likelihood as sum of log of scaling factors (more stable)
            _, scale_factors = self._forward(X)
            log_likelihood = np.sum(np.log(scale_factors))
            log_likelihoods.append(log_likelihood)

            if iteration > 0 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

        self.gamma_ = gamma
        self.log_likelihoods_ = log_likelihoods

    def predict(self, X):
        X = np.asarray(X).flatten()
        alpha, _ = self._forward(X)
        return np.argmax(alpha, axis=1)

    def _viterbi(self, X):
        T, N = len(X), self.n_states
        delta = np.zeros((T, N))  # max prob of any path that reaches state i at time t
        psi = np.zeros((T, N), dtype=int)  # backpointers for path

        # Initialization
        for j in range(N):
            delta[0, j] = self.pi[j] * self._gaussian_prob(X[0], self.mu[j], self.sigma[j])
        delta[0] /= delta[0].sum()

        # Recursion
        for t in range(1, T):
            for j in range(N):
                prob = self._gaussian_prob(X[t], self.mu[j], self.sigma[j])
                temp = delta[t - 1] * self.A[:, j]
                psi[t, j] = np.argmax(temp)
                delta[t, j] = prob * np.max(temp)
            delta[t] /= delta[t].sum()

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


def step7_check_convergence(log_likelihoods, tol=1e-6, plot=True):
    """
    Checks convergence based on log-likelihoods.

    Parameters:
    - log_likelihoods: list or array of log-likelihood values per iteration.
    - tol: tolerance threshold for convergence.
    - plot: if True, plots the log-likelihood curve.

    Returns:
    - converged: True if convergence criterion met, else False.
    """

    if len(log_likelihoods) < 2:
        print("Not enough iterations to check convergence.")
        return False

    diff = abs(log_likelihoods[-1] - log_likelihoods[-2])
    converged = diff < tol

    print(f"Log-likelihood difference between last two iterations: {diff:.8f}")
    print("Converged." if converged else "Not converged yet.")

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(log_likelihoods, marker='o')
        plt.title("Log-Likelihood per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Log-Likelihood")
        plt.grid(True)
        plt.show()

    return converged


def _viterbi(self, X):
    T, N = len(X), self.n_states
    delta = np.zeros((T, N))  # highest prob of any path that reaches state i at time t
    psi = np.zeros((T, N), dtype=int)  # argmax backpointer

    # Initialization
    for j in range(N):
        delta[0, j] = self.pi[j] * self._gaussian_prob(X[0], self.mu[j], self.sigma[j])
    # Normalize to prevent underflow
    delta[0] /= delta[0].sum()

    # Recursion
    for t in range(1, T):
        for j in range(N):
            prob = self._gaussian_prob(X[t], self.mu[j], self.sigma[j])
            temp = delta[t - 1] * self.A[:, j]
            psi[t, j] = np.argmax(temp)
            delta[t, j] = prob * np.max(temp)
        delta[t] /= delta[t].sum()

    # Backtracking
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states

def step8_infer_hidden_states(model, X, use_viterbi=False):
    """
    Decode hidden states given observations X using Forward-Backward or Viterbi.

    Parameters:
    - model: fitted GaussianHMM instance
    - X: array-like, observations
    - use_viterbi: bool, if True use Viterbi algorithm, else Forward-Backward

    Returns:
    - states: inferred hidden states sequence
    """
    X = np.asarray(X).flatten()

    if use_viterbi:
        # Use Viterbi decoding
        states = model._viterbi(X)
    else:
        # Use Forward-Backward decoding (posterior mode)
        alpha, scale_factors = model._forward(X)
        beta = model._backward(X, scale_factors)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        states = np.argmax(gamma, axis=1)

    return states

def forecast_all_next_states_and_observations(model, X):
    """
    For each time t in X, compute probabilities of next hidden state and expected next observation.

    Returns:
        P_next_states: array of shape (T-1, n_states) - next hidden state probabilities for each t
        E_next_obs: array of shape (T-1,) - expected next observation for each t
    """
    X = np.asarray(X).flatten()
    T = len(X)
    n_states = model.n_states

    # Run forward-backward to get gamma for all time steps
    alpha, scale_factors = model._forward(X)
    beta = model._backward(X, scale_factors)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)  # shape (T, n_states)

    P_next_states = np.zeros((T-1, n_states))
    E_next_obs = np.zeros(T-1)

    for t in range(T-1):
        # Next state probabilities for time t+1
        P_next_states[t] = np.dot(gamma[t], model.A)
        # Expected next observation
        E_next_obs[t] = np.dot(P_next_states[t], model.mu)

    return P_next_states, E_next_obs

# --- Helper function: evaluate bearish signals ---
def evaluate_bear_signals(df_ohlc, signals, horizon):
    """
    For each signal, compute return from next open to close after horizon periods.
    Count as negative (successful bear signal) if return < 0.
    """
    total_signals = signals.sum()
    negatives = 0

    for i in range(len(df_ohlc) - horizon):
        if signals.iloc[i]:
            next_open = df_ohlc['open'].iloc[i + 1]
            horizon_close = df_ohlc['close'].iloc[i + horizon]
            ret = horizon_close / next_open - 1
            if ret < 0:  # Negative return = correct bearish prediction
                negatives += 1

    ratio = negatives / total_signals if total_signals > 0 else np.nan
    return total_signals, negatives, ratio

# --- Generate Bear performance table ---
def generate_bear_performance_table(df_train, df_test, bear_cross_train, bear_cross_test, horizons=[1, 4, 6, 10]):
    results = {'Horizon': []}
    train_signals, train_negatives, train_ratios = [], [], []
    test_signals, test_negatives, test_ratios = [], [], []

    for h in horizons:
        # --- Train ---
        total_train, negatives_train, ratio_train = evaluate_bear_signals(
            df_ohlc=df_train,
            signals=bear_cross_train,
            horizon=h
        )
        # --- Test ---
        total_test, negatives_test, ratio_test = evaluate_bear_signals(
            df_ohlc=df_test,
            signals=bear_cross_test,
            horizon=h
        )

        results['Horizon'].append(h)
        train_signals.append(total_train)
        train_negatives.append(negatives_train)
        train_ratios.append(f"{ratio_train:.2%}")

        test_signals.append(total_test)
        test_negatives.append(negatives_test)
        test_ratios.append(f"{ratio_test:.2%}")

    table = pd.DataFrame({
        'Horizon': results['Horizon'],
        'Train Signals': train_signals,
        'Train Negatives': train_negatives,
        'Train Negative Ratio': train_ratios,
        'Test Signals': test_signals,
        'Test Negatives': test_negatives,
        'Test Negative Ratio': test_ratios,
    })

    table.set_index('Horizon', inplace=True)
    return table