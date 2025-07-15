# Quant‑Models‑Lab

A curated playground of quantitative finance notebooks, each designed to be run directly in **Google Colab**. The goal is to make advanced modelling techniques for market data transparent, reproducible, and easy to extend.

---

## 📦 Repository structure


``` 
quant-models-lab/
  experiments/
    bayesian_gmm_mcmc/
      - Bayesian_GMM_MH.ipynb
      - functions.py
      - readme.md
README.md
``` 


*Everything lives in `experiments/` so new ideas stay organised and discoverable.*

---

## 🚀 Quick start

1. **Open in Colab** – Click the Colab badge at the top of any notebook or run `Open with → Google Colab` from the GitHub UI.
2. **Install deps** – Each notebook contains a first‐cell `%pip install` block that pins exact versions (no local environment required).
3. **Run all** – Runtime → Run all. GPU/TPU is optional unless stated.

---

## 🧪 Experiments catalogue

| # | Notebook | Focus | Key techniques |
|---|----------|-------|----------------|
| 1 | **Bayesian_GMM_MH.ipynb** | Bayesian inference of Gaussian Mixture Models on equity return distributions | Metropolis–Hastings MCMC, posterior predictive checks|

> **Coming soon**  
> - Hidden Markov Chain 
> - SWARCH + LSTM 

---

## 🏗️ Adding a new experiment

1. Fork → clone your fork.
2. `mkdir experiments/my_cool_idea && cd` there.
3. Add a notebook and a short Markdown README explaining: *problem*, *data*, *methodology*, *results*.
4. Open a PR. Automated CI will run `nbqa flake8` + smoke tests to keep quality high.

---

## 🤝 Contributing guidelines

* Use Python ≥3.10.
* Keep external dependencies minimal; prefer pure‑Python or widely‑available packages.
* Write notebooks as teaching material: narrative first, code blocks interleaved, tidy outputs.
* Include a YAML header with metadata (`title`, `authors`, `date`, `tags`).

---

## 📜 License

MIT – free for academic and commercial use. See `LICENSE` for full text.

---

## ✨ Acknowledgements

Inspired by open‑source quant communities and the generosity of researchers sharing their work.

*Happy modelling!*
