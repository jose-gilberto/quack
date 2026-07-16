# Quack: Quantification in Python

Welcome to **Quack** 🦆! 

**Quack** is a lightweight, clean, and extensible Python library dedicated to **Quantification** (also known as prevalence estimation).

Unlike standard classification tasks that aim to predict the labels of individual instances, quantification algorithms aim to estimate the class distribution (prevalences) of a target population. This is particularly useful in scenarios subject to dataset shift, such as prior probability shift and covariate shift.

---

## Why quack?

* **Pure Python & SciPy Stack**: Built on top of NumPy, SciPy, and Scikit-Learn.
* **Unified Interface**: All quantifiers implement a clean, Scikit-Learn-compatible `fit`/`predict` API.
* **Shift Simulation**: Built-in bag generators allow you to easily simulate and test algorithms against controlled dataset shifts.
* **Core Metrics**: Standard error metrics specifically designed to measure quantification performance.

---

## Next Steps

| Feature       | Status                                    |
| ------------- | ----------------------------------------- |
| Implement bag generation and test pipelines | <input type="checkbox" disabled/> |
| Implement ensembles strategies | <input type="checkbox" disabled/> |
| Create the visualization module    | <input type="checkbox" disabled/> |

And some other features that aren't listed here. 

---

## Features

- Implementation of quantification baselines as Classify & Count, and its variants (Adjusted Classify & Count, Probabilistic Classify & Count, Probabilistic Adjusted Classify & Count).
- Implementation of Threshold based quantification methods as T50, Max, X and MedianSweep.
- Implementation of iteration based quantification methods as Expectation-Maximization Quantifier (EM).
- Implementation of feature based quantification methods as HDx and ReadMe.
- Implementation of distribution based quantification methods as HDy, DyS, FMM, etc.
- Dataset loaders for almost 30 UCI datasets (used on Quantification Review papers).
  - **Next steps** will include add more loaders to classical papers as (Forman 2008), hierarchical and multilabel dataset.
  - Loaders to signal based datasets from PhysioNet.

---

## Citing

If you use Quack, please consider citing the repository (at the moment we do not have any publication related to the original repository), in your paper.

```tex
@misc{quack,
  author = {de Medeiros Júnior, J.G.B.},
  title = {Quack: A Quantification Kit Library for Python},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jose-gilberto/quack}},
}
```

## Contributing

If you want to contribute in any way, look at the `CONTRIBUTING.md` file, and the instructions at the Contribution page. We appreciate that! :)