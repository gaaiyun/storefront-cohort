"""Optional advanced analytics modules (heavy ML dependencies).

These modules are **not** imported by the core report pipeline and are not
required to run ``storefront-cohort``. They pull in scikit-learn / SHAP and are
provided for users who want to go further than RFM + cohorts + CLV:

* ``churn_predictor``    -- supervised churn classification (scikit-learn)
* ``clustering_engine``  -- unsupervised customer clustering (scikit-learn)
* ``shap_explainer``     -- model explainability (SHAP)

Install the extras to use them::

    pip install scikit-learn shap

Import them explicitly, e.g.::

    from storefront_cohort.advanced import clustering_engine

They are intentionally kept out of ``storefront_cohort/__init__.py`` so the
core tool stays light (pandas + numpy only) and never fails to import because
an optional ML dependency is missing.
"""

from __future__ import annotations

__all__ = ["churn_predictor", "clustering_engine", "shap_explainer"]
