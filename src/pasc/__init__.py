"""pasc: code of the thesis "Predicting Post-Acute Sequelae of COVID-19 from
Electronic Health Records: The Added Value of Mechanism-Based Feature Sets".

Subpackages
    config    repository paths and OMOP schema settings
    db        connection to the OMOP CDM database
    cohort    cohort construction and the PASC label
    features  feature families: Antony et al. A-E, extended covariates,
              engagement controls, mechanism indicators, laboratory summaries,
              composites, configuration builder
    modeling  the repeated hold-out pipeline (balancing, Boruta, forest, SHAP)
              and calibration measures
    analysis  paired tests shared by the reporting scripts

The entry points live in ``scripts/``; see the README.
"""

__version__ = "1.0.0"
