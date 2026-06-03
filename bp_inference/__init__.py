"""PPG-only cuffless blood-pressure adapter for the experiment engine.

Problem-specific layer: PulseDB loaders, subject-disjoint splits, calibration
logic, the regression + AAMI/BHS metric suite, and the model families. The
generic engine in `framework/` dispatches to the `run_from_dir(run_dir,
data_root)` entry point of each family registered in
`framework.render.FAMILY_ENTRY_POINTS`.
"""

TARGETS = ("sbp", "dbp")
