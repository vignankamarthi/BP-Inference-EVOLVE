"""Integration: full single-iteration round trip.

Exercises ledger + population + mutation + constraints + render in sequence.
HIPs are mocked (no real cluster touch). Verifies that the loop's state
transitions match the workflow in FRAMEWORK.md Section 8.
"""
from framework import ledger, population, mutation, constraints, render, loop


def test_one_iteration_full_chain(tmp_db_path, sample_program_spec, tmp_path):
    """Build islands, seed one, sample parent, assemble prompt, constraint-check,
    render, and persist to ledger. End-to-end integration of all six modules."""
    led = ledger.Ledger(tmp_db_path)
    try:
        led.init_schema()
        isl = population.Islands(m=4, k=5, reset_cadence=100, rng_seed=0)
        # Seed island 0 so sample_parent has someone to pick.
        parent_id = led.allocate_run_id()
        led.write_experiment(parent_id, sample_program_spec, parent_id=None,
                             island_id=0)
        isl.seed(0, parent_id, {"balanced_acc": 0.5})

        parent_rid = isl.sample_parent(island_id=0, tournament_size=3)
        assert parent_rid == parent_id

        meta = mutation.MetaState(p_lit=0.5, novelty_alpha=0.3, temperature=0.7,
                                  failure_boost_active=False)
        prompt = mutation.assemble_mutation_prompt(
            parent_spec=sample_program_spec,
            island_best_specs=[sample_program_spec],
            recent_failures=[],
            meta=meta,
        )
        assert isinstance(prompt, str) and "## Parent program" in prompt

        v = constraints.rule_guards(sample_program_spec, max_params=10_000_000,
                                    max_train_seconds=1800)
        assert v is None

        run_dir = tmp_path / "child"
        out = render.render_spec_to_code(sample_program_spec, run_dir)
        assert out.exists()
    finally:
        led.close()


def test_iteration_pauses_at_hip_d(tmp_path):
    """The loop driver must yield IterationPaused at HIP-D before any cluster touch."""
    out = loop.advance_one_iteration(experiments_root=tmp_path / "experiments",
                                     ledger_path=tmp_path / "ledger.db")
    assert isinstance(out, loop.IterationPaused)
    assert out.hip == "HIP-D"
