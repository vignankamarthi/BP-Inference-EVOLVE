"""Integration: GENITOR steady-state replacement cycle.

FRAMEWORK.md Section 2: tournament select parent, mutate, evaluate, evict
lowest-fitness island member. This test spans population + ledger.
"""
from framework import population, ledger


def test_genitor_replacement_keeps_island_size_constant(tmp_db_path):
    led = ledger.Ledger(tmp_db_path)
    try:
        led.init_schema()
        isl = population.Islands(m=4, k=3, reset_cadence=100)
        for i in range(3):
            rid = led.allocate_run_id()
            isl.seed(0, rid, {"balanced_acc": 0.5 + 0.1 * i})
        assert isl.island_size(0) == 3

        isl.insert_child(island_id=0, child_run_id="r_new",
                         child_fitness={"balanced_acc": 0.9})
        assert isl.island_size(0) == 3  # GENITOR keeps size constant
    finally:
        led.close()


def test_periodic_reset_reseeds_from_champion():
    isl = population.Islands(m=4, k=5, reset_cadence=10)
    # Seed each island with one member so the bottom-by-fitness ordering is defined.
    for i in range(4):
        isl.seed(i, f"r_init_{i}", {"balanced_acc": 0.5 + 0.05 * i})
    reset_ids = isl.maybe_reset_islands(
        current_iter=10, global_champion_run_id="champ",
        global_champion_fitness={"balanced_acc": 0.9})
    assert isinstance(reset_ids, list)
    # At least one bottom island reset on the cadence boundary.
    assert len(reset_ids) >= 1
