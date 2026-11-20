"""Hostile calls for the run's progress — `seedvr2_neo/run_progress.py`.

The pipeline's callback hands over whatever it got (`current`, `total`), DiT blocks can start in any number, and the
host's report function is someone else's code. Checked here: the fraction is always a finite number in 0…1, the text
always names a real phase and pass, odd pass sizes never divide by zero, the real call order never moves the bar back,
and a raising report function surfaces to the caller instead of being swallowed half-way.
"""
import math
import random
from decimal import Decimal
from fractions import Fraction

import pytest

from seedvr2_neo.run_progress import PHASES, UPSCALE_PHASE, RunProgress


class Reports(list):
    def __call__(self, fraction, text):
        self.append((fraction, text))

    @property
    def fractions(self):
        return [fraction for fraction, _ in self]


def assert_sane(reports, passes):
    phase_names = {phase.name for phase in PHASES}
    for fraction, text in reports:
        assert isinstance(fraction, float) and math.isfinite(fraction) and 0.0 <= fraction <= 1.0, fraction
        assert text.startswith("SeedVR2: ")
        name = text[len("SeedVR2: "):].split(",")[0]
        assert name in phase_names
        if passes > 1:
            pass_no, total = text.rsplit("pass ", 1)[1].split(" of ")
            assert 1 <= int(pass_no) <= int(total) == passes


# --- odd pass sizes ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [[], [0], [-5], [0, 0, 0], [10 ** 9], [1, 10 ** 12], [True, False], [2.9, 0.1],
                                   ["3", " 4 "], [Decimal("2"), Fraction(7, 2)]], ids=repr)
def test_odd_pass_sizes_never_give_a_fraction_outside_0_to_1(sizes):
    reports = Reports()
    progress = RunProgress(sizes, reports)
    passes = max(1, len(sizes))
    for index in range(passes):
        progress.start_pass(index)
        for phase in range(len(PHASES)):
            progress.start_phase(phase, 3)
            for _ in range(5):
                progress.block_started()
            progress.batch_done(1, 1)
    assert_sane(reports, passes)
    assert reports.fractions[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("sizes", [[None], ["abc"], [float("nan")], [float("inf")], [object()]], ids=repr)
def test_unreadable_pass_sizes_fail_at_construction_not_later(sizes):
    """The run builds them from `len()` of each pass; anything else is a programming error, raised at once."""
    with pytest.raises((TypeError, ValueError, OverflowError)):
        RunProgress(sizes, Reports())


def test_a_billion_images_in_a_pass_still_moves_the_bar_inside_its_share():
    reports = Reports()
    progress = RunProgress([10 ** 9, 1], reports)
    progress.start_pass(0)
    progress.start_phase(UPSCALE_PHASE, 32)
    for _ in range(31):
        progress.block_started()
    assert 0.0 < reports.fractions[-1] < 1.0 and reports.fractions == sorted(reports.fractions)


# --- odd callback values -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("current, total", [(None, None), (1, 0), (0, 0), ("a", 2), (1, "b"), ([], 1), (1, []),
                                            (complex(1, 1), 2), (object(), 1), (-3, 2), (5, 2), (2, -1), (-1, -1),
                                            ("1", "2"), (1.5, 3), (True, 2), (float("inf"), 1), (1, float("inf")),
                                            (Decimal("1"), Decimal("4")), (10 ** 300, 10 ** 301)], ids=repr)
def test_odd_batch_values_keep_the_fraction_in_0_to_1(current, total):
    reports = Reports()
    progress = RunProgress([2], reports)
    progress.start_pass(0)
    progress.start_phase(2)
    before = progress.fraction
    progress.batch_done(current, total)
    assert_sane(reports, 1)
    assert progress.fraction >= before


@pytest.mark.parametrize("current, total", [(float("nan"), 1), (1, float("nan")), (float("inf"), float("inf")),
                                            (Decimal("NaN"), 1)], ids=repr)
def test_nan_batch_values_never_reach_the_report(current, total):
    reports = Reports()
    progress = RunProgress([1], reports)
    progress.start_phase(1)
    progress.batch_done(current, total)
    assert_sane(reports, 1)


def test_a_huge_int_batch_value_is_ignored_like_other_unreadable_values():
    progress = RunProgress([1], Reports())
    progress.batch_done(10 ** 400, 1)


def test_a_nan_batch_after_a_real_one_does_not_wipe_the_progress():
    reports = Reports()
    progress = RunProgress([1], reports)
    progress.batch_done(1, 2)
    progress.batch_done(float("nan"), 1)
    assert_sane(reports, 1)


# --- out-of-order and repeated calls ---------------------------------------------------------------------------------

@pytest.mark.parametrize("index", [-10 ** 9, -1, 4, 10 ** 9, True])
def test_phase_and_pass_indexes_out_of_range_are_clamped(index):
    reports = Reports()
    progress = RunProgress([1, 2], reports)
    progress.start_pass(index)
    progress.start_phase(index)
    assert_sane(reports, 2)


@pytest.mark.parametrize("blocks", [0, -4, None, 0.5, float("inf"), float("nan"), 10 ** 12, True])
def test_odd_block_counts_never_break_the_dit_phase(blocks):
    reports = Reports()
    progress = RunProgress([3], reports)
    progress.start_phase(UPSCALE_PHASE, blocks)
    for _ in range(100):
        progress.block_started()
    progress.batch_done(1, 3)
    assert_sane(reports, 1)


def test_blocks_outside_the_dit_phase_or_with_an_unknown_count_report_nothing():
    reports = Reports()
    progress = RunProgress([1], reports)
    for phase in range(len(PHASES)):
        progress.start_phase(phase, None if phase == UPSCALE_PHASE else 8)
        count = len(reports)
        for _ in range(50):
            progress.block_started()
        assert len(reports) == count


def test_a_million_blocks_for_one_image_never_pass_that_image():
    progress = RunProgress([4], Reports())
    progress.start_phase(UPSCALE_PHASE, 10)
    for _ in range(1_000_000):
        progress.block_started()
    weights = sum(phase.weight for phase in PHASES)
    one_image_of_dit = PHASES[UPSCALE_PHASE].weight / weights / 4
    assert progress.fraction == pytest.approx(PHASES[0].weight / weights + one_image_of_dit)


def test_random_call_order_keeps_every_report_sane():
    rng = random.Random(2024)
    for _ in range(200):
        sizes = [rng.randint(-2, 6) for _ in range(rng.randint(0, 5))]
        reports = Reports()
        progress = RunProgress(sizes, reports)
        for _ in range(300):
            action = rng.randrange(4)
            if action == 0:
                progress.start_pass(rng.randint(-3, 8))
            elif action == 1:
                progress.start_phase(rng.randint(-2, 6), rng.choice([None, 0, -1, 1, 7, 40]))
            elif action == 2:
                progress.batch_done(rng.randint(-2, 12), rng.randint(-1, 10))
            else:
                progress.block_started()
        assert_sane(reports, max(1, len(sizes)))


def test_the_real_order_never_moves_back_for_any_block_count():
    """Passes in order, phases in order, each image a batch; the DiT may start fewer, exactly or more blocks per image
    than it has (a guidance pass counts blocks twice)."""
    rng = random.Random(7)
    for _ in range(300):
        sizes = [rng.randint(1, 5) for _ in range(rng.randint(1, 4))]
        blocks = rng.randint(1, 40)
        reports = Reports()
        progress = RunProgress(sizes, reports)
        for pass_index, images in enumerate(sizes):
            progress.start_pass(pass_index)
            for phase in range(len(PHASES)):
                progress.start_phase(phase, blocks if phase == UPSCALE_PHASE else None)
                for image in range(images):
                    if phase == UPSCALE_PHASE:
                        for _ in range(rng.choice([0, blocks // 2, blocks, blocks * 2 + 1])):
                            progress.block_started()
                    progress.batch_done(image + 1, images)
        # float rounding may step back by ~1e-17 (0.2 after 0.19999999999999998); Forge's bar has 1000 steps
        assert all(later >= earlier - 1e-9 for earlier, later in zip(reports.fractions, reports.fractions[1:])),             (sizes, blocks)
        assert reports.fractions[-1] == pytest.approx(1.0)


# --- the report function ---------------------------------------------------------------------------------------------

class Boom(Exception):
    pass


@pytest.mark.parametrize("call", ["start_pass", "start_phase", "batch_done", "block_started"])
def test_a_raising_report_surfaces_to_the_caller(call):
    """The progress keeps no half-state to hide: the error goes to the caller (the run's boundary turns it into a
    failure; `tests/seedvr2_neo/test_upscale_run_stress.py`)."""
    def report(fraction, text):
        raise Boom()

    progress = RunProgress([2], report)
    progress._phase = UPSCALE_PHASE
    progress._blocks_per_image = 4
    args = {"start_pass": (0,), "start_phase": (UPSCALE_PHASE, 4), "batch_done": (1, 2), "block_started": ()}[call]
    with pytest.raises(Boom):
        getattr(progress, call)(*args)


def test_a_report_that_fails_once_does_not_corrupt_later_reports():
    calls = []

    def report(fraction, text):
        calls.append(fraction)
        if len(calls) == 2:
            raise Boom()

    progress = RunProgress([1], report)
    progress.start_pass(0)
    with pytest.raises(Boom):
        progress.start_phase(UPSCALE_PHASE, 2)
    progress.block_started()
    progress.batch_done(1, 1)
    assert calls == sorted(calls) and all(0.0 <= value <= 1.0 for value in calls)


def test_a_report_returning_junk_is_ignored():
    progress = RunProgress([1], lambda fraction, text: "not None")
    progress.start_pass(0)
    progress.batch_done(1, 1)
    assert progress.fraction > 0
