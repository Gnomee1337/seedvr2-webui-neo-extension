"""The share of an upscale run that is done — `seedvr2_neo/run_progress.py`."""

import pytest

from seedvr2_neo.run_progress import PHASES, UPSCALE_PHASE, RunProgress


class Reports(list):
    def __call__(self, fraction, text):
        self.append((fraction, text))

    @property
    def fractions(self):
        return [fraction for fraction, _ in self]


def run_one_pass(progress, pass_index, images, blocks=0):
    """Every phase of one pass, each image a batch; `blocks` DiT blocks start per image in the DiT phase."""
    progress.start_pass(pass_index)
    for phase in range(len(PHASES)):
        progress.start_phase(phase, blocks if phase == UPSCALE_PHASE else None)
        for image in range(images):
            if phase == UPSCALE_PHASE:
                for _ in range(blocks):
                    progress.block_started()
            progress.batch_done(image + 1, images)


def test_a_run_starts_at_zero_and_ends_at_one():
    reports = Reports()
    progress = RunProgress([1], reports)
    run_one_pass(progress, 0, 1, blocks=4)
    assert reports.fractions[0] == 0.0
    assert reports.fractions[-1] == pytest.approx(1.0)


def test_the_bar_never_moves_back_through_a_run_of_several_passes():
    reports = Reports()
    progress = RunProgress([2, 1, 3], reports)
    for index, images in enumerate([2, 1, 3]):
        run_one_pass(progress, index, images, blocks=5)
    assert reports.fractions == sorted(reports.fractions)
    assert reports.fractions[-1] == pytest.approx(1.0)


def test_passes_count_by_their_images():
    progress = RunProgress([3, 1], lambda *args: None)
    run_one_pass(progress, 0, 3)
    assert progress.fraction == pytest.approx(0.75)


def test_dit_blocks_move_the_bar_inside_one_image():
    progress = RunProgress([1], lambda *args: None)
    progress.start_phase(UPSCALE_PHASE, 4)
    start = progress.fraction
    progress.block_started()
    progress.block_started()
    halfway = progress.fraction
    total = sum(phase.weight for phase in PHASES)
    assert halfway - start == pytest.approx(PHASES[UPSCALE_PHASE].weight / total / 2)


def test_blocks_never_carry_an_image_past_its_own_share():
    progress = RunProgress([2], lambda *args: None)
    progress.start_phase(UPSCALE_PHASE, 2)
    for _ in range(10):  # e.g. a guidance pass runs the blocks twice
        progress.block_started()
    one_image = progress.fraction
    progress.batch_done(1, 2)
    assert progress.fraction == pytest.approx(one_image)


def test_blocks_count_only_in_the_dit_phase_with_a_known_block_count():
    reports = Reports()
    progress = RunProgress([1], reports)
    progress.start_phase(0)
    progress.block_started()
    progress.start_phase(UPSCALE_PHASE, None)
    progress.block_started()
    assert len(reports) == 2  # only the two phase starts


@pytest.mark.parametrize("current, total", [(None, None), ("a", 2), (1, 0), (object(), 1)])
def test_unreadable_callback_values_are_ignored(current, total):
    progress = RunProgress([1], lambda *args: None)
    progress.start_phase(0)
    progress.batch_done(current, total)
    assert progress.fraction == 0.0


def test_a_callback_past_the_total_does_not_leave_the_phase():
    progress = RunProgress([1], lambda *args: None)
    progress.start_phase(0)
    progress.batch_done(5, 2)
    assert progress.fraction == pytest.approx(PHASES[0].weight / sum(phase.weight for phase in PHASES))


@pytest.mark.parametrize("sizes", [[], [0], [-3, 2]])
def test_empty_or_negative_pass_sizes_still_give_a_fraction(sizes):
    progress = RunProgress(sizes, lambda *args: None)
    progress.start_pass(5)
    progress.start_phase(99)
    progress.batch_done(1, 1)
    assert 0.0 <= progress.fraction <= 1.0


def test_the_text_names_the_phase_and_the_pass_when_there_are_several():
    reports = Reports()
    RunProgress([1], reports).start_phase(UPSCALE_PHASE, 32)
    RunProgress([1, 1], reports).start_pass(1)
    assert [text for _, text in reports] == ["SeedVR2: Upscaling", "SeedVR2: Encoding, pass 2 of 2"]
