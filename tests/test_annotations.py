"""Annotation geometry: what the model may ask for, and where it lands."""

from __future__ import annotations

import math

import pytest

from ui.annotations import (
    GRID,
    MAX_SHAPES,
    Shape,
    box_from_points,
    from_pixels,
    parse_annotation,
    parse_shape,
    to_pixels,
    wobbly_circle,
    wobbly_line,
)

# ── parsing what the model sent ──────────────────────────────────────────


def test_a_shape_with_an_unknown_kind_is_dropped() -> None:
    assert parse_shape({"kind": "explosion", "x": 10, "y": 10}) is None


def test_coordinates_outside_the_grid_are_clamped_not_rejected() -> None:
    """A model that overshoots slightly still meant somewhere on the screen."""
    shape = parse_shape({"kind": "ring", "x": 1400, "y": -30})
    assert shape is not None
    assert (shape.x, shape.y) == (GRID, 0.0)


def test_a_ring_cannot_swallow_the_whole_screen() -> None:
    shape = parse_shape({"kind": "ring", "x": 500, "y": 500, "r": 900})
    assert shape is not None
    assert shape.r <= GRID / 3


def test_a_zero_radius_ring_is_widened_to_something_visible() -> None:
    shape = parse_shape({"kind": "ring", "x": 500, "y": 500, "r": 0})
    assert shape is not None
    assert shape.r >= 6


def test_a_degenerate_box_is_dropped() -> None:
    assert parse_shape({"kind": "box", "x": 10, "y": 10, "w": 1, "h": 1}) is None


def test_an_arrow_that_goes_nowhere_is_dropped() -> None:
    assert parse_shape({"kind": "arrow", "x": 100, "y": 100, "x2": 101, "y2": 100}) is None


def test_a_label_without_text_is_dropped() -> None:
    assert parse_shape({"kind": "label", "x": 100, "y": 100}) is None


def test_nan_and_infinity_fall_back_to_zero() -> None:
    shape = parse_shape({"kind": "ring", "x": float("nan"), "y": float("inf")})
    assert shape is not None
    assert not math.isnan(shape.x)
    assert shape.y in (0.0, GRID)


def test_one_bad_shape_does_not_lose_the_good_ones() -> None:
    """Four marks in the right place beat none because a fifth was malformed."""
    annotation = parse_annotation(
        {
            "shapes": [
                {"kind": "ring", "x": 100, "y": 100},
                {"kind": "nonsense"},
                {"kind": "ring", "x": 200, "y": 200},
            ]
        }
    )
    assert len(annotation.shapes) == 2


def test_a_flood_of_shapes_is_capped() -> None:
    annotation = parse_annotation(
        {"shapes": [{"kind": "ring", "x": i, "y": i} for i in range(50)]}
    )
    assert len(annotation.shapes) == MAX_SHAPES


def test_unparseable_json_is_an_empty_annotation_not_a_crash() -> None:
    assert parse_annotation("{not json").shapes == []


def test_a_zero_ttl_means_the_annotation_persists() -> None:
    """Walkthrough steps stay until the step is passed."""
    assert parse_annotation({"shapes": [{"kind": "ring"}], "ttl_ms": 0}).persistent


def test_a_positive_ttl_does_not_persist() -> None:
    assert not parse_annotation({"shapes": [{"kind": "ring"}], "ttl_ms": 2000}).persistent


# ── the grid ─────────────────────────────────────────────────────────────


def test_grid_coordinates_scale_to_the_screen() -> None:
    shape = to_pixels(Shape(kind="ring", x=500, y=500, r=100), 1920, 1080)
    assert shape.x == pytest.approx(960)
    assert shape.y == pytest.approx(540)


def test_a_ring_stays_circular_on_a_wide_monitor() -> None:
    """Scaling the radius by each axis separately would draw an ellipse."""
    shape = to_pixels(Shape(kind="ring", x=500, y=500, r=100), 3440, 1440)
    assert shape.r == pytest.approx(100 / GRID * 1440)


def test_pixels_round_trip_back_to_the_grid() -> None:
    x, y = from_pixels(960, 540, 1920, 1080)
    assert (x, y) == pytest.approx((500.0, 500.0))


def test_a_zero_sized_screen_does_not_divide_by_zero() -> None:
    assert from_pixels(10, 10, 0, 0) == (0.0, 0.0)


def test_a_stroke_becomes_its_bounding_box() -> None:
    assert box_from_points([(10, 20), (40, 5), (25, 60)]) == (10, 5, 30, 55)


def test_an_empty_stroke_has_no_box() -> None:
    assert box_from_points([]) == (0.0, 0.0, 0.0, 0.0)


# ── the hand-drawn look ──────────────────────────────────────────────────


def test_a_shape_wobbles_the_same_way_every_frame() -> None:
    """Re-rolled jitter is a shape that vibrates instead of one that was drawn."""
    shape = Shape(kind="ring", x=100, y=100, r=50)
    assert wobbly_circle(0, 0, 50, shape.seed) == wobbly_circle(0, 0, 50, shape.seed)


def test_two_different_shapes_wobble_differently() -> None:
    a = Shape(kind="ring", x=100, y=100, r=50).seed
    b = Shape(kind="ring", x=101, y=100, r=50).seed
    assert wobbly_circle(0, 0, 50, a) != wobbly_circle(0, 0, 50, b)


def test_the_wobble_stays_close_to_the_true_shape() -> None:
    """A pointer that misses its target by 30% is not pointing at anything."""
    points = wobbly_circle(0.0, 0.0, 100.0, seed=12345)
    radii = [math.hypot(x, y) for x, y in points]
    assert all(85 <= r <= 115 for r in radii)


def test_a_wobbly_line_still_starts_and_ends_where_it_was_told() -> None:
    points = wobbly_line(0.0, 0.0, 100.0, 0.0, seed=7)
    assert points[0] == pytest.approx((0.0, 0.0))
    assert points[-1] == pytest.approx((100.0, 0.0))
