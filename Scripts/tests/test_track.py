import json
import math

import pytest

import config
from sim.geometry import Segment, Vec2, raycast
from sim.track import Track, TrackLoadError


def sample_track() -> Track:
    return Track(
        walls=[
            Segment(Vec2(0, 0), Vec2(800, 0)),
            Segment(Vec2(800, 0), Vec2(800, 600)),
            Segment(Vec2(800, 600), Vec2(0, 600)),
            Segment(Vec2(0, 600), Vec2(0, 0)),
        ],
        checkpoints=[
            Segment(Vec2(200, 0), Vec2(200, 600)),
            Segment(Vec2(400.5, 0), Vec2(400.5, 600)),
        ],
        start_pos=Vec2(100, 300),
        start_angle=3.14159,
    )


def test_track_convert():
    track = Track(
        walls=[
            Segment(start=Vec2(100, 100), end=Vec2(300, 300)),
            Segment(start=Vec2(300, 200), end=Vec2(100, 150)),
        ],
        checkpoints=[
            Segment(start=Vec2(150, 150), end=Vec2(250, 250)),
            Segment(start=Vec2(250, 200), end=Vec2(150, 100)),
        ],
        start_pos=Vec2(200, 200),
        start_angle=40.6,
    )
    assert Track.from_dict(track.to_dict()) == track


# --- Save and load ---


def test_save_then_load_gives_the_same_track(tmp_path):
    path = tmp_path / "track.json"
    track = sample_track()
    track.save(path)
    assert Track.load(path) == track


def test_save_creates_the_file(tmp_path):
    path = tmp_path / "track.json"
    sample_track().save(path)
    assert path.exists()


def test_saved_file_is_valid_json_matching_to_dict(tmp_path):
    path = tmp_path / "track.json"
    track = sample_track()
    track.save(path)
    assert json.loads(path.read_text()) == track.to_dict()


def test_saving_twice_overwrites_the_old_file(tmp_path):
    path = tmp_path / "track.json"
    sample_track().save(path)
    other = Track(walls=[], checkpoints=[], start_pos=Vec2(5, 5), start_angle=0.0)
    other.save(path)
    assert Track.load(path) == other


def test_empty_track_round_trips(tmp_path):
    path = tmp_path / "empty.json"
    track = Track(walls=[], checkpoints=[], start_pos=Vec2(0, 0), start_angle=0.0)
    track.save(path)
    assert Track.load(path) == track


# --- Loading errors ---


def test_loading_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        Track.load(tmp_path / "does_not_exist.json")


def test_loading_broken_json_raises_track_load_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("this is not json")
    with pytest.raises(TrackLoadError):
        Track.load(path)


def test_loading_an_empty_file_raises_track_load_error(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("")
    with pytest.raises(TrackLoadError):
        Track.load(path)


def test_loading_a_cut_off_file_raises_track_load_error(tmp_path):
    # Like a save that crashed halfway through writing.
    path = tmp_path / "cut_off.json"
    path.write_text(json.dumps(sample_track().to_dict())[:40])
    with pytest.raises(TrackLoadError):
        Track.load(path)


def test_broken_json_error_keeps_the_original_cause(tmp_path):
    # This is what `raise ... from err` gives you.
    path = tmp_path / "broken.json"
    path.write_text("{oops")
    with pytest.raises(TrackLoadError) as excinfo:
        Track.load(path)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_error_message_names_the_file(tmp_path):
    path = tmp_path / "my_bad_track.json"
    path.write_text("not json")
    with pytest.raises(TrackLoadError) as excinfo:
        Track.load(path)
    assert "my_bad_track.json" in str(excinfo.value)


# --- Valid JSON, but not a track ---


def test_missing_key_raises_track_load_error(tmp_path):
    data = sample_track().to_dict()
    del data["walls"]
    path = tmp_path / "no_walls.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TrackLoadError):
        Track.load(path)


def test_wall_missing_a_point_raises_track_load_error(tmp_path):
    data = sample_track().to_dict()
    del data["walls"][0]["end"]
    path = tmp_path / "half_wall.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TrackLoadError):
        Track.load(path)


def test_wrong_type_raises_track_load_error(tmp_path):
    data = sample_track().to_dict()
    data["walls"] = 5  # a number where a list should be
    path = tmp_path / "wrong_type.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TrackLoadError):
        Track.load(path)


def test_json_list_instead_of_object_raises_track_load_error(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(TrackLoadError):
        Track.load(path)


# --- The real oval.json ---

OVAL_PATH = config.DATA_DIR / "tracks" / "oval.json"


def test_data_dir_points_at_the_data_folder():
    assert config.DATA_DIR.is_dir()
    assert (config.DATA_DIR / "tracks").is_dir()


def test_oval_loads():
    assert isinstance(Track.load(OVAL_PATH), Track)


def test_oval_has_walls():
    assert len(Track.load(OVAL_PATH).walls) > 0


def test_oval_has_checkpoints():
    assert len(Track.load(OVAL_PATH).checkpoints) > 0


def test_oval_start_is_inside_the_window():
    start = Track.load(OVAL_PATH).start_pos
    assert 0 <= start.x <= config.WINDOW_WIDTH
    assert 0 <= start.y <= config.WINDOW_HEIGHT


def test_oval_start_is_boxed_in_by_walls():
    # Rays in 16 directions should all hit a wall, so the car can't drive off the track.
    track = Track.load(OVAL_PATH)
    far = 2000.0
    for i in range(16):
        angle = i * 2 * math.pi / 16
        assert raycast(track.start_pos, angle, far, track.walls) < far


def test_oval_start_is_not_touching_a_wall():
    track = Track.load(OVAL_PATH)
    for i in range(16):
        angle = i * 2 * math.pi / 16
        assert raycast(track.start_pos, angle, 2000.0, track.walls) > 10
