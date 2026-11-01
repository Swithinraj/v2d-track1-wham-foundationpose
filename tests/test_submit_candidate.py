import datetime as dt


def test_upload_window():
    from submit_candidate import window_is_open
    utc = dt.timezone.utc
    assert not window_is_open(dt.datetime(2026, 11, 1, 21, 59, tzinfo=utc))
    assert window_is_open(dt.datetime(2026, 11, 1, 22, 0, tzinfo=utc))
    assert window_is_open(dt.datetime(2026, 11, 4, 21, 59, tzinfo=utc))
    assert not window_is_open(dt.datetime(2026, 11, 4, 22, 0, tzinfo=utc))
