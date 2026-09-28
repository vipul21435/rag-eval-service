from ragsvc.features.conflict_detector import detect_conflicts


def test_agreeing_sources_are_not_a_conflict():
    sources = [
        {"text": "Revenue grew 12% in 2024.", "type": "local"},
        {"text": "In 2024 revenue grew 12% year over year.", "type": "web"},
    ]

    assert detect_conflicts(sources) is False


def test_disagreeing_numbers_are_a_conflict():
    sources = [
        {"text": "Revenue grew 12% in 2024.", "type": "local"},
        {"text": "Revenue grew 15% in 2024.", "type": "web"},
    ]

    assert detect_conflicts(sources) is True


def test_sources_without_numbers_never_conflict():
    sources = [{"text": "No figures here."}, {"excerpt": "Still no figures."}]

    assert detect_conflicts(sources) is False
    assert detect_conflicts([]) is False
