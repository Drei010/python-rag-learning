from services.classifier_service import classify_sheet


def test_classify_sheet_returns_structured_for_uniform_table():
    values = [
        ["name", "age", "city"],
        ["Alice", 30, "Manila"],
        ["Ben", 25, "Cebu"],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_classify_sheet_returns_structured_for_blank_cells_inside_table():
    values = [
        ["name", "age", "city"],
        ["Alice", 30, "Manila"],
        ["Ben", None, "Cebu"],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_classify_sheet_returns_unstructured_for_mixed_column_types():
    values = [
        ["name", "age", "city"],
        ["Alice", 30, "Manila"],
        ["Ben", "unknown", "Cebu"],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "columns contain mixed data types"


def test_classify_sheet_returns_unstructured_for_non_uniform_styles():
    values = [
        ["name", "age"],
        ["Alice", 30],
        ["Ben", 25],
    ]
    styles = [
        [1, 1],
        [2, 2],
        [2, 3],
    ]

    classification, reason = classify_sheet(values, styles)

    assert classification == "unstructured"
    assert reason == "cell styles are not uniform"


def test_classify_sheet_returns_unstructured_for_dirty_pivot_layout():
    values = [
        [
            "Segment>>",
            "Consumer",
            None,
            None,
            None,
            "Consumer Total",
            "Corporate",
        ],
        [
            "Ship Mode>>",
            "First Class",
            "Same Day",
            "Second Class",
            "Standard Class",
            None,
            "First Class",
        ],
        ["Order ID", None, None, None, None, None, None],
        ["CA-2011-100706", None, None, 129.44, None, 129.44, None],
        ["CA-2011-100895", None, None, None, 605.47, 605.47, None],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "header row contains blank cells"
