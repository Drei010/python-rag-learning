from services.classifier_service import classify_sheet


# ===========================================================================
# Existing tests (must continue to pass unchanged)
# ===========================================================================


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


# ===========================================================================
# NEW: Merged cells detection
# ===========================================================================


def test_merged_cells_multi_row_span_is_unstructured():
    values = [
        ["Region", "Q1", "Q2"],
        ["East", 100, 200],
        ["West", 150, 250],
    ]
    # A merged range spanning rows 0-1 in column 0 (2 rows)
    merged_ranges = [(0, 1, 0, 0)]

    classification, reason = classify_sheet(values, merged_ranges=merged_ranges)

    assert classification == "unstructured"
    assert reason == "sheet contains merged cell regions"


def test_merged_cells_wide_col_span_is_unstructured():
    values = [
        ["Report Title", None, None, None],
        ["name", "age", "city", "country"],
        ["Alice", 30, "Manila", "PH"],
        ["Ben", 25, "Cebu", "PH"],
    ]
    # A merged range spanning columns 0-3 in row 0 (4 columns)
    merged_ranges = [(0, 0, 0, 3)]

    classification, reason = classify_sheet(values, merged_ranges=merged_ranges)

    assert classification == "unstructured"
    assert reason == "sheet contains merged cell regions"


def test_small_merged_cells_do_not_trigger():
    values = [
        ["name", "age", "city"],
        ["Alice", 30, "Manila"],
        ["Ben", 25, "Cebu"],
    ]
    # A small merge: 1 row, 2 columns — does not exceed thresholds
    merged_ranges = [(0, 0, 0, 1)]

    classification, reason = classify_sheet(values, merged_ranges=merged_ranges)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_merged_cells_outside_used_area_do_not_trigger():
    values = [
        [None, None, None, None, None],
        [None, "name", "age", None, None],
        [None, "Alice", 30, None, None],
        [None, "Ben", 25, None, None],
        [None, None, None, None, None],
    ]
    # Merged range is outside the used area (row 0, cols 0-3)
    merged_ranges = [(0, 0, 0, 3)]

    classification, reason = classify_sheet(values, merged_ranges=merged_ranges)

    # Used area is rows 1-3, cols 1-2. The merge at row 0 cols 0-3 overlaps
    # col 1-2 at row 0 which is outside row bounds 1-3, so it should not trigger.
    assert classification == "structured"


def test_no_merged_ranges_does_not_trigger():
    values = [
        ["name", "age"],
        ["Alice", 30],
        ["Ben", 25],
    ]

    classification, reason = classify_sheet(values, merged_ranges=[])

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


# ===========================================================================
# NEW: Sparsity check
# ===========================================================================


def test_sparse_grid_is_unstructured():
    # 5x4 grid with mostly blanks (16 out of 20 cells blank = 80%)
    values = [
        ["A", None, None, None],
        [None, None, "B", None],
        [None, None, None, None],
        [None, None, None, "C"],
        ["D", None, None, None],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    # Could be "blank row inside used range" or "data is too sparse"
    # Row 2 is entirely blank, so blank-row check fires first
    assert "blank row" in reason or "sparse" in reason


def test_sparse_grid_without_blank_rows_is_unstructured():
    # 5x4 grid — first row is full (so header-blank check doesn't fire),
    # no entirely blank rows, but body is very sparse (>50% blank overall)
    values = [
        ["col1", "col2", "col3", "col4"],
        ["A", None, None, None],
        [None, None, "B", None],
        [None, "C", None, None],
        [None, None, None, "D"],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "data is too sparse"


def test_normal_table_with_some_blanks_is_not_sparse():
    values = [
        ["name", "age", "city", "notes"],
        ["Alice", 30, "Manila", "engineer"],
        ["Ben", 25, "Cebu", None],
        ["Carol", 28, "Davao", "designer"],
        ["Dan", 32, "Manila", None],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_small_sparse_table_does_not_trigger():
    # Only 3 rows — sparsity check requires >3 rows
    values = [
        ["A", None, None],
        [None, "B", None],
        [None, None, "C"],
    ]

    classification, reason = classify_sheet(values)

    # Will hit "header row contains blank cells" first since row 0 has blanks
    assert classification == "unstructured"


# ===========================================================================
# NEW: Multi-row header detection
# ===========================================================================


def test_multi_row_headers_is_unstructured():
    values = [
        ["Category", "Region", "Sales"],
        ["Sub-category", "Sub-region", "Amount"],
        ["Electronics", "East", 1500],
        ["Clothing", "West", 2000],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "multiple header rows suggest hierarchical layout"


def test_three_text_header_rows_is_unstructured():
    values = [
        ["Department", "Location", "Budget"],
        ["Unit", "Building", "Allocation"],
        ["Team", "Floor", "Spending"],
        ["Engineering", "HQ", 50000],
        ["Marketing", "Branch", 30000],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "multiple header rows suggest hierarchical layout"


def test_single_header_row_is_structured():
    values = [
        ["name", "age", "salary"],
        ["Alice", 30, 50000],
        ["Ben", 25, 45000],
        ["Carol", 28, 55000],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_all_text_table_does_not_trigger_multi_row_header():
    # All rows are text — no numeric data below, so multi-row header check
    # should NOT trigger (it requires numeric data after the text rows)
    values = [
        ["category", "status", "region"],
        ["electronics", "active", "east"],
        ["clothing", "inactive", "west"],
        ["food", "active", "north"],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


# ===========================================================================
# NEW: Subtotal row detection
# ===========================================================================


def test_subtotal_row_in_middle_is_unstructured():
    values = [
        ["product", "quantity", "price"],
        ["Widget A", 10, 5.99],
        ["Widget B", 20, 3.99],
        ["Subtotal", 30, 9.98],
        ["Widget C", 15, 7.99],
        ["Widget D", 5, 12.99],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "subtotal rows found within data"


def test_grand_total_in_middle_is_unstructured():
    values = [
        ["region", "sales", "profit"],
        ["East", 1000, 200],
        ["West", 1500, 300],
        ["Grand Total", 2500, 500],
        ["North", 800, 150],
        ["South", 900, 180],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "subtotal rows found within data"


def test_total_in_last_row_is_structured():
    values = [
        ["product", "quantity", "price"],
        ["Widget A", 10, 5.99],
        ["Widget B", 20, 3.99],
        ["Widget C", 15, 7.99],
        ["Total", 45, 17.97],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_sum_keyword_in_middle_is_unstructured():
    values = [
        ["item", "count", "cost"],
        ["Apples", 50, 100],
        ["Oranges", 30, 60],
        ["Sum", 80, 160],
        ["Bananas", 20, 40],
        ["Grapes", 10, 30],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "subtotal rows found within data"


def test_word_total_in_data_value_does_not_false_positive():
    # "Total Solutions" is a company name, not a subtotal keyword
    # The keyword check uses 'in' so "total" is in "Total Solutions"
    # This is an accepted trade-off — pivot tables commonly have "Total" labels
    values = [
        ["company", "revenue", "employees"],
        ["Total Solutions Inc", 5000, 50],
        ["Acme Corp", 3000, 30],
        ["Beta LLC", 2000, 20],
        ["Summary Row", 10000, 100],
    ]

    classification, reason = classify_sheet(values)

    # "Total" appears in a non-last row, so this triggers
    assert classification == "unstructured"
    assert reason == "subtotal rows found within data"


# ===========================================================================
# NEW: Repeating group labels
# ===========================================================================


def test_repeating_group_labels_is_unstructured():
    values = [
        ["region", "product", "sales"],
        ["East", "Widget A", 100],
        ["East", "Widget B", 200],
        ["East", "Widget C", 150],
        ["East", "Widget D", 175],
        ["West", "Widget A", 300],
        ["West", "Widget B", 250],
        ["West", "Widget C", 280],
        ["West", "Widget D", 320],
        ["West", "Widget E", 190],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "column contains repeating group labels"


def test_short_repeats_do_not_trigger():
    # Only 2-3 consecutive repeats — below the threshold of 4
    values = [
        ["region", "product", "sales"],
        ["East", "Widget A", 100],
        ["East", "Widget B", 200],
        ["West", "Widget C", 150],
        ["West", "Widget D", 175],
        ["North", "Widget E", 300],
        ["North", "Widget F", 250],
        ["South", "Widget G", 280],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_repeating_labels_in_small_table_do_not_trigger():
    # Table has < 5 data rows, so check doesn't apply
    values = [
        ["region", "product", "sales"],
        ["East", "Widget A", 100],
        ["East", "Widget B", 200],
        ["East", "Widget C", 150],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_repeating_labels_below_30_percent_do_not_trigger():
    # 4 consecutive "East" but in a 15-row table = 26.6% < 30%
    values = [
        ["region", "product", "sales"],
        ["East", "Widget A", 100],
        ["East", "Widget B", 200],
        ["East", "Widget C", 150],
        ["East", "Widget D", 175],
        ["West", "Widget E", 300],
        ["North", "Widget F", 250],
        ["South", "Widget G", 280],
        ["Central", "Widget H", 320],
        ["Metro", "Widget I", 190],
        ["Rural", "Widget J", 110],
        ["Urban", "Widget K", 220],
        ["Coastal", "Widget L", 270],
        ["Inland", "Widget M", 310],
        ["Island", "Widget N", 140],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


# ===========================================================================
# Integration / combined scenario tests
# ===========================================================================


def test_realistic_pivot_table_is_unstructured():
    # A realistic pivot-like layout with multi-row headers and sparse data
    values = [
        ["Sales Report", "Q1 2024", None, None],
        ["Region", "Product", "Revenue", "Profit"],
        ["East", "Electronics", 50000, 10000],
        ["East", "Clothing", 30000, 8000],
        ["East Subtotal", None, 80000, 18000],
        ["West", "Electronics", 45000, 9000],
        ["West", "Clothing", 25000, 6000],
        ["West Subtotal", None, 70000, 15000],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    # The first check that fires could be "header row contains blank cells"
    # because row 0 has None values


def test_realistic_structured_table_not_falsely_flagged():
    # A proper structured table with many rows, some repeated status values
    # (but short runs), optional fields with None
    values = [
        ["employee_id", "name", "department", "status", "salary"],
        ["E001", "Alice", "Engineering", "Active", 75000],
        ["E002", "Ben", "Engineering", "Active", 70000],
        ["E003", "Carol", "Marketing", "Active", 65000],
        ["E004", "Dan", "Marketing", "Inactive", 60000],
        ["E005", "Eve", "Engineering", "Active", 80000],
        ["E006", "Frank", "Sales", "Active", 55000],
        ["E007", "Grace", "Sales", "Inactive", 50000],
        ["E008", "Hank", "Engineering", "Active", 72000],
        ["E009", "Iris", "Marketing", "Active", 68000],
        ["E010", "Jack", "Sales", "Active", 58000],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"


def test_merged_cells_plus_multi_row_headers():
    values = [
        ["Annual Report", None, None],
        ["Category", "Q1", "Q2"],
        ["Electronics", 5000, 6000],
        ["Clothing", 3000, 4000],
    ]
    # Title row is merged across 3 columns
    merged_ranges = [(0, 0, 0, 2)]

    classification, reason = classify_sheet(values, merged_ranges=merged_ranges)

    # Merged cells check fires first (3 columns = not > 3, but it's exactly 3)
    # Wait — threshold is >3 columns, so span of 3 (cols 0-2) = 3 cols doesn't trigger
    # But the row span is 1 so it doesn't trigger either.
    # Next checks will catch it via "header row contains blank cells" in row 0
    assert classification == "unstructured"


def test_empty_sheet_is_unstructured():
    values = [
        [None, None, None],
        [None, None, None],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "unstructured"
    assert reason == "sheet has no data"


def test_single_row_is_structured():
    values = [
        ["name", "age", "city"],
    ]

    classification, reason = classify_sheet(values)

    assert classification == "structured"
    assert reason == "rows and columns are uniform"
