# SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
# SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
# SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
# SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
# SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The paper must verify the coefficients it actually includes."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_paper_consistency import parse_corr_from_main, parse_corr_tex  # noqa: E402


@pytest.mark.parametrize("included", [False, True])
def test_inline_and_included_tables(tmp_path, included):
    original = ROOT / "results/lipschitz-margin-python/correlations_0.999.tex"
    expected = parse_corr_tex(original)
    table = tmp_path / "correlations_0.999.tex"
    table.write_text(original.read_text())
    body = r"\input{correlations_0.999.tex}" if included else table.read_text()
    main = tmp_path / "main.tex"
    main.write_text(
        "\\begin{table}\n\\label{tab:correlations}\n" + body + "\n\\end{table}\n"
    )
    np.testing.assert_array_equal(parse_corr_from_main(main), expected)
    if included:
        table.write_text(table.read_text().replace("$1.00$", "$0.99$", 1))
        assert not np.array_equal(parse_corr_from_main(main), expected)


def test_missing_included_table_is_an_error(tmp_path):
    main = tmp_path / "main.tex"
    main.write_text(
        "\\begin{table}\n\\label{tab:correlations}\n"
        "\\input{missing.tex}\n\\end{table}\n"
    )
    with pytest.raises(FileNotFoundError):
        parse_corr_from_main(main)


def test_coefficients_outside_the_table_do_not_satisfy_the_check(tmp_path):
    original = ROOT / "results/lipschitz-margin-python/correlations_0.999.tex"
    main = tmp_path / "main.tex"
    main.write_text(
        "\\begin{table}\n\\label{tab:correlations}\n\\end{table}\n"
        + original.read_text()
    )
    with pytest.raises(ValueError, match="size"):
        parse_corr_from_main(main)
