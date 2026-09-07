# qrobustness (Python)

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Python package that mirrors the MATLAB `+qrobustness` toolbox.

Install:

```bash
pip install -e ".[dev]"
pytest
```

The shared API contract is in [`../docs/api.md`](../docs/api.md). Selectable
margin solvers, with Algorithm 1 remaining the default, are described in
[`../docs/margin-solvers-notes.md`](../docs/margin-solvers-notes.md).
