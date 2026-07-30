# qrobustness (Python)

> SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>\
> SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>\
> SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
> SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>\
> SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>\
>
> SPDX-License-Identifier: AGPL-3.0-or-later

Python package mirroring the MATLAB `+qrobustness` toolbox.

Install:

```bash
pip install -e ".[dev]"
pytest
```

See [`../docs/api.md`](../docs/api.md) for the shared API contract and
[`../docs/margin-solvers-notes.md`](../docs/margin-solvers-notes.md) for
selectable margin solvers (default remains Algorithm 1).
