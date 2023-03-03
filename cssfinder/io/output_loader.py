# Copyright 2023 Krzysztof Wiśniewski <argmaster.world@gmail.com>
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the “Software”), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Contains class for loading output of Gilbert algorithm."""

from __future__ import annotations

import pandas as pd

from cssfinder.cssfproject import Task


class GilbertOutputLoader:
    """Loader of Gilbert output files."""

    def load_corrections(self, task: Task) -> pd.DataFrame:
        """Load corrections saved from executed task.

        DataFrame object will have 3 columns: "iteration", "index" and "value".
        """
        data_frame = pd.read_json(task.output / "corrections.json")
        data_frame.rename(  # pylint: disable=no-member
            columns={0: "iteration", 1: "index", 2: "value"}, inplace=True
        )

        return data_frame
