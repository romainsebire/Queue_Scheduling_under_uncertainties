import json
from enum import IntEnum
from pathlib import Path


class Breaks:
    class BREAK_ID(IntEnum):
        SMALL = 1
        LONG = 2

    _FILE = Path("breaks.json")

    _DEFAULTS_DURATION = {
        BREAK_ID.SMALL: 15,
        BREAK_ID.LONG: 60,
    }

    def __init__(self):
        self._durations = self._DEFAULTS_DURATION.copy()
        self._load()

    def get_durations(self, break_id: BREAK_ID) -> int:
        return self._durations[break_id]

    def set_durations(self, break_id: BREAK_ID, minutes: int):
        self._durations[break_id] = minutes

    def save(self):
        """
        Save breaks duration to breaks.json file
        """
        with self._FILE.open("w") as f:
            json.dump(
                {k.value: v for k, v in self._durations.items()},
                f,
                indent=2
            )


    def _load(self):
        """
        Load break durations from break.json file
        """
        if not self._FILE.exists():
            return

        with self._FILE.open() as f:
            data = json.load(f)

        for key, value in data:
            self._durations[self.BREAK_ID(key)] = value