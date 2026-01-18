# src/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class PlanState:
    excluded_poi_ids: Set[str] = field(default_factory=set)
    pinned_poi_ids: Set[str] = field(default_factory=set)

    def exclude(self, poi_id: str) -> None:
        self.excluded_poi_ids.add(poi_id)
        if poi_id in self.pinned_poi_ids:
            self.pinned_poi_ids.remove(poi_id)

    def reset(self) -> None:
        self.excluded_poi_ids.clear()
        self.pinned_poi_ids.clear()
