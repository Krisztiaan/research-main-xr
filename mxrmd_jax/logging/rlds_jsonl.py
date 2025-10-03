from __future__ import annotations
import orjson, os, uuid, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional

@dataclass
class Step:
    observation: Any
    action: Any
    reward: float
    discount: float
    is_first: bool
    is_last: bool
    is_terminal: bool
    info: Optional[Dict[str, Any]] = None

@dataclass
class EpisodeHeader:
    episode_id: str
    seed: int
    env_id: str
    config: Dict[str, Any]
    start_time_unix: float

class RLDSWriter:
    """Writes RLDS-lite JSONL episodes compatible with RLDS schema fields."""
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'wb')

    def write_episode(self, seed: int, env_id: str, config: Dict[str, Any], steps: Iterable[Step]):
        eid = str(uuid.uuid4())
        header = EpisodeHeader(eid, seed, env_id, config, time.time())
        self.f.write(orjson.dumps({"episode": asdict(header)}) + b"\n")
        for st in steps:
            self.f.write(orjson.dumps({"step": asdict(st)}) + b"\n")
        self.f.flush()

    def close(self):
        self.f.close()
