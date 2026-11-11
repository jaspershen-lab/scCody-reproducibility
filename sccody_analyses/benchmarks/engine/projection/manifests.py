from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

def normalize_age_key(value: object) -> float:
    """Return the numeric identity used for target-age manifest keys."""
    return float(value)

@dataclass(frozen=True)
class SourceAgeSelection:
    target_age: float
    source_ages: tuple[float, ...]

@dataclass(frozen=True)
class RandomSourceAgeManifest:
    """Target-age to source-age mapping used by fixed random rollouts."""
    selections: Mapping[float, tuple[float, ...]]

    @classmethod
    def load(cls, path: str | Path | None) -> RandomSourceAgeManifest | None:
        if not path:
            return None
        payload = json.loads(Path(path).read_text())
        return cls({float(target): tuple(sorted({float(age) for age in ages}))
                    for target, ages in payload.items()})

    def as_dict(self) -> dict[float, list[float]]:
        return {target: list(ages) for target, ages in self.selections.items()}

    def resolve(self, target_age: float, eligible_ages: Iterable[float], *, allow_missing: bool=False) -> SourceAgeSelection | None:
        target = normalize_age_key(target_age)
        selected = self.selections.get(target)
        if selected is None:
            if allow_missing:
                return None
            raise KeyError(f'Source-age manifest has no entry for target age {target:g}')
        eligible = {normalize_age_key(age) for age in eligible_ages}
        if allow_missing:
            selected = tuple((age for age in selected if age in eligible))
        invalid = [age for age in selected if age not in eligible]
        if invalid or (not selected and (not allow_missing)):
            raise ValueError(f'Fixed source ages {list(selected)} are not all eligible ages for target age {target:g}')
        return SourceAgeSelection(target, selected)

def load_matched_source_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open() as handle:
        payload = json.load(handle)
    match_by, groups = payload['match_by'], payload['groups']
    seen: set[str] = set()
    for group in groups:
        group_id = str(group.get('id', ''))
        if not group_id or group_id in seen:
            raise ValueError(f'Invalid or duplicate matched-source group id {group_id!r}')
        seen.add(group_id)
        if set(group.get('match', {})) != set(match_by):
            raise ValueError(f'Group {group_id!r} does not define every match_by key')
        direction = str(group.get('source_direction', payload.get('source_direction', 'earlier')))
        if direction not in {'earlier', 'bidirectional'}:
            raise ValueError(f'Group {group_id!r} has invalid source_direction {direction!r}')
        allow_same_age_source = bool(group.get('allow_same_age_source', payload.get('allow_same_age_source', False)))
        if allow_same_age_source and direction != 'bidirectional':
            raise ValueError(f"Group {group_id!r} can allow a same-age source only when source_direction='bidirectional'")
        target_age = normalize_age_key(group.get('target_age'))
        source_ages = [normalize_age_key(age) for age in group.get('source_ages', [])]
        if any((age == target_age for age in source_ages)) and (not allow_same_age_source):
            raise ValueError(f'Group {group_id!r} includes its target age as a source')
        if direction == 'earlier' and any((age >= target_age for age in source_ages)):
            raise ValueError(f'Group {group_id!r} includes a non-earlier source age')
        group['source_direction'] = direction
        group['allow_same_age_source'] = allow_same_age_source
    return payload
