import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParsedEpisode:
    episode: Optional[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    raw_text: str


def _extract_json_object(raw_text: str) -> Optional[str]:
    try:
        json.loads(raw_text)
        return raw_text
    except json.JSONDecodeError:
        pass

    matches = list(re.finditer(r"\{.*\}", raw_text, flags=re.DOTALL))
    if not matches:
        return None
    # Pick the longest JSON-looking block.
    match = max(matches, key=lambda m: len(m.group(0)))
    return match.group(0)


def _coerce_int(value, field_name: str, warnings: List[str]) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        warnings.append(f"{field_name} is not an integer: {value}")
        return None


def _validate_segments(segments: List[Dict[str, Any]], max_frame: int, warnings: List[str]) -> List[Dict[str, Any]]:
    valid_segments: List[Dict[str, Any]] = []
    prev_end = 1
    for seg in segments:
        start = _coerce_int(seg.get("start_frame"), "start_frame", warnings)
        end = _coerce_int(seg.get("end_frame"), "end_frame", warnings)
        if start is None or end is None:
            warnings.append("Skipping segment with invalid frame indices.")
            continue
        if start < 1:
            warnings.append(f"start_frame clipped to 1 from {start}")
            start = 1
        if end < start:
            warnings.append(f"end_frame raised to start_frame for segment ({start}, {end}).")
            end = start
        if end > max_frame:
            warnings.append(f"end_frame clipped to {max_frame} from {end}")
            end = max_frame
        if start < prev_end:
            warnings.append(f"start_frame raised to maintain monotonic order ({start} -> {prev_end}).")
            start = prev_end
        valid_segments.append(
            {
                "start_frame": start,
                "end_frame": end,
                "action": seg.get("action", ""),
                "visual_state": seg.get("visual_state", ""),
                "memory_context": seg.get("memory_context", ""),
            }
        )
        prev_end = end
    return valid_segments


def validate_episode_payload(
    episode_id: str,
    max_frame_count: int,
    payload: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    if not isinstance(payload, dict):
        return None, warnings, ["Payload is not a JSON object."]

    result = dict(payload)
    result["episode_id"] = episode_id

    skill_score = payload.get("skill_score")
    skill_score_int = _coerce_int(skill_score, "skill_score", warnings)
    if skill_score_int is None or not (1 <= skill_score_int <= 3):
        warnings.append("skill_score missing or out of range; setting to null.")
        result["skill_score"] = None
    else:
        result["skill_score"] = skill_score_int

    segments_raw = payload.get("segments") or []
    if not isinstance(segments_raw, list):
        warnings.append("segments is not a list; replacing with empty list.")
        segments_raw = []

    result["segments"] = _validate_segments(segments_raw, max_frame_count, warnings)

    if not result["segments"]:
        warnings.append("No valid segments found after validation.")

    summary = payload.get("overall_summary", "")
    if not summary:
        warnings.append("overall_summary missing; inserting empty string.")
    result["overall_summary"] = summary
    result["skill_comment"] = payload.get("skill_comment", "")

    return result, warnings, errors


def parse_gemini_response(
    episode_id: str,
    max_frame_count: int,
    raw_text: str,
) -> ParsedEpisode:
    warnings: List[str] = []
    errors: List[str] = []

    json_block = _extract_json_object(raw_text)
    if not json_block:
        errors.append("No JSON object found in model response.")
        return ParsedEpisode(None, warnings, errors, raw_text)

    try:
        payload = json.loads(json_block)
    except json.JSONDecodeError as exc:
        errors.append(f"Failed to parse JSON: {exc}")
        return ParsedEpisode(None, warnings, errors, raw_text)

    episode, payload_warnings, payload_errors = validate_episode_payload(
        episode_id=episode_id,
        max_frame_count=max_frame_count,
        payload=payload,
    )
    warnings.extend(payload_warnings)
    errors.extend(payload_errors)
    return ParsedEpisode(episode, warnings, errors, raw_text)
