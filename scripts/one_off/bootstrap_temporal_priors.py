"""Bootstrap temporal priors from existing supra session and profile files.

Purpose:
    One-time migration that seeds temporal_priors.yaml by converting existing
    session files (poetic names) and temporally-named profiles into temporal
    prior observations. Also renames legacy session files from poetic names
    to the new deterministic {segment}-{date}.yaml format.

Lifetime: temporary (one-off migration, safe to delete after running)
Stage: supra infrastructure

Sources migrated:
    Sessions (poetic names):
        - swift-branching-isle.yaml  (last_attuned 2026-03-08T10:00:00Z -> sunday-morning)
        - azure-listening-tide.yaml  (last_attuned 2026-03-08T10:30:00Z -> sunday-morning)
        - mossy-spreading-leaf.yaml  (last_attuned 2026-03-08T12:00:00Z -> sunday-afternoon)

    Profiles (temporal names only):
        - creative-evening.yaml      (saved_at 2026-03-06T17:27:22Z    -> friday-evening)
        - sunday-wrapup.yaml         (saved_at 2026-03-08T10:00:00Z    -> sunday-morning)
        - friday-evening.yaml        (saved_at 2026-03-13T18:00:00Z    -> friday-evening)

    Non-temporal profiles skipped:
        - deep-research.yaml, ship-it.yaml, stage2-unet-setup.yaml, training.yaml
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add hooks directory to path so supra_reader is importable
REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

import supra_reader  # noqa: E402

SESSIONS_DIR = REPO_ROOT / ".claude" / "supra" / "sessions"
PROFILES_DIR = REPO_ROOT / ".claude" / "supra" / "profiles"

# Profiles with temporal names that map to segment keys.
# Others (deep-research, ship-it, stage2-unet-setup, training) are task-driven,
# not time-driven — do not migrate.
TEMPORAL_PROFILES = {
    "creative-evening",
    "friday-evening",
    "sunday-wrapup",
}


def segment_from_timestamp(ts_str: str) -> tuple[str, str]:
    """Return (segment_key, date_str) for an ISO 8601 timestamp string.

    Converts to CET/CEST (UTC+1 in winter, UTC+2 in summer).
    March 2026 dates are still winter time (CET = UTC+1).
    """
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is not None:
        # Winter: UTC+1 (DST starts late March 2026)
        month = dt.month
        # Rough DST: April–October = UTC+2, rest = UTC+1
        offset = 2 if 4 <= month <= 10 else 1
        dt = dt.astimezone(timezone(timedelta(hours=offset)))

    days = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday"]
    day = days[dt.weekday()]
    hour = dt.hour

    if 6 <= hour < 12:
        bucket = "morning"
    elif 12 <= hour < 17:
        bucket = "afternoon"
    elif 17 <= hour < 22:
        bucket = "evening"
    else:
        bucket = "night"

    # For night bucket after midnight, back-date to previous day
    date_str = dt.date().isoformat()
    if hour < 6:
        date_str = (dt.date() - timedelta(days=1)).isoformat()

    return f"{day}-{bucket}", date_str


def migrate_sessions() -> list[dict]:
    """Process all poetic-named session files, record observations, rename files.

    Returns list of migration result dicts.
    """
    results = []
    if not SESSIONS_DIR.is_dir():
        print(f"  Sessions dir not found: {SESSIONS_DIR}")
        return results

    yaml = supra_reader.yaml
    if yaml is None:
        print("  ERROR: PyYAML not available")
        return results

    for session_file in sorted(SESSIONS_DIR.glob("*.yaml")):
        stem = session_file.stem

        # Skip files that are already in the new deterministic format
        # (e.g. friday-evening-2026-03-13.yaml has 3 hyphen groups ending in a date)
        parts = stem.rsplit("-", 3)
        if len(parts) >= 2 and len(parts[-1]) == 4 and parts[-1].isdigit():
            print(f"  Skipping already-migrated session: {stem}.yaml")
            continue

        data = supra_reader._read_yaml(session_file)
        if not data:
            print(f"  WARNING: Could not read or empty: {session_file.name}")
            continue

        ts = data.get("last_attuned")
        if not ts:
            print(f"  WARNING: No last_attuned in {session_file.name}, skipping")
            continue

        segment, date_str = segment_from_timestamp(str(ts))
        new_name = f"{segment}-{date_str}"
        new_path = SESSIONS_DIR / f"{new_name}.yaml"

        # Record temporal observation using the timestamp from the file
        # We pass the timestamp as a fake "now" by temporarily calling record directly
        ok = supra_reader.record_temporal_observation(data, segment=segment)
        status = "recorded" if ok else "FAILED to record"

        # Rename: merge coordinators list if target already exists
        if new_path.exists() and new_path != session_file:
            existing = supra_reader._read_yaml(new_path)
            # Merge coordinators lists
            existing_coords = set(existing.get("coordinators", []))
            new_coords = set(data.get("coordinators", [stem]))
            if not new_coords:
                new_coords = {stem}
            merged_coords = sorted(existing_coords | new_coords)
            existing["coordinators"] = merged_coords
            # Keep more recent last_attuned
            existing_ts = existing.get("last_attuned", "")
            if str(ts) > str(existing_ts):
                existing["last_attuned"] = ts
            import os
            tmp = new_path.with_suffix(".yaml.tmp")
            tmp.write_text(
                yaml.dump(existing, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            os.replace(str(tmp), str(new_path))
            session_file.unlink()
            action = f"merged into existing {new_name}.yaml (coordinators: {merged_coords})"
        elif new_path != session_file:
            # Build new-format session file
            import os
            new_data = {
                "supra_session_id": new_name,
                "temporal_segment": segment,
                "date": date_str,
                "created_at": ts,
                "last_attuned": ts,
                "last_attuned_by": data.get("last_attuned_by", stem),
                "coordinators": data.get("coordinators", [stem]),
                "mode": data.get("mode", "focused"),
                "dimensions": data.get("dimensions", {}),
                "focus": data.get("focus", []),
                "suppress": data.get("suppress", []),
            }
            # Ensure coordinators is a list
            if isinstance(new_data["coordinators"], str):
                new_data["coordinators"] = [new_data["coordinators"]]
            if not new_data["coordinators"]:
                new_data["coordinators"] = [stem]

            tmp = new_path.with_suffix(".yaml.tmp")
            tmp.write_text(
                yaml.dump(new_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            os.replace(str(tmp), str(new_path))
            session_file.unlink()
            action = f"renamed to {new_name}.yaml"
        else:
            action = "already has target name (no rename)"

        results.append({
            "source": session_file.name,
            "segment": segment,
            "timestamp": ts,
            "new_name": new_name,
            "observation_status": status,
            "action": action,
        })

    return results


def migrate_profiles() -> list[dict]:
    """Record temporal observations from profiles with temporal names.

    Profiles are NOT renamed (they serve a different purpose).
    Returns list of migration result dicts.
    """
    results = []
    if not PROFILES_DIR.is_dir():
        print(f"  Profiles dir not found: {PROFILES_DIR}")
        return results

    for profile_name in TEMPORAL_PROFILES:
        profile_path = PROFILES_DIR / f"{profile_name}.yaml"
        if not profile_path.exists():
            print(f"  Profile not found, skipping: {profile_name}.yaml")
            continue

        data = supra_reader._read_yaml(profile_path)
        if not data:
            print(f"  WARNING: Could not read or empty: {profile_name}.yaml")
            continue

        ts = data.get("saved_at")
        if not ts:
            print(f"  WARNING: No saved_at in {profile_name}.yaml, skipping")
            continue

        segment, date_str = segment_from_timestamp(str(ts))
        ok = supra_reader.record_temporal_observation(data, segment=segment)
        status = "recorded" if ok else "FAILED to record"

        results.append({
            "source": f"{profile_name}.yaml (profile)",
            "segment": segment,
            "timestamp": ts,
            "new_name": "(not renamed — profiles preserved)",
            "observation_status": status,
            "action": "observation recorded, file unchanged",
        })

    return results


def print_results(session_results: list[dict], profile_results: list[dict]) -> None:
    """Print a human-readable summary of the migration."""
    all_results = session_results + profile_results
    print("\n=== Bootstrap Temporal Priors: Migration Results ===\n")

    if not all_results:
        print("  Nothing to migrate.")
        return

    for r in all_results:
        print(f"  Source    : {r['source']}")
        print(f"  Timestamp : {r['timestamp']}")
        print(f"  Segment   : {r['segment']}")
        print(f"  Obs status: {r['observation_status']}")
        print(f"  Action    : {r['action']}")
        print()

    print("=== Final temporal_priors.yaml state ===\n")
    priors = supra_reader.read_temporal_priors()
    segments = priors.get("segments", {})
    if not segments:
        print("  (empty — all record calls may have failed)")
        return

    for seg_key, seg_data in sorted(segments.items()):
        obs = seg_data.get("observations", 0)
        prior = seg_data.get("prior", {})
        mode = prior.get("mode", "?")
        dims = prior.get("dimensions", {})
        min_obs = priors.get("min_observations", 2)
        ready = "READY (>= min_observations)" if obs >= min_obs else f"NOT YET ({obs}/{min_obs} obs)"
        print(f"  Segment: {seg_key}  [{ready}]")
        print(f"    Observations : {obs}")
        print(f"    Mode         : {mode}")
        print(f"    Dimensions   : {', '.join(f'{k}={v:.2f}' for k, v in dims.items())}")
        print()


def main() -> None:
    print("Bootstrap Temporal Priors — migration script")
    print(f"Repo root    : {REPO_ROOT}")
    print(f"Sessions dir : {SESSIONS_DIR}")
    print(f"Profiles dir : {PROFILES_DIR}")
    print(f"Priors file  : {supra_reader.TEMPORAL_PRIORS_PATH}")
    print()

    print("--- Migrating session files ---")
    session_results = migrate_sessions()
    print(f"  Processed {len(session_results)} session file(s).")

    print("\n--- Migrating temporal profiles ---")
    profile_results = migrate_profiles()
    print(f"  Processed {len(profile_results)} profile file(s).")

    print_results(session_results, profile_results)


if __name__ == "__main__":
    main()
