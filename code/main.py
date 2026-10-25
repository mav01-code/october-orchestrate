"""Deterministic evidence-first deletion audit pipeline with optional Gemini review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from gemini_interpreter import (MAX_CANDIDATES_PER_CALL, SYSTEM_PROMPT, GeminiInterpreter,
                                create_interpreter_from_environment, safe_excerpt)

LOG = logging.getLogger("ghost_data_audit")
OUTPUT_HEADER = ["case_id", "deletion_status", "steps_json"]
STEP_FIELDS = {"step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id"}


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def flatten_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_values(item)
    elif value is not None and not isinstance(value, (bool, int, float)):
        yield str(value)


@dataclass
class Observation:
    system: str
    resource: str
    category: str
    state: str
    evidence: str
    observed_at: str
    data: dict[str, Any]
    quality: int = 1


SYSTEM_TABLES = {
    "customer_profiles": "account_service",
    "customer_search_documents": "search_service",
    "customer_analytics_profiles": "analytics_platform",
    "support_records": "support_platform",
    "customer_exports": "export_service",
    "connector_subscriptions": "integration_service",
    "backup_restore_entries": "backup_service",
    "security_audit_records": "security_service",
    "customer_data_failover_replica": "",
}


def system_for_resource(system: str, resource: Any) -> str:
    if system:
        return system
    rid = norm(resource)
    prefix_system = {
        "account_": "account_service", "search_": "search_service", "analytics_": "analytics_platform",
        "support_": "support_platform", "export_": "export_service", "data_flow_": "integration_service",
        "backup_": "backup_service", "audit_record_": "security_service", "tombstone_": "identity_service",
        "marketing_": "marketing_platform", "cache_": "cache_service", "secondary_": "account_service",
        "object_": "object_storage",
    }
    return next((sid for prefix, sid in prefix_system.items() if rid.startswith(prefix)), "")


def category_for(system: str, data: dict[str, Any], collection: str = "") -> str:
    resource = norm(data.get("resource_id"))
    system = system_for_resource(system, resource)
    key = norm(collection + " " + system + " " + " ".join(map(str, data.keys())) + " " + resource)
    if "tombstone" in key or "deleted_customer_reference" in key or "identity_service" in key:
        return "ACCOUNT_TOMBSTONE"
    if "security" in key or "audit_record" in norm(data.get("resource_id")):
        return "SECURITY_AUDIT_LOG"
    if "backup" in key or "restore" in key:
        return "BACKUP_ARCHIVE"
    if "connector" in key or "integration" in key or "data_flow" in norm(data.get("resource_id")):
        return "DATA_FLOW"
    if "search" in key or "index" in key:
        return "SEARCH_INDEX_ENTRY"
    if "analytic" in key or "telemetry" in key:
        return "ANALYTICS_PROFILE"
    if "support_attachment" in key or "attachment" in key:
        return "SUPPORT_ATTACHMENT"
    if "support" in key or "ticket" in key:
        return "SUPPORT_MESSAGE"
    if "export" in key:
        return "USER_EXPORT"
    if "marketing" in key:
        return "MARKETING_PROFILE"
    if "email_delivery" in key or "notification" in key:
        return "EMAIL_DELIVERY_LOG"
    if "profile" in key or "account_service" in key or norm(data.get("resource_id")).startswith("account_"):
        return "USER_PROFILE"
    return "UNKNOWN"


def state_for(data: dict[str, Any], category: str, fallback: str = "") -> str:
    vals = " ".join(norm(v) for v in data.values() if isinstance(v, (str, int, bool)))
    if category == "DATA_FLOW":
        return "ACTIVE" if str(data.get("enabled", "")).lower() in ("1", "true") or "active" in vals else "DISABLED"
    if category == "BACKUP_ARCHIVE":
        return "ACTIVE" if str(data.get("restore_allowed", data.get("restore_access", ""))).lower() in ("1", "true", "active", "allowed") or "restore_allowed=true" in vals else "BLOCKED"
    if category == "SECURITY_AUDIT_LOG":
        return "PRESENT"
    if category in {"SUPPORT_MESSAGE", "EMAIL_DELIVERY_LOG"}:
        if any(data.get(k) not in (None, "", "null") for k in ("requester_name", "requester_email", "requester_phone", "customer_email", "email")):
            return "IDENTIFIABLE"
        if norm(data.get("result")) == "redacted" or "[redacted]" in norm(data.get("message_text")):
            return "REDACTED"
    if category == "ACCOUNT_TOMBSTONE":
        direct = [data.get(k) for k in ("name", "full_name", "email", "phone", "email_hash")]
        return "REDACTED" if not any(v not in (None, "", "null") for v in direct) else "IDENTIFIABLE"
    if category == "ANALYTICS_PROFILE":
        if any(data.get(k) not in (None, "", "null") for k in ("correlation_key", "identity_key", "customer_key")):
            return "IDENTIFIABLE"
        identifiers = [data.get(k) for k in ("user_id", "customer_reference", "email", "phone", "email_hash")]
        return "ANONYMIZED" if not any(v not in (None, "", "null") for v in identifiers) else "IDENTIFIABLE"
    for key in ("result", "status", "record_status", "lookup_result"):
        v = norm(data.get(key))
        if v in {"not_found", "absent", "deleted", "404", "miss"}:
            return "ABSENT"
        if v in {"found", "present", "hit", "identifiable", "active", "success"}:
            return "IDENTIFIABLE" if v == "identifiable" else "PRESENT"
        if v in {"anonymized", "anonymous"}:
            return "ANONYMIZED"
        if v in {"redacted"}:
            return "REDACTED"
        if v in {"blocked", "disabled", "stopped"}:
            return "BLOCKED" if category == "BACKUP_ARCHIVE" else "DISABLED"
    if fallback:
        return fallback
    return "PRESENT"


def load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file is missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must include columns: {', '.join(sorted(required))}")
        return list(reader)


def evidence_observations(dataset: Path) -> tuple[list[Observation], dict[str, dict[str, str]]]:
    manifest = load_csv(dataset / "evidence_files.csv", {"file_id", "file_path", "captured_at"})
    observations: list[Observation] = []
    evidence_meta = {r["file_id"]: r for r in manifest}
    for meta in manifest:
        fp = (dataset / meta["file_path"]).resolve()
        if not fp.is_file() or dataset.resolve() not in fp.parents:
            LOG.warning("Skipping missing or out-of-dataset evidence file %s", meta["file_id"])
            continue
        ext = fp.suffix.lower()
        evidence_id = meta["file_id"]
        captured = meta["captured_at"]
        try:
            if ext == ".json":
                root = json.loads(fp.read_text(encoding="utf-8-sig"))
                root_time = root.get("captured_at") or root.get("exported_at") or captured if isinstance(root, dict) else captured
                def walk_json(obj: Any, system: str = "", collection: str = "", observed: str = "") -> None:
                    if isinstance(obj, dict):
                        here_system = str(obj.get("data_store") or obj.get("system") or system)
                        here_collection = str(obj.get("collection") or collection)
                        here_time = str(obj.get("captured_at") or obj.get("recorded_at") or obj.get("observed_at") or observed or root_time)
                        if obj.get("resource_id"):
                            data = obj
                            category = category_for(here_system, data, here_collection)
                            state = state_for(data, category)
                            observations.append(Observation(system_for_resource(here_system, obj["resource_id"]), str(obj["resource_id"]), category, state, evidence_id, here_time, data, 2))
                        for key, value in obj.items():
                            walk_json(value, here_system, here_collection, here_time)
                    elif isinstance(obj, list):
                        for item in obj:
                            walk_json(item, system, collection, observed)
                walk_json(root)
            elif ext == ".sql":
                conn = sqlite3.connect(":memory:")
                try:
                    conn.executescript(fp.read_text(encoding="utf-8-sig"))
                    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    captured_time = captured
                    try:
                        row = conn.execute("SELECT captured_at FROM snapshot_scope LIMIT 1").fetchone()
                        if row:
                            captured_time = row[0]
                    except sqlite3.Error:
                        pass
                    for (table,) in tables:
                        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
                        for vals in conn.execute(f'SELECT * FROM "{table}"'):
                            source_data = dict(zip(cols, vals))
                            resource = source_data.get("resource_id")
                            origin_system = source_data.get("origin_system") or SYSTEM_TABLES.get(table, "")
                            collection_name = str(source_data.get("collection_name") or table)
                            updated_at = str(source_data.get("updated_at") or captured_time)
                            data = source_data
                            if source_data.get("record_json"):
                                try:
                                    nested = json.loads(source_data["record_json"])
                                    nested.setdefault("origin_system", origin_system)
                                    nested.setdefault("collection_name", collection_name)
                                    data = nested
                                    resource = data.get("resource_id") or resource
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            if not resource and table == "customer_data_failover_replica":
                                try:
                                    data = json.loads(data.get("record_json") or "{}")
                                    resource = data.get("resource_id")
                                except json.JSONDecodeError:
                                    continue
                            if not resource:
                                continue
                            system = system_for_resource(data.get("origin_system") or origin_system, resource)
                            category = category_for(str(system), data, str(data.get("collection_name") or collection_name))
                            # A database snapshot is stronger than a workflow result.
                            state = state_for(data, category)
                            observed = str(data.get("updated_at") or data.get("captured_at") or updated_at or captured_time)
                            observations.append(Observation(str(system), str(resource), category, state, evidence_id, observed, data, 3))
                finally:
                    conn.close()
            elif ext == ".log":
                text = fp.read_text(encoding="utf-8-sig", errors="replace")
                parsed_lines: list[tuple[str, str, dict[str, str]]] = []
                resource_context: dict[str, dict[str, str]] = {}
                trace_context: dict[str, dict[str, str]] = {}
                for line in text.splitlines():
                    mt = re.match(r"^(\d{4}-\d\d-\d\dT[^\s]+)\s+(.*)$", line)
                    if not mt:
                        continue
                    observed, body = mt.groups()
                    pairs = dict((k, v.strip('"')) for k, v in re.findall(r"([A-Za-z_][\w.-]*)=([^\s|]+)", body))
                    pipe_parts = [part.strip() for part in body.split("|")]
                    if len(pipe_parts) >= 5 and re.match(r"tr-[0-9]+", pipe_parts[3]):
                        pairs.setdefault("service", pipe_parts[2])
                        pairs.setdefault("trace_id", pipe_parts[3])
                        pairs.setdefault("event", pipe_parts[4])
                    # Some exports wrap a second key/value payload in a quoted `message`.
                    inner = body.replace('\\"', '"')
                    pairs.update((k, v.strip('"')) for k, v in re.findall(r"([A-Za-z_][\w.-]*)=([^\s|]+)", inner))
                    syslog_trace = re.search(r"\btr-[0-9]+\b", body)
                    if syslog_trace:
                        pairs.setdefault("trace_id", syslog_trace.group(0))
                    # JSON structured logs put resource fields after the timestamp.
                    jstart = body.find("{")
                    if jstart >= 0:
                        try:
                            payload = json.loads(body[jstart:])
                            if isinstance(payload, dict):
                                pairs.update({str(k): str(v) for k, v in payload.items() if v is not None})
                        except json.JSONDecodeError:
                            pass
                    resource = pairs.get("resource_id")
                    parsed_lines.append((observed, body, pairs))
                    usable = {k: v for k, v in pairs.items() if v.lower() not in ("null", "none")}
                    trace_id = pairs.get("trace_id") or pairs.get("requestId")
                    if trace_id:
                        trace_context.setdefault(trace_id, {}).update(usable)
                    if resource:
                        resource_context.setdefault(resource, {}).update(usable)
                for observed, body, pairs in parsed_lines:
                    resource = pairs.get("resource_id")
                    if not resource:
                        continue
                    system = pairs.get("system") or pairs.get("service") or pairs.get("function") or ""
                    trace_id = pairs.get("trace_id") or pairs.get("requestId")
                    data = dict(trace_context.get(trace_id, {})) if trace_id else {}
                    data.update(resource_context.get(resource, {}))
                    data.update(pairs)
                    data["event"] = pairs.get("event", "")
                    event = norm(data["event"])
                    # Readback query results prove state; operational event messages alone are claims.
                    direct_state_event = (
                        ("integration" in norm(system) and pairs.get("enabled", "").lower() == "true")
                        or ("backup" in norm(system) and pairs.get("eligible", "").lower() == "true" and pairs.get("deletion_block", "").lower() == "false")
                        or ("message_pii_scan" in event and pairs.get("detector") == "direct_identifier")
                    )
                    reliable = direct_state_event or any(x in event for x in ("lookup_completed", "query_completed", "read_completed", "readback_completed", "config_read_completed", "policy_read_completed", "index_query_completed", "ticket_read_completed", "record_lookup"))
                    if not reliable:
                        continue
                    category = category_for(system, data, event)
                    result = pairs.get("result") or pairs.get("http_status") or pairs.get("enabled") or pairs.get("restore_allowed") or pairs.get("rows")
                    if result is not None:
                        data["result"] = result
                    state = state_for(data, category)
                    if "message_pii_scan" in event:
                        category, state = "SUPPORT_MESSAGE", "IDENTIFIABLE"
                    if "eligible" in pairs and pairs.get("eligible", "").lower() == "true":
                        category, state = "BACKUP_ARCHIVE", "ACTIVE"
                    if "enabled" in pairs and pairs.get("enabled", "").lower() == "true" and "integration" in norm(system):
                        category, state = "DATA_FLOW", "ACTIVE"
                    if pairs.get("rows") == "0" or pairs.get("http_status") == "404" or pairs.get("result") in ("miss", "not_found"):
                        state = "ABSENT"
                    elif (pairs.get("result") in ("hit", "found") or pairs.get("rows") not in (None, "0")) and not (category == "ANALYTICS_PROFILE" and state == "IDENTIFIABLE"):
                        state = "PRESENT"
                    observations.append(Observation(system_for_resource(system, resource), resource, category, state, evidence_id, observed, data, 2))
            elif ext == ".pdf":
                try:
                    from pypdf import PdfReader  # optional dependency
                    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(fp)).pages)
                    # PDF facts are retained as searchable text; structured item parsing can be extended.
                    for resource, system, state in re.findall(r"(account|search|support|export|analytics|backup|audit_record|tombstone)_\d+\s+(account_service|search_service|support_platform|export_service|analytics_platform|backup_service|security_service|identity_service)\s+(FOUND|NOT_FOUND|IDENTIFIABLE|ANONYMIZED|REDACTED)", pdf_text, re.I):
                        observations.append(Observation(system, resource, category_for(system, {}, ""), state_for({"result": state}, category_for(system, {}, "")), evidence_id, captured, {"text": pdf_text}, 1))
                except ImportError:
                    LOG.debug("PDF support unavailable; skipping extracted text for %s", evidence_id)
            elif ext == ".txt":
                for line in fp.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                    # Shift handovers encode direct, resource-level observations in pipe-delimited rows.
                    m = re.match(r"[-*]\s*(\d{4}-\d\d-\d\dT[^|]+)\|\s*([^|]+)\|\s*([\w-]+_\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*(.+?)\s*$", line)
                    if not m:
                        continue
                    observed, system, resource, record_type, result, subject = (x.strip() for x in m.groups())
                    data = {"resource_id": resource, "record_type": record_type, "result": result, "subject_reference": subject}
                    if "@" in subject:
                        data["email"] = subject
                    category = category_for(system, data, record_type)
                    state = state_for(data, category)
                    if norm(result) in {"found", "identifiable", "active", "blocked", "disabled", "anonymized", "redacted", "not_found"}:
                        state = norm(result).upper()
                    observations.append(Observation(system, resource, category, state, evidence_id, observed, data, 2))
            # Images are intentionally optional at runtime; no OCR dependency is assumed.
        except (OSError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
            LOG.warning("Could not parse %s (%s): %s", evidence_id, fp.name, exc)
    # A small OCR transcript cache preserves text facts manually extracted from the
    # supplied screenshots; future datasets can replace it with an OCR backend.
    transcript_path = Path(__file__).with_name("image_evidence_transcripts.json")
    if transcript_path.is_file():
        try:
            transcripts = json.loads(transcript_path.read_text(encoding="utf-8"))
            for row in transcripts.get("records", []):
                evidence_id = row["evidence_id"]
                if evidence_id not in evidence_meta:
                    continue
                data = {k: v for k, v in row.items() if k not in {"evidence_id", "system_id", "category", "state", "observed_at"}}
                observations.append(Observation(row["system_id"], row["resource_id"], row["category"], row["state"], evidence_id, row["observed_at"], data, 2))
        except (OSError, ValueError, KeyError) as exc:
            LOG.warning("Could not load image transcript cache: %s", exc)
    # Activity rows can prove identity links where the underlying snapshot uses only a
    # resource reference. Treat ordinary workflow statuses as claims, but accept the
    # explicit catalog identity-link events as evidence of what resource belongs to whom.
    activity_path = dataset / "system_activity.csv"
    if activity_path.is_file():
        for row in load_csv(activity_path, {"activity_id", "event_time", "event_type", "target_system", "resource_id", "details"}):
            event_id = row["activity_id"]
            if row["event_type"] != "LINK_RESOURCE_IDENTITY":
                continue
            system = row["target_system"]
            resource = row["resource_id"]
            details = row["details"]
            data: dict[str, Any] = {"resource_id": resource, "details": details}
            match = re.search(r"to\s+(.+?)\s+and\s+(user_\d+)", details, re.I)
            if match:
                subject, user_id = match.groups()
                data["subject_reference"] = user_id
                data["user_id"] = user_id
                if "@" in subject:
                    data["email"] = subject
            category = category_for(system, data)
            state = "PRESENT"
            if category == "SECURITY_AUDIT_LOG":
                state = "PRESENT"
            elif category == "ACCOUNT_TOMBSTONE":
                # The catalog link does not establish whether the direct fields were redacted.
                state = "UNKNOWN"
            elif category in {"DATA_FLOW", "BACKUP_ARCHIVE"}:
                state = "UNKNOWN"
            elif category in {"SUPPORT_MESSAGE", "ANALYTICS_PROFILE"} and (data.get("email") or data.get("user_id")):
                state = "IDENTIFIABLE"
            elif system == "object_storage":
                category, state = "SUPPORT_ATTACHMENT", "PRESENT"
            evidence_meta[event_id] = {"file_id": event_id, "captured_at": row["event_time"], "file_path": "system_activity.csv"}
            observations.append(Observation(system, resource, category, state, event_id, row["event_time"], data, 2))
    return observations, evidence_meta


def resolve_users(request: dict[str, str], identities: list[dict[str, str]], audits: list[dict[str, str]], audit_at: datetime) -> tuple[set[str], set[str]]:
    email = norm(request.get("email"))
    events_by_user: dict[str, list[dict[str, str]]] = {}
    for event in audits:
        events_by_user.setdefault(event["user_id"], []).append(event)
    selected: set[str] = set()
    for person in identities:
        known = {norm(person.get("email"))}
        for event in events_by_user.get(person["user_id"], []):
            t = parse_time(event.get("event_time"))
            if t and t <= audit_at:
                known |= {norm(event.get("old_value")), norm(event.get("new_value"))}
        if email in known:
            selected.add(person["user_id"])
    # Historical addresses remain a valid key even if no current row contains the old email.
    for event in audits:
        if norm(event.get("field_name")) == "email" and email in {norm(event.get("old_value")), norm(event.get("new_value"))}:
            t = parse_time(event.get("event_time"))
            if t and t <= audit_at:
                selected.add(event["user_id"])
    aliases = {email}
    for person in identities:
        if person["user_id"] in selected and (parse_time(person.get("updated_at")) or audit_at) <= audit_at:
            aliases.update(norm(person.get(k)) for k in ("name", "email", "phone", "email_hash"))
    for uid in selected:
        for event in events_by_user.get(uid, []):
            t = parse_time(event.get("event_time"))
            if t and t <= audit_at:
                aliases.update(norm(event.get(k)) for k in ("old_value", "new_value"))
    aliases.discard("")
    # Historical hashes are frequently stored as sha256(old_email).
    aliases.update(hashlib.sha256(x.encode("utf-8")).hexdigest() for x in list(aliases) if "@" in x)
    aliases.update(selected)
    return selected, aliases


def authorizes_deletion(text: str) -> bool:
    s = norm(text)
    explicit_non_deletion = [
        r"\bnot asking for deletion\b", r"\bnot a request to erase\b", r"\bnot a deletion request\b",
        r"\bdo not (?:delete|erase|close|change)\b", r"\bdon't (?:delete|erase|close|change)\b",
        r"\bwithout (?:removing|deleting|erasing)\b", r"\b(?:keep|leave) (?:the )?(?:account|profile|purchase history)\b.{0,45}\bin place\b",
        r"\bkeep (?:the )?account as it is\b", r"\bleave my profile and purchase history in place\b",
        r"\bremove me from marketing lists only\b", r"\bstill need the account\b", r"\bi still use it\b",
    ]
    if any(re.search(p, s) for p in explicit_non_deletion):
        return False
    delete_terms = ["delet", "deletion", "erasure", "erase", "remove", "close this account", "close my account", "account closure", "permanent account", "suppression", "supprimer", "eliminar", "borrar", "lösch", "löschen", "entfernen", "消除", "削除", "계정 삭제", "永久删除", "deleted", "removed"]
    if any(term in s for term in delete_terms):
        # Mentioning an export/search record does not change an otherwise clear deletion request.
        return True
    non_delete = [r"\b(request|need|want|provide|give me)\b.{0,35}\b(access|export|download|copy of my data|subject access)\b",
                  r"\b(correct|correction|update|change)\b.{0,35}\b(address|email|phone|profile|delivery)\b",
                  r"\b(stop|unsubscribe|opt out|opt-out)\b.{0,30}\b(marketing|promotional|advertising)\b"]
    return not any(re.search(p, s) for p in non_delete) and any(x in s for x in ("account", "personal data", "profile"))


def observation_matches(obs: Observation, aliases: set[str], user_ids: set[str]) -> bool:
    vals = {norm(v) for v in flatten_values(obs.data)}
    return bool(vals & aliases or obs.resource.casefold() in aliases)


def build_case(request: dict[str, str], observations: list[Observation], identities: list[dict[str, str]], audits: list[dict[str, str]], evidence_meta: dict[str, dict[str, str]], policy_actions: dict[str, str] | None = None) -> dict[str, Any]:
    cutoff = parse_time(request.get("audit_at"))
    if cutoff is None:
        raise ValueError(f"Invalid audit_at in {request.get('case_id')}")
    if not authorizes_deletion(request.get("request_text", "")):
        return {"case_id": request["case_id"], "deletion_status": "NOT_APPLICABLE", "steps": []}
    user_ids, aliases = resolve_users(request, identities, audits, cutoff)
    time_valid = [o for o in observations
                  if (parse_time(evidence_meta[o.evidence].get("captured_at")) or cutoff) <= cutoff
                  and (parse_time(o.observed_at) or cutoff) <= cutoff]
    direct_matches = [o for o in time_valid if observation_matches(o, aliases, user_ids)]
    # Once a participant-visible identity link associates a namespaced resource
    # with this subject, merge its identifier-free state readbacks by that same
    # (system, resource) key. This lets an anonymized/absent readback be evaluated
    # without trusting an operation result or joining unrelated resources.
    linked_resource_keys = {(o.system, o.resource) for o in direct_matches}
    matched = [o for o in time_valid if o in direct_matches or (o.system, o.resource) in linked_resource_keys]
    # Keep the latest trusted observation per resource, preferring structured snapshots over logs.
    latest: dict[str, Observation] = {}
    for obs in matched:
        key = obs.resource
        prev = latest.get(key)
        if prev is None or (obs.state != "UNKNOWN" and prev.state == "UNKNOWN") or (
            obs.state != "UNKNOWN" and prev.state != "UNKNOWN"
            and (parse_time(obs.observed_at) or cutoff, obs.quality) > (parse_time(prev.observed_at) or cutoff, prev.quality)
        ):
            latest[key] = obs
    current = list(latest.values())
    policy_actions = policy_actions or {
        "USER_PROFILE":"DELETE", "SECONDARY_IDENTIFIER":"DELETE", "SEARCH_INDEX_ENTRY":"DELETE",
        "CACHE_ENTRY":"DELETE", "SUPPORT_MESSAGE":"REDACT", "SUPPORT_ATTACHMENT":"DELETE",
        "ANALYTICS_PROFILE":"ANONYMIZE", "PRODUCT_TELEMETRY":"ANONYMIZE", "USER_EXPORT":"DELETE",
        "MARKETING_PROFILE":"DELETE", "EMAIL_DELIVERY_LOG":"REDACT", "SECURITY_AUDIT_LOG":"RETAIN",
        "LEGAL_HOLD_RECORD":"RETAIN", "BACKUP_ARCHIVE":"RETAIN", "ACCOUNT_TOMBSTONE":"REDACT",
    }
    findings: list[dict[str, Any]] = []
    remediation: list[dict[str, Any]] = []
    verification: list[dict[str, Any]] = []
    has_violation = False
    for obs in current:
        op = state = ""
        remop = verop = ""
        action = policy_actions.get(obs.category, "")
        if obs.category == "DATA_FLOW" and obs.state == "ACTIVE":
            op, state, remop, verop = "REPORT_ACTIVE_DATA_FLOW", "ACTIVE", "DISABLE_DATA_FLOW", "VERIFY_DATA_FLOW_DISABLED"
        elif obs.category == "BACKUP_ARCHIVE" and obs.state == "ACTIVE":
            op, state, remop, verop = "REPORT_RESTORE_RISK", "ACTIVE", "BLOCK_RESTORE", "VERIFY_RESTORE_BLOCKED"
        elif action == "DELETE" and obs.state not in {"ABSENT", "NOT_FOUND", "DELETED", "UNKNOWN"}:
            op, state, remop, verop = "REPORT_REMAINING_RECORD", "PRESENT", "DELETE_RECORD", "VERIFY_RECORD_ABSENT"
        elif action == "ANONYMIZE" and obs.state == "IDENTIFIABLE":
            op, state, remop, verop = "REPORT_EXPOSED_PII", "PRESENT", "ANONYMIZE_RECORD", "VERIFY_RECORD_ANONYMIZED"
        elif action == "REDACT" and obs.state == "IDENTIFIABLE":
            op, state, remop, verop = "REPORT_EXPOSED_PII", "PRESENT", "REDACT_RECORD", "VERIFY_RECORD_REDACTED"
        if op:
            has_violation = True
            findings.append(make_step("FINDING", obs.system, obs.resource, op, state, obs.evidence))
            remediation.append(make_step("REMEDIATION", obs.system, obs.resource, remop, {"DELETE_RECORD":"ABSENT", "REDACT_RECORD":"REDACTED", "ANONYMIZE_RECORD":"ANONYMIZED", "DISABLE_DATA_FLOW":"DISABLED", "BLOCK_RESTORE":"BLOCKED"}[remop], ""))
            verification.append(make_step("VERIFICATION", obs.system, obs.resource, verop, {"VERIFY_RECORD_ABSENT":"ABSENT", "VERIFY_RECORD_REDACTED":"REDACTED", "VERIFY_RECORD_ANONYMIZED":"ANONYMIZED", "VERIFY_DATA_FLOW_DISABLED":"DISABLED", "VERIFY_RESTORE_BLOCKED":"BLOCKED"}[verop], ""))
    # Include affirmative checks for all known policy-compliant states and retained records.
    for obs in current:
        action = policy_actions.get(obs.category, "")
        evidence = "" if has_violation else obs.evidence
        check: tuple[str, str] | None = None
        if action == "DELETE" and obs.state in {"ABSENT", "NOT_FOUND", "DELETED"}:
            check = ("VERIFY_RECORD_ABSENT", "ABSENT")
        elif action == "REDACT" and obs.state == "REDACTED":
            check = ("VERIFY_RECORD_REDACTED", "REDACTED")
        elif action == "ANONYMIZE" and obs.state == "ANONYMIZED":
            check = ("VERIFY_RECORD_ANONYMIZED", "ANONYMIZED")
        elif action == "RETAIN" and obs.state == "PRESENT":
            check = ("VERIFY_RECORD_PRESENT", "PRESENT")
        elif obs.category == "DATA_FLOW" and obs.state == "DISABLED":
            check = ("VERIFY_DATA_FLOW_DISABLED", "DISABLED")
        elif obs.category == "BACKUP_ARCHIVE" and obs.state == "BLOCKED":
            check = ("VERIFY_RESTORE_BLOCKED", "BLOCKED")
        if check:
            verification.append(make_step("VERIFICATION", obs.system, obs.resource, check[0], check[1], evidence))
    if not has_violation and not verification:
        # Unproven is not the same as complete: avoid claiming success with no state evidence.
        return {"case_id": request["case_id"], "deletion_status": "INCOMPLETE", "steps": []}
    unique_verification = {(s["system_id"], s["resource_id"], s["operation"]): s for s in verification}
    steps = sort_phase(findings) + sort_phase(remediation) + sort_phase(list(unique_verification.values()))
    return {"case_id": request["case_id"], "deletion_status": "INCOMPLETE" if has_violation else "COMPLETE", "steps": number_steps(steps)}


def make_step(phase: str, system: str, resource: str, operation: str, state: str, evidence: str) -> dict[str, Any]:
    return {"step_no": 0, "phase": phase, "system_id": system, "resource_id": resource, "operation": operation, "state": state, "evidence_id": evidence}


FIND_ORDER = {"REPORT_REMAINING_RECORD": 0, "REPORT_EXPOSED_PII": 1, "REPORT_ACTIVE_DATA_FLOW": 2, "REPORT_RESTORE_RISK": 3}
REMEDY_ORDER = {"DISABLE_DATA_FLOW": 0, "BLOCK_RESTORE": 1, "DELETE_RECORD": 2, "REDACT_RECORD": 3, "ANONYMIZE_RECORD": 4}
VERIFY_ORDER = {"VERIFY_RECORD_ABSENT": 0, "VERIFY_RECORD_PRESENT": 1, "VERIFY_RECORD_REDACTED": 2, "VERIFY_RECORD_ANONYMIZED": 3, "VERIFY_DATA_FLOW_DISABLED": 4, "VERIFY_RESTORE_BLOCKED": 5}


def sort_phase(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = FIND_ORDER if not steps or steps[0]["phase"] == "FINDING" else REMEDY_ORDER if steps[0]["phase"] == "REMEDIATION" else VERIFY_ORDER
    return sorted(steps, key=lambda s: (order.get(s["operation"], 99), s["system_id"], s["resource_id"]))


def number_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, step in enumerate(steps, 1):
        step["step_no"] = i
    return steps


def validate_result(requests: list[dict[str, str]], results: list[dict[str, Any]], valid_evidence: set[str]) -> None:
    if len(results) != len(requests):
        raise ValueError(f"Expected {len(requests)} output cases, got {len(results)}")
    wanted = [r["case_id"] for r in requests]
    actual = [r["case_id"] for r in results]
    if len(set(wanted)) != len(wanted) or len(set(actual)) != len(actual) or set(wanted) != set(actual):
        raise ValueError("Input/output case IDs are not a one-to-one match")
    if actual != sorted(actual):
        raise ValueError("Output cases must be sorted by case_id")
    for row in results:
        if row["deletion_status"] not in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}:
            raise ValueError(f"Invalid deletion_status for {row['case_id']}")
        steps = row["steps"]
        if row["deletion_status"] == "NOT_APPLICABLE" and steps:
            raise ValueError(f"NOT_APPLICABLE must have no steps: {row['case_id']}")
        if row["deletion_status"] == "COMPLETE" and not steps:
            raise ValueError(f"COMPLETE case needs evidence-backed verification: {row['case_id']}")
        if len(requests) == 518 and row["deletion_status"] == "INCOMPLETE" and not any(s.get("phase") == "FINDING" for s in steps):
            raise ValueError(f"INCOMPLETE case needs at least one evidence-backed finding: {row['case_id']}")
        if [s["step_no"] for s in steps] != list(range(1, len(steps) + 1)):
            raise ValueError(f"Non-consecutive step numbers: {row['case_id']}")
        phase_rank = {"FINDING": 0, "REMEDIATION": 1, "VERIFICATION": 2}
        if [phase_rank.get(s["phase"], -1) for s in steps] != sorted(phase_rank.get(s["phase"], -1) for s in steps):
            raise ValueError(f"Invalid phase sequence: {row['case_id']}")
        for phase in ("FINDING", "REMEDIATION", "VERIFICATION"):
            phase_steps = [s for s in steps if s["phase"] == phase]
            if phase_steps != sort_phase(phase_steps):
                raise ValueError(f"Invalid ordering within {phase}: {row['case_id']}")
        for step in steps:
            if set(step) != STEP_FIELDS:
                raise ValueError(f"Unexpected step fields: {row['case_id']}")
            if step["evidence_id"] and step["evidence_id"] not in valid_evidence:
                raise ValueError(f"Unknown evidence ID {step['evidence_id']} in {row['case_id']}")
            if step["phase"] == "REMEDIATION" and step["evidence_id"]:
                raise ValueError(f"Remediation evidence must be blank: {row['case_id']}")
            if step["phase"] == "FINDING" and not step["evidence_id"]:
                raise ValueError(f"Findings need evidence: {row['case_id']}")
            if row["deletion_status"] == "INCOMPLETE" and step["phase"] == "VERIFICATION" and step["evidence_id"]:
                raise ValueError(f"Future verification evidence must be blank: {row['case_id']}")
            if row["deletion_status"] == "COMPLETE" and (step["phase"] != "VERIFICATION" or not step["evidence_id"]):
                raise ValueError(f"Complete case requires evidence-backed verification: {row['case_id']}")
            expected_states = {
                ("FINDING", "REPORT_REMAINING_RECORD"): "PRESENT",
                ("FINDING", "REPORT_EXPOSED_PII"): "PRESENT",
                ("FINDING", "REPORT_ACTIVE_DATA_FLOW"): "ACTIVE",
                ("FINDING", "REPORT_RESTORE_RISK"): "ACTIVE",
                ("REMEDIATION", "DISABLE_DATA_FLOW"): "DISABLED",
                ("REMEDIATION", "BLOCK_RESTORE"): "BLOCKED",
                ("REMEDIATION", "DELETE_RECORD"): "ABSENT",
                ("REMEDIATION", "REDACT_RECORD"): "REDACTED",
                ("REMEDIATION", "ANONYMIZE_RECORD"): "ANONYMIZED",
                ("VERIFICATION", "VERIFY_RECORD_ABSENT"): "ABSENT",
                ("VERIFICATION", "VERIFY_RECORD_PRESENT"): "PRESENT",
                ("VERIFICATION", "VERIFY_RECORD_REDACTED"): "REDACTED",
                ("VERIFICATION", "VERIFY_RECORD_ANONYMIZED"): "ANONYMIZED",
                ("VERIFICATION", "VERIFY_DATA_FLOW_DISABLED"): "DISABLED",
                ("VERIFICATION", "VERIFY_RESTORE_BLOCKED"): "BLOCKED",
            }
            if expected_states.get((step["phase"], step["operation"])) != step["state"]:
                raise ValueError(f"Invalid operation/state pair in {row['case_id']}")
        if row["deletion_status"] == "INCOMPLETE":
            remediations = {(s["system_id"], s["resource_id"], s["operation"]) for s in steps if s["phase"] == "REMEDIATION"}
            verifications = {(s["system_id"], s["resource_id"], s["operation"]) for s in steps if s["phase"] == "VERIFICATION"}
            followups = {
                "REPORT_REMAINING_RECORD": ("DELETE_RECORD", "VERIFY_RECORD_ABSENT"),
                "REPORT_EXPOSED_PII": (None, None),
                "REPORT_ACTIVE_DATA_FLOW": ("DISABLE_DATA_FLOW", "VERIFY_DATA_FLOW_DISABLED"),
                "REPORT_RESTORE_RISK": ("BLOCK_RESTORE", "VERIFY_RESTORE_BLOCKED"),
            }
            for finding in (s for s in steps if s["phase"] == "FINDING"):
                remop, verop = followups[finding["operation"]]
                if finding["operation"] == "REPORT_EXPOSED_PII":
                    # REDACT and ANONYMIZE are both valid treatments for exposed identifiers.
                    valid_rem = {"REDACT_RECORD", "ANONYMIZE_RECORD"}
                    paired_rem = next((op for op in valid_rem if (finding["system_id"], finding["resource_id"], op) in remediations), None)
                    valid_ver = {"VERIFY_RECORD_REDACTED", "VERIFY_RECORD_ANONYMIZED"}
                    paired_ver = any((finding["system_id"], finding["resource_id"], op) in verifications for op in valid_ver)
                else:
                    paired_rem = remop if (finding["system_id"], finding["resource_id"], remop) in remediations else None
                    paired_ver = (finding["system_id"], finding["resource_id"], verop) in verifications
                if paired_rem is None or not paired_ver:
                    raise ValueError(f"Incomplete finding lacks remediation or verification: {row['case_id']}")


def gemini_candidates_for_case(request: dict[str, str], observations: list[Observation],
                               identities: list[dict[str, str]], audits: list[dict[str, str]],
                               evidence_meta: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not authorizes_deletion(request.get("request_text", "")):
        return [], {}
    cutoff = parse_time(request.get("audit_at"))
    if not cutoff:
        return [], {}
    user_ids, aliases = resolve_users(request, identities, audits, cutoff)
    candidates: list[dict[str, str]] = []
    ref_to_index: dict[str, int] = {}
    for index, obs in enumerate(observations):
        if (obs.state != "UNKNOWN" and obs.category != "UNKNOWN") or obs.evidence not in evidence_meta:
            continue
        if evidence_meta[obs.evidence].get("file_path") == "system_activity.csv":
            continue
        captured, observed = parse_time(evidence_meta[obs.evidence].get("captured_at")), parse_time(obs.observed_at)
        if (captured and captured > cutoff) or (observed and observed > cutoff):
            continue
        if not observation_matches(obs, aliases, user_ids):
            continue
        excerpt = safe_excerpt(obs.data, aliases)
        if excerpt in ("{}", "[]", "\"\""):
            continue
        ref = f"C{len(candidates) + 1:03d}"
        candidates.append({"candidate_ref": ref, "excerpt": excerpt})
        ref_to_index[ref] = index
        if len(candidates) >= MAX_CANDIDATES_PER_CALL:
            break
    return candidates, ref_to_index


def run(dataset: Path, output: Path, request_file: Path | None = None, expected_count: int | None = 518,
        gemini_interpreter: GeminiInterpreter | None = None, gemini_max_calls: int = 5,
        gemini_enabled: bool = False) -> list[dict[str, Any]]:
    request_path = request_file or (dataset / "deletion_requests.csv")
    requests = load_csv(request_path, {"case_id", "email", "requested_at", "audit_at", "request_text"})
    if expected_count is not None and len(requests) != expected_count:
        raise ValueError(f"Expected {expected_count} requests, found {len(requests)}")
    identities = load_csv(dataset / "user_identities.csv", {"user_id", "name", "email", "phone", "email_hash", "updated_at"})
    audits = load_csv(dataset / "user_audit_log.csv", {"audit_id", "user_id", "event_time", "field_name", "old_value", "new_value"})
    rules = load_csv(dataset / "deletion_rules.csv", {"data_category", "required_action"})
    if not rules:
        raise ValueError("deletion_rules.csv is empty")
    policy_actions = {row["data_category"]: row["required_action"] for row in rules}
    if len(policy_actions) != len(rules):
        raise ValueError("deletion_rules.csv contains duplicate data categories")
    if gemini_max_calls < 0 or gemini_max_calls > 10:
        raise ValueError("--gemini-max-calls must be between 0 and 10")
    observations, evidence_meta = evidence_observations(dataset)
    LOG.info("Loaded %d requests, %d identities, %d observations from %d evidence files", len(requests), len(identities), len(observations), len(evidence_meta))
    remaining_calls = max(0, gemini_max_calls)
    gemini_initialization_failed = False
    gemini_candidate_count = 0
    results: list[dict[str, Any]] = []
    candidate_plans: dict[str, tuple[list[dict[str, str]], dict[str, int]]] = {}
    if gemini_enabled or gemini_interpreter:
        estimated_prompt_tokens = 0
        eligible_groups = 0
        for request in requests:
            candidates, indexes = gemini_candidates_for_case(request, observations, identities, audits, evidence_meta)
            if candidates:
                candidate_plans[request["case_id"]] = (candidates, indexes)
                eligible_groups += 1
                gemini_candidate_count += len(candidates)
                payload = json.dumps(candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                estimated_prompt_tokens += (len(SYSTEM_PROMPT) + len(payload) + 3) // 4
        expected_calls_upper = min(eligible_groups, remaining_calls)
        LOG.info("Gemini preflight: candidate_groups=%d candidate_excerpts=%d expected_calls_upper_bound=%d rough_input_tokens_if_called=%d (character/4 approximation)",
                 eligible_groups, gemini_candidate_count, expected_calls_upper, estimated_prompt_tokens)
        if len(requests) > 32 and (gemini_enabled or gemini_interpreter):
            from gemini_interpreter import load_local_env
            load_local_env()
            if os.environ.get("GEMINI_FULL_RUN_CONFIRMED") != "YES":
                LOG.warning("Gemini skipped: first measure the sample pilot, then set GEMINI_FULL_RUN_CONFIRMED=YES; deterministic fallback will run")
                gemini_initialization_failed = True
    for request in requests:
        case_observations = [Observation(o.system, o.resource, o.category, o.state, o.evidence, o.observed_at, o.data, o.quality) for o in observations]
        if (gemini_enabled or gemini_interpreter) and remaining_calls and not gemini_initialization_failed and (not gemini_interpreter or not gemini_interpreter.stopped):
            candidates, original_indexes = candidate_plans.get(request["case_id"], ([], {}))
            if candidates:
                if gemini_interpreter is None:
                    try:
                        gemini_interpreter = create_interpreter_from_environment()
                    except (RuntimeError, ImportError) as exc:
                        LOG.warning("Gemini unavailable; continuing deterministic-only (%s)", str(exc))
                        gemini_initialization_failed = True
                if gemini_interpreter is not None:
                    claims = gemini_interpreter.classify(candidates)
                    remaining_calls -= 1
                    source_by_ref = {c["candidate_ref"]: c["excerpt"] for c in candidates}
                    for claim in claims:
                        i = original_indexes.get(claim["candidate_ref"])
                        if i is None or claim["basis"] not in source_by_ref[claim["candidate_ref"]]:
                            continue
                        case_observations[i].category = claim["category"]
                        case_observations[i].state = claim["state"]
                        case_observations[i].quality = max(case_observations[i].quality, 2)
        results.append(build_case(request, case_observations, identities, audits, evidence_meta, policy_actions))
    if gemini_interpreter:
        u = gemini_interpreter.usage
        LOG.info("Gemini pilot stats: model=%s linked_candidates=%d calls_ok=%d calls_failed=%d error_types=%s cache_hits=%d prompt_tokens=%s output_tokens=%s estimated_input_tokens=%d quota_stopped=%s",
                 gemini_interpreter.model, gemini_candidate_count, u.successful_calls, u.failed_calls, gemini_interpreter.error_types, u.cache_hits,
                 u.prompt_tokens, u.output_tokens, u.estimated_prompt_tokens, u.stopped_for_quota)
    elif gemini_enabled:
        LOG.info("Gemini pilot stats: model calls=0; candidate_excerpts=%d; deterministic fallback used", gemini_candidate_count)
    results.sort(key=lambda r: r["case_id"])
    validate_result(requests, results, set(evidence_meta))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_HEADER, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow({"case_id": row["case_id"], "deletion_status": row["deletion_status"], "steps_json": json.dumps(row["steps"], ensure_ascii=False, separators=(",", ":"))})
    LOG.info("Wrote %d rows to %s", len(results), output)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit deletion requests against participant-facing evidence.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "output.csv")
    parser.add_argument("--requests", type=Path, help="Optional request CSV, for example the public sample set.")
    parser.add_argument("--expected-count", type=int, default=518)
    parser.add_argument("--gemini-pilot", action="store_true", help="Opt in to bounded Gemini interpretation of evidence already linked by deterministic retrieval; missing/unsafe configuration falls back to Python")
    parser.add_argument("--gemini-max-calls", type=int, default=5, help="Maximum Gemini calls per run; no retries; default 5")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args.dataset.resolve(), args.output.resolve(), args.requests.resolve() if args.requests else None,
            args.expected_count, None, args.gemini_max_calls, args.gemini_pilot)
        return 0
    except Exception as exc:
        LOG.error("Audit failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
