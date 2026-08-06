import os
import json
import time
import re
import hashlib
import sqlite3
import threading
import traceback
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from openai import OpenAI


# ============================================================
# Environment / App Setup
# ============================================================

load_dotenv()

app = FastAPI(title="WhatsApp Real Estate Bot")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Keep all secrets in Render Environment Variables.
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
META_TOKEN = os.getenv("META_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
GRAPH_VERSION = os.getenv("GRAPH_API_VERSION", "v22.0")

# Put your Google Sheet CSV export URL in Render env as GOOGLE_SHEET_URL.
# Example format:
# https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/export?format=csv&gid=YOUR_GID
GOOGLE_SHEET_URL = os.getenv("GOOGLE_SHEET_URL", "")

JSON_FILE = os.getenv("JSON_FILE", "knowledge.json")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

# Google Places API key for verified nearby amenities.
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

# Nearby amenities settings.
GOOGLE_PLACES_RADIUS_DEFAULT = int(os.getenv("GOOGLE_PLACES_RADIUS_DEFAULT", "1500"))
GOOGLE_PLACES_CACHE_TTL_SECONDS = int(os.getenv("GOOGLE_PLACES_CACHE_TTL_SECONDS", "86400"))
MAX_NEARBY_PLACES_PER_TYPE = int(os.getenv("MAX_NEARBY_PLACES_PER_TYPE", "3"))

# Responses API tool settings.
MAX_TOOL_LISTINGS_TO_RETURN = int(os.getenv("MAX_TOOL_LISTINGS_TO_RETURN", "6"))

# Keep false first. Your deterministic formatter is safer for production.
ENABLE_RESPONSES_CONSULTANT = os.getenv("ENABLE_RESPONSES_CONSULTANT", "false").lower() == "true"

# Maximum number of matching properties shown in a single WhatsApp reply.
# Keeps replies readable instead of dumping the whole sheet on the client.
MAX_PROPERTIES_TO_SHOW = int(os.getenv("MAX_PROPERTIES_TO_SHOW", "4"))

# Session memory timeout.
SESSION_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT_SECONDS", "900"))  # 15 minutes

# Keep the chat history bounded so token usage does not grow forever.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "16"))

# Cache Google Sheet data to avoid slow loading on every message.
PROPERTY_CACHE_TTL_SECONDS = int(os.getenv("PROPERTY_CACHE_TTL_SECONDS", "300"))  # 5 minutes

# Cache knowledge.json.
KNOWLEDGE_CACHE_TTL_SECONDS = int(os.getenv("KNOWLEDGE_CACHE_TTL_SECONDS", "300"))

# Store processed webhook message IDs to avoid Meta retry duplicates.
MESSAGE_ID_TTL_SECONDS = int(os.getenv("MESSAGE_ID_TTL_SECONDS", "86400"))  # 24 hours

# File used to persist "already handled" message IDs across restarts (see
# REACTIVE-ONLY GUARANTEE below). A plain in-memory dict would forget
# everything on every restart/wake-up, which is exactly the failure mode
# this exists to close.
DEDUP_DB_PATH = os.getenv("DEDUP_DB_PATH", "message_dedup.db")

# Reactive-only safeguard: never reply to a webhook message older than this.
# Meta retries/queues undelivered webhooks for a period, so if the server
# was asleep or restarting, a burst of old events can arrive all at once
# right after it wakes up. Any message whose own WhatsApp timestamp is
# older than this is dropped with no reply, no matter what else happens.
MAX_MESSAGE_AGE_SECONDS = int(os.getenv("MAX_MESSAGE_AGE_SECONDS", "600"))  # 10 minutes

# WhatsApp Cloud API text body limit is around 4096 chars.
# Keep below that because we may add part labels.
WHATSAPP_TEXT_LIMIT = int(os.getenv("WHATSAPP_TEXT_LIMIT", "3800"))

# If your sheet has unavailable/rented/sold rows and you want to hide them by default,
# set DEFAULT_AVAILABLE_ONLY=true in Render.
DEFAULT_AVAILABLE_ONLY = os.getenv("DEFAULT_AVAILABLE_ONLY", "false").lower() == "true"

# Optional: comma-separated zero-based column indexes to drop.
# Leave empty for complete data extraction.
# Example: DROP_COLUMN_INDEXES=8,9,10
DROP_COLUMN_INDEXES_RAW = os.getenv("DROP_COLUMN_INDEXES", "").strip()


# ============================================================
# REACTIVE-ONLY GUARANTEE (read this before changing the webhook code)
# ============================================================
#
# This bot must never send a WhatsApp message on its own -- only in direct
# reply to an incoming client message. There is exactly one call site for
# send_whatsapp_message() in this whole file, and it is only reachable via:
#
#   POST /webhook  --(FastAPI request)-->  background_tasks.add_task(process_message_background)
#                                       --> process_message_background()
#                                       --> send_whatsapp_message()
#
# There is no scheduler, cron, polling loop, retry timer, or startup/wake
# hook anywhere in this file. The only trigger is an actual incoming HTTP
# POST from Meta carrying a real client message. Two extra safeguards make
# this hold even under server restarts and Meta's webhook retries:
#
#   1. mark_message_seen() is backed by a small SQLite file (DEDUP_DB_PATH),
#      not an in-memory dict. A dict forgets everything on every restart --
#      including routine sleep/wake cycles on free hosting tiers -- so a
#      webhook Meta retries after downtime would look brand new and get a
#      fresh reply. SQLite survives the restart, so it doesn't.
#   2. Independently of #1, any message older than MAX_MESSAGE_AGE_SECONDS
#      (by its own WhatsApp timestamp) is dropped with no reply at all,
#      full stop -- even if it were somehow never marked "seen" before.
#      This is what actually rules out "wakes up after being idle for 2/4/6
#      hours and messages someone out of nowhere": a message that old never
#      reaches send_whatsapp_message() regardless of dedup state.
#
# Do not add any code that calls send_whatsapp_message() or
# process_message_background() from outside an incoming POST /webhook
# request (no @app.on_event("startup"), no background scheduler, etc.)
# without re-reading this block.
# ============================================================


# ============================================================
# Global In-Memory Stores
# ============================================================

http_session = requests.Session()

# Per-user conversation sessions.
user_sessions: Dict[str, Dict[str, Any]] = {}
sessions_lock = threading.RLock()

# Per-user locks so two messages from the same user do not process out of order.
user_locks: Dict[str, threading.Lock] = {}
user_locks_guard = threading.RLock()

# Google Sheet cache.
property_cache: Dict[str, Any] = {
    "loaded_at": 0.0,
    "records": [],
    "schema": {},
    "columns": [],
}
property_cache_lock = threading.RLock()

# knowledge.json cache.
knowledge_cache: Dict[str, Any] = {
    "loaded_at": 0.0,
    "data": {},
}
knowledge_cache_lock = threading.RLock()

# Google Places cache to reduce API cost and latency.
places_cache: Dict[str, Any] = {
    "items": {},
}
places_cache_lock = threading.RLock()

# WhatsApp message de-duplication. Storage itself is a small SQLite file
# (see DEDUP_DB_PATH / REACTIVE-ONLY GUARANTEE above) so it survives
# restarts; this lock just serializes access from this process.
seen_lock = threading.RLock()


# ============================================================
# Constants / Matching Helpers
# ============================================================

INTERNAL_SEARCH_KEY = "__search_text"

HEADER_HINTS = {
    "property", "building", "tower", "project", "location", "area", "community",
    "unit", "flat", "apartment", "type", "bed", "bhk", "price", "rent",
    "amount", "size", "sqft", "sq ft", "status", "availability",
    "description", "remarks", "notes", "details"
}

STOP_WORDS = {
    "a", "an", "the", "in", "at", "for", "of", "on", "and", "or", "to",
    "with", "me", "my", "we", "us", "you", "your", "do", "does", "did",
    "have", "has", "any", "some", "please", "pls", "show", "send", "give",
    "want", "need", "looking", "find", "search", "property", "properties",
    "real", "estate", "dubai", "uae", "al"
}

PROPERTY_KEYWORDS = {
    "property", "properties", "building", "tower", "project", "unit", "flat",
    "apartment", "studio", "bedroom", "bedrooms", "bhk", "br", "rent",
    "price", "size", "sqft", "sq ft", "available", "availability", "vacant",
    "sale", "buy", "lease", "viewing", "book", "booking"
}

# Which filter keys count as "specific" and should be remembered across a
# client's follow-up messages in the same session. Defined once here so
# every place that touches session/filter persistence (and the "is this a
# specific-enough search" check) stays in sync automatically.
PERSISTABLE_FILTER_KEYS = {
    "location", "building", "unit_type", "available_only",
    "min_price", "max_price",
}

COLUMN_ALIASES = {
    "location": [
        "location", "area/location", "community", "locality", "district",
        "city", "place", "zone", "area"
    ],
    "building": [
        "property name", "building name", "building", "project name",
        "project", "tower name", "tower", "property"
    ],
    "unit_type": [
        "unit type", "property type", "type", "bedroom", "bedrooms",
        "beds", "bed", "bhk", "layout"
    ],
    "unit_no": [
        "unit no", "unit number", "unit #", "apartment no", "apartment number",
        "flat no", "flat number", "flat", "unit"
    ],
    "price": [
        "price", "rent", "annual rent", "yearly rent", "monthly rent",
        "selling price", "sale price", "amount", "rate"
    ],
    "size": [
        "size", "sqft", "sq ft", "area sqft", "area sq ft",
        "built up area", "bua"
    ],
    "status": [
        "status", "availability", "available", "vacant"
    ],
    "description": [
        "description", "details", "remarks", "features", "notes",
        "comment", "comments"
    ],
}

FIELD_EXCLUDES = {
    "location": ["sqft", "sq ft", "size", "bua", "built"],
    "building": ["type", "unit no", "unit number", "unit #"],
    "unit_type": ["unit no", "unit number", "unit #"],
    "unit_no": ["unit type", "property type"],
    "price": ["size", "sqft", "sq ft"],
    "size": ["price", "rent", "amount"],
}

# Extra schema fields used by the Responses API tools, Google Places,
# rental yield display, and video-tour handling.
COLUMN_ALIASES.update({
    "id": [
        "id", "unique id", "property id", "listing id", "prop id",
        "prop code", "property code", "code", "ref", "reference"
    ],
    "rental_yield": [
        "rental yield", "yield", "roi", "return on investment",
        "gross yield", "net yield", "returns"
    ],
    "landmark_keywords": [
        "landmark keywords", "landmark_keywords", "location keywords",
        "nearby keywords", "nearby locations", "nearby", "landmarks",
        "google place keywords", "places keywords"
    ],
    "video_link": [
        "video link", "video_link", "video", "video tour",
        "tour link", "walkthrough", "youtube", "youtube link",
        "drive video", "property video"
    ],
    "latitude": [
        "latitude", "lat", "property latitude"
    ],
    "longitude": [
        "longitude", "lng", "long", "property longitude"
    ],
})

FIELD_EXCLUDES.update({
    "id": [],
    "rental_yield": ["price", "rent", "amount"],
    "landmark_keywords": [],
    "video_link": [],
    "latitude": ["longitude"],
    "longitude": ["latitude"],
})

PROPERTY_KEYWORDS.update({
    "villa", "townhouse", "penthouse", "yield", "roi", "investment",
    "invest", "nearby", "amenities", "metro", "school", "mall",
    "park", "supermarket", "video", "tour"
})


# ============================================================
# Basic Text Utilities
# ============================================================

def clean_cell(value: Any) -> str:
    """
    Cleans Google Sheet cell values while preserving human-readable text.
    """
    if value is None:
        return ""

    text = str(value).replace("\xa0", " ").strip()

    if text.lower() in {"nan", "none", "nat", "<na>"}:
        return ""

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_text(value: Any) -> str:
    """
    Lowercase normalization used for comparison.
    """
    text = clean_cell(value).lower()
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def searchable_text(value: Any) -> str:
    """
    Converts text into a search-safe form:
    - lowercase
    - punctuation replaced with spaces
    - multiple spaces collapsed
    """
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_header(value: Any) -> str:
    return searchable_text(value)


def meaningful_tokens(text: Any) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    return [
        token for token in tokens
        if token not in STOP_WORDS and (len(token) >= 2 or token.isdigit())
    ]


def phrase_in_text(needle: Any, haystack: Any) -> bool:
    """
    Safe phrase check after search normalization.
    """
    n = searchable_text(needle)
    h = searchable_text(haystack)

    if not n or not h:
        return False

    return f" {n} " in f" {h} "


def compact_for_history(text: str, limit: int = 1800) -> str:
    text = clean_cell(text)
    if len(text) <= limit:
        return text
    return text[:limit] + " ... [truncated]"


def row_search_text(record: Dict[str, Any]) -> str:
    cached = record.get(INTERNAL_SEARCH_KEY)
    if cached is not None:
        return cached

    return " ".join(
        searchable_text(value)
        for key, value in record.items()
        if key != INTERNAL_SEARCH_KEY and clean_cell(value)
    )


# ============================================================
# Session Memory
# ============================================================

def get_user_lock(sender: str) -> threading.Lock:
    with user_locks_guard:
        if sender not in user_locks:
            user_locks[sender] = threading.Lock()
        return user_locks[sender]


def cleanup_sessions_locked(now: float) -> None:
    expired = [
        sender for sender, session in user_sessions.items()
        if now - session.get("last_updated", 0.0) > SESSION_TIMEOUT_SECONDS
    ]

    for sender in expired:
        user_sessions.pop(sender, None)


def get_session_snapshot(sender: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Returns a copy of the previous conversation history and state.
    """
    now = time.time()

    with sessions_lock:
        cleanup_sessions_locked(now)

        if sender not in user_sessions:
            user_sessions[sender] = {
                "history": [],
                "state": {},
                "last_updated": now,
            }

        session = user_sessions[sender]
        session["last_updated"] = now

        history_copy = list(session.get("history", []))
        state_copy = dict(session.get("state", {}))

    return history_copy, state_copy


def update_session(
    sender: str,
    user_text: str,
    assistant_text: str,
    filters: Dict[str, Any] = None,
    search_text: str = "",
    matched: bool = False,
) -> None:
    """
    Stores line-by-line conversation history and updates last search filters.
    """
    now = time.time()
    filters = filters or {}

    with sessions_lock:
        if sender not in user_sessions:
            user_sessions[sender] = {
                "history": [],
                "state": {},
                "last_updated": now,
            }

        session = user_sessions[sender]
        session["last_updated"] = now

        session["history"].append({
            "role": "user",
            "content": compact_for_history(user_text, 1000),
        })

        session["history"].append({
            "role": "assistant",
            "content": compact_for_history(assistant_text, 1800),
        })

        if len(session["history"]) > MAX_HISTORY_MESSAGES:
            session["history"] = session["history"][-MAX_HISTORY_MESSAGES:]

        if matched:
            saved_filters = {
                key: value
                for key, value in filters.items()
                if key in PERSISTABLE_FILTER_KEYS and value
            }

            if saved_filters:
                session["state"]["last_filters"] = saved_filters

            if clean_cell(search_text):
                session["state"]["last_search_text"] = clean_cell(search_text)

            session["state"]["last_matched_at"] = now


def reset_session(sender: str) -> None:
    with sessions_lock:
        user_sessions.pop(sender, None)


def is_reset_command(text: str) -> bool:
    norm = normalize_text(text)
    return norm in {
        "reset",
        "clear",
        "restart",
        "start over",
        "new search",
        "forget",
        "clear chat",
        "reset chat",
    }


# ============================================================
# Google Sheet Loading / Schema Detection
# ============================================================

def parse_drop_column_indexes() -> List[int]:
    if not DROP_COLUMN_INDEXES_RAW:
        return []

    indexes = []
    for item in DROP_COLUMN_INDEXES_RAW.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            indexes.append(int(item))
        except ValueError:
            print(f"Ignoring invalid DROP_COLUMN_INDEXES item: {item}")

    return indexes


def make_unique_columns(columns: List[Any]) -> List[str]:
    cleaned = []
    seen = {}

    for index, column in enumerate(columns):
        name = clean_cell(column)

        if not name or name.lower().startswith("unnamed"):
            name = f"Column_{index + 1}"

        name = re.sub(r"\s+", " ", name).strip()
        base = name

        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1

        cleaned.append(name)

    return cleaned


def detect_header_row(raw_df: pd.DataFrame) -> int:
    """
    Detects the most likely header row within the first 15 rows.

    This is safer than blindly doing:
        df.columns = df.iloc[0]
    because Google Sheets CSV sometimes already exposes headers properly,
    and sometimes has blank/metadata rows above the real header.
    """
    best_index = 0
    best_score = -1.0
    max_rows = min(15, len(raw_df))

    for idx in range(max_rows):
        row_values = [normalize_header(value) for value in raw_df.iloc[idx].tolist()]
        joined = " ".join(row_values)
        non_empty_count = sum(1 for value in row_values if value)

        score = 0.0

        for hint in HEADER_HINTS:
            if hint in joined:
                score += 2.0

        # Stronger signals for common real-estate sheet headers.
        strong_phrases = [
            "property name",
            "building name",
            "unit no",
            "unit number",
            "unit type",
            "annual rent",
            "sale price",
            "availability",
        ]

        for phrase in strong_phrases:
            if phrase in joined:
                score += 5.0

        score += min(non_empty_count, 12) * 0.1

        if score > best_score:
            best_score = score
            best_index = idx

    return best_index


def find_column(
    columns: List[str],
    aliases: List[str],
    excludes: List[str] = None,
    used: set = None,
) -> str:
    excludes = excludes or []
    used = used or set()

    normalized_columns = [
        (column, normalize_header(column))
        for column in columns
        if column not in used
    ]

    def allowed(normalized_column: str) -> bool:
        return not any(exclude in normalized_column for exclude in excludes)

    # Exact match first.
    for alias in aliases:
        normalized_alias = normalize_header(alias)
        if not normalized_alias:
            continue

        for column, normalized_column in normalized_columns:
            if not allowed(normalized_column):
                continue

            if normalized_column == normalized_alias:
                return column

    # Partial match second.
    for alias in aliases:
        normalized_alias = normalize_header(alias)
        if not normalized_alias or len(normalized_alias) < 3:
            continue

        for column, normalized_column in normalized_columns:
            if not allowed(normalized_column):
                continue

            if normalized_alias in normalized_column:
                return column

    return ""


def resolve_schema(columns: List[str]) -> Dict[str, str]:
    """
    Attempts to identify important real estate columns.

    Even if this does not perfectly detect every column, the formatter still
    prints all columns from each matching row.
    """
    schema = {}
    used = set()

    field_order = [
        "id",
        "location",
        "building",
        "unit_type",
        "unit_no",
        "price",
        "size",
        "status",
        "rental_yield",
        "landmark_keywords",
        "video_link",
        "latitude",
        "longitude",
        "description",
    ]

    for field in field_order:
        column = find_column(
            columns=columns,
            aliases=COLUMN_ALIASES.get(field, []),
            excludes=FIELD_EXCLUDES.get(field, []),
            used=used,
        )

        if column:
            schema[field] = column
            used.add(column)

    return schema


def load_properties_from_sheet() -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    """
    Loads complete Google Sheet data.

    Main improvements over your original function:
    - Uses header=None and detects the header row safely.
    - Does not drop columns by default.
    - Preserves all non-empty columns.
    - Preserves values as strings.
    - Adds an internal normalized search field for speed.
    """
    if not GOOGLE_SHEET_URL:
        print("GOOGLE_SHEET_URL is not configured.")
        return [], {}, []

    raw = pd.read_csv(
        GOOGLE_SHEET_URL,
        header=None,
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
    )

    raw = raw.fillna("")

    # Remove fully empty rows.
    raw = raw.loc[
        raw.apply(lambda row: any(clean_cell(value) for value in row), axis=1)
    ].reset_index(drop=True)

    # Remove fully empty columns.
    raw = raw.loc[
        :,
        raw.apply(lambda col: any(clean_cell(value) for value in col), axis=0)
    ]

    if raw.empty:
        print("Google Sheet loaded, but no usable rows were found.")
        return [], {}, []

    header_index = detect_header_row(raw)
    columns = make_unique_columns(raw.iloc[header_index].tolist())

    df = raw.iloc[header_index + 1:].copy()
    df.columns = columns

    # Clean every cell.
    df = df.apply(lambda col: col.map(clean_cell))

    # Remove empty rows after header.
    df = df.loc[
        df.apply(lambda row: any(clean_cell(value) for value in row), axis=1)
    ]

    # Optional column dropping. Disabled by default to preserve complete data.
    drop_indexes = parse_drop_column_indexes()
    if drop_indexes:
        drop_columns = [
            df.columns[index]
            for index in drop_indexes
            if 0 <= index < len(df.columns)
        ]
        if drop_columns:
            df = df.drop(columns=drop_columns, errors="ignore")

    columns = list(df.columns)
    schema = resolve_schema(columns)

    records = df.to_dict(orient="records")

    # Remove empty records and precompute internal search text.
    cleaned_records = []
    for record in records:
        if not any(clean_cell(value) for value in record.values()):
            continue

        record[INTERNAL_SEARCH_KEY] = " ".join(
            searchable_text(value)
            for key, value in record.items()
            if key != INTERNAL_SEARCH_KEY and clean_cell(value)
        )

        cleaned_records.append(record)

    print(f"Loaded {len(cleaned_records)} property records.")
    print(f"Detected columns: {columns}")
    print(f"Detected schema: {schema}")

    return cleaned_records, schema, columns


def get_properties(force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    """
    Cached property loader.
    """
    now = time.time()

    with property_cache_lock:
        cache_valid = (
            property_cache["records"]
            and now - property_cache["loaded_at"] < PROPERTY_CACHE_TTL_SECONDS
        )

        if cache_valid and not force_refresh:
            return (
                property_cache["records"],
                property_cache["schema"],
                property_cache["columns"],
            )

    try:
        records, schema, columns = load_properties_from_sheet()

        with property_cache_lock:
            property_cache["loaded_at"] = now
            property_cache["records"] = records
            property_cache["schema"] = schema
            property_cache["columns"] = columns

        return records, schema, columns

    except Exception as error:
        print("Error loading Google Sheet:", error)
        traceback.print_exc()

        # If old cache exists, use it instead of failing completely.
        with property_cache_lock:
            return (
                property_cache.get("records", []),
                property_cache.get("schema", {}),
                property_cache.get("columns", []),
            )


# ============================================================
# knowledge.json Loading
# ============================================================

def get_knowledge() -> Dict[str, Any]:
    now = time.time()

    with knowledge_cache_lock:
        cache_valid = (
            knowledge_cache["data"]
            and now - knowledge_cache["loaded_at"] < KNOWLEDGE_CACHE_TTL_SECONDS
        )

        if cache_valid:
            return knowledge_cache["data"]

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

    except FileNotFoundError:
        data = {}

    except Exception as error:
        print("Error loading knowledge.json:", error)
        traceback.print_exc()
        data = {}

    with knowledge_cache_lock:
        knowledge_cache["loaded_at"] = now
        knowledge_cache["data"] = data

    return data


# ============================================================
# Entity Extraction / Search Logic
# ============================================================

def split_possible_values(value: str) -> List[str]:
    """
    Useful for location cells like:
        "Bur Dubai - Al Raffa"
        "Al Raffa, Dubai"
    """
    text = clean_cell(value)
    if not text:
        return []

    parts = [text]

    split_parts = re.split(r"\s*(?:,|/|;|\||\n|\r| - | – | — )\s*", text)
    for part in split_parts:
        part = clean_cell(part)
        if part:
            parts.append(part)

    unique = []
    seen = set()

    for part in parts:
        key = searchable_text(part)
        if key and key not in seen:
            seen.add(key)
            unique.append(part)

    return unique


def get_unique_column_values(
    records: List[Dict[str, Any]],
    column: str,
    split_values: bool = False,
) -> List[str]:
    values = []
    seen = set()

    if not column:
        return values

    for record in records:
        raw_value = clean_cell(record.get(column, ""))

        if not raw_value:
            continue

        candidates = split_possible_values(raw_value) if split_values else [raw_value]

        for candidate in candidates:
            key = searchable_text(candidate)
            if key and key not in seen:
                seen.add(key)
                values.append(candidate)

    return values


def best_value_match(user_text: str, values: List[str], mode: str = "exact") -> str:
    """
    Finds the best known value mentioned in the user's text.

    mode:
    - exact: value must be present as a phrase in the query.
    - partial/location: allows token overlap for area/building follow-ups.
    """
    query = searchable_text(user_text)
    if not query:
        return ""

    padded_query = f" {query} "
    query_tokens = [
        token for token in meaningful_tokens(user_text)
        if len(token) >= 3 or token.isdigit()
    ]

    candidates = []

    for value in values:
        value_clean = clean_cell(value)
        value_search = searchable_text(value_clean)

        if not value_search:
            continue

        # Strong phrase match.
        if f" {value_search} " in padded_query:
            candidates.append((10000 + len(value_search), value_clean))
            continue

        if mode in {"partial", "location"}:
            value_tokens = set(meaningful_tokens(value_clean))
            common_tokens = [token for token in query_tokens if token in value_tokens]

            if common_tokens:
                strong = (
                    any(len(token) >= 3 and not token.isdigit() for token in common_tokens)
                    or len(common_tokens) >= 2
                )

                if strong:
                    score = (
                        500
                        + 100 * len(common_tokens)
                        + sum(len(token) for token in common_tokens)
                        + len(value_search) / 100
                    )
                    candidates.append((score, value_clean))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def extract_unit_type_from_text(text: str) -> str:
    """
    Supports:
    - studio
    - 1BR / 1 BR / 1 bedroom / 1 bhk
    - 2BR / 2 bedroom / 2 bhk
    - one bedroom, two bedroom, etc.
    """
    norm = normalize_text(text)
    search = searchable_text(text)
    compact = re.sub(r"[^a-z0-9]+", "", norm)

    if "studio" in search:
        return "Studio"

    word_numbers = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }

    for word, number in word_numbers.items():
        pattern = rf"\b{word}\s*(?:br|bed|beds|bedroom|bedrooms|bhk)\b"
        if re.search(pattern, norm):
            return f"{number} BR"

    # Example: 1 br, 1 bedroom, 2 bhk
    match = re.search(
        r"\b([1-9])\s*(?:br|b r|b/r|bed|beds|bedroom|bedrooms|bhk)\b",
        search,
    )
    if match:
        return f"{match.group(1)} BR"

    # Compact form: 1br, 1bhk, 2bedroom
    match = re.search(r"([1-9])(?:br|bed|beds|bedroom|bedrooms|bhk)", compact)
    if match:
        return f"{match.group(1)} BR"

    return ""


def canonical_unit_type(value: str) -> str:
    detected = extract_unit_type_from_text(value)
    if detected:
        return detected
    return clean_cell(value)


def wants_available_only(text: str) -> bool:
    norm = normalize_text(text)
    phrases = [
        "available",
        "availability",
        "vacant",
        "ready to move",
        "ready now",
        "ready",
    ]
    return any(phrase in norm for phrase in phrases)


MONEY_VALUE_PATTERN = (
    r"\d[\d,]*(?:\.\d+)?\s*"
    r"(?:million|mn|mil|m|k|lakh|lac|crore|cr)?"
)

def parse_money_value(text: Any) -> Optional[float]:
    raw = clean_cell(text)
    if not raw:
        return None

    t = raw.lower().strip()
    t = t.replace("د.إ", " aed ")

    t = re.sub(
        r"\b(aed|dhs|dh|dirhams?|per\s*annum|per\s*year|/\s*year|"
        r"yearly|annual|annually|pa|only)\b",
        " ",
        t,
    )
    t = re.sub(r"\s+", " ", t).strip()

    match = re.search(
        r"(\d[\d,]*(?:\.\d+)?)\s*"
        r"(million|mn|mil|m|k|lakh|lac|crore|cr)?\b",
        t,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    number_text = match.group(1).replace(",", "")

    try:
        number = float(number_text)
    except ValueError:
        return None

    suffix = (match.group(2) or "").lower().strip()

    multipliers = {
        "k": 1_000,
        "m": 1_000_000,
        "mn": 1_000_000,
        "mil": 1_000_000,
        "million": 1_000_000,
        "lakh": 100_000,
        "lac": 100_000,
        "crore": 10_000_000,
        "cr": 10_000_000,
    }

    return number * multipliers.get(suffix, 1)


def extract_budget_from_text(text: str) -> Dict[str, float]:
    norm = normalize_text(text)
    result: Dict[str, float] = {}
    money = MONEY_VALUE_PATTERN

    between = re.search(
        rf"between\s+({money})\s*(?:and|to|-)\s*({money})",
        norm,
    )
    if between:
        low = parse_money_value(between.group(1))
        high = parse_money_value(between.group(2))
        if low is not None and high is not None:
            result["min_price"] = min(low, high)
            result["max_price"] = max(low, high)
            return result

    negated_max = re.search(
        rf"(?:nothing|not|no)\s+(?:above|over|more than)\s*"
        rf"(?:aed|dhs|dh)?\s*({money})",
        norm,
    )
    if negated_max:
        value = parse_money_value(negated_max.group(1))
        if value is not None:
            result["max_price"] = value
            return result

    negated_min = re.search(
        rf"(?:nothing|not|no)\s+(?:under|below|less than)\s*"
        rf"(?:aed|dhs|dh)?\s*({money})",
        norm,
    )
    if negated_min:
        value = parse_money_value(negated_min.group(1))
        if value is not None:
            result["min_price"] = value
            return result

    max_patterns = [
        rf"(?:under|below|less than|max(?:imum)?|up to|within)\s*"
        rf"(?:aed|dhs|dh)?\s*({money})",
        rf"budget(?:\s+is)?\s*(?:around|about|roughly)?\s*"
        rf"(?:aed|dhs|dh)?\s*[:=]?\s*({money})",
        rf"(?:budget|range)\s*(?:of|around|about)?\s*({money})\s*"
        rf"(?:max|maximum)?",
    ]

    for pattern in max_patterns:
        match = re.search(pattern, norm)
        if match:
            value = parse_money_value(match.group(1))
            if value is not None:
                result["max_price"] = value
                break

    min_patterns = [
        rf"(?:above|over|more than|min(?:imum)?|starting from|from)\s*"
        rf"(?:aed|dhs|dh)?\s*({money})",
    ]

    for pattern in min_patterns:
        match = re.search(pattern, norm)
        if match:
            value = parse_money_value(match.group(1))
            if value is not None:
                result["min_price"] = value
                break

    return result


def extract_filters_from_text(
    user_text: str,
    records: List[Dict[str, Any]],
    schema: Dict[str, str],
) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}

    location_column = schema.get("location", "")
    building_column = schema.get("building", "")
    unit_type_column = schema.get("unit_type", "")

    # Location detection.
    if location_column:
        location_values = get_unique_column_values(
            records,
            location_column,
            split_values=True,
        )
        location = best_value_match(user_text, location_values, mode="location")
        if location:
            filters["location"] = location

    # Building detection.
    if building_column:
        building_values = get_unique_column_values(
            records,
            building_column,
            split_values=False,
        )

        # First try exact building phrase.
        building = best_value_match(user_text, building_values, mode="exact")

        # If no location was detected, allow partial building match.
        # This allows users to type only "Fardan" instead of full building name.
        if not building and not filters.get("location"):
            building = best_value_match(user_text, building_values, mode="partial")

        if building:
            filters["building"] = building

    # Unit type detection.
    unit_type = extract_unit_type_from_text(user_text)

    # Also support unit types from sheet like Office, Shop, Villa, Penthouse, etc.
    if not unit_type and unit_type_column:
        unit_values = get_unique_column_values(
            records,
            unit_type_column,
            split_values=False,
        )
        unit_match = best_value_match(user_text, unit_values, mode="partial")
        if unit_match:
            unit_type = canonical_unit_type(unit_match)

    if unit_type:
        filters["unit_type"] = unit_type

    if wants_available_only(user_text):
        filters["available_only"] = True

    # Budget detection ("under 70k", "budget 50000", "between 40k and 60k").
    budget = extract_budget_from_text(user_text)
    if budget:
        filters.update(budget)

    return filters


def is_greeting(text: str) -> bool:
    norm = normalize_text(text)

    greetings = {
        "hi",
        "hello",
        "hey",
        "salam",
        "salaam",
        "assalamualaikum",
        "assalamu alaikum",
        "good morning",
        "good afternoon",
        "good evening",
    }

    return norm in greetings


def is_show_all_reset_request(text: str) -> bool:
    norm = normalize_text(text)

    if norm in {"all", "show all", "send all", "list all", "everything"}:
        return True

    phrases = [
        "show all",
        "send all",
        "list all",
        "all units",
        "all details",
        "full list",
        "complete list",
        "everything",
    ]

    return any(phrase in norm for phrase in phrases)


def looks_like_followup(text: str) -> bool:
    if is_greeting(text):
        return False

    norm = normalize_text(text)

    if is_show_all_reset_request(text):
        return True

    # A bare budget statement ("actually under 50k") is a refinement of the
    # ongoing search, not a fresh context-free query -- it should inherit
    # whatever location/building was already being discussed.
    if extract_budget_from_text(text):
        return True

    followup_words = {
        "details",
        "detail",
        "price",
        "rent",
        "size",
        "sqft",
        "status",
        "available",
        "availability",
        "viewing",
        "book",
        "booking",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "more",
    }

    tokens = set(meaningful_tokens(text))

    if tokens.intersection(followup_words):
        return True

    phrases = [
        "what about",
        "how about",
        "tell me more",
        "more details",
        "send details",
        "share details",
    ]

    return any(phrase in norm for phrase in phrases)


def build_effective_filters(
    current_filters: Dict[str, Any],
    previous_state: Dict[str, Any],
    user_text: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Merges current message filters with memory.

    Example:
    Previous: location=Al Raffa
    Current: Studio
    Effective: location=Al Raffa + unit_type=Studio

    Budget (min_price/max_price) is treated as a standing constraint: once a
    client states one, it carries forward across follow-ups the same way
    location/building already do, unless they state a new one.
    """
    previous_filters = previous_state.get("last_filters", {}) or {}
    previous_search_text = clean_cell(previous_state.get("last_search_text", ""))

    effective: Dict[str, Any] = {}
    search_text = user_text

    current_location = current_filters.get("location")
    current_building = current_filters.get("building")
    current_unit_type = current_filters.get("unit_type")

    def inherit_budget() -> None:
        """Carries previous budget forward unless this turn states a new one."""
        for key in ("min_price", "max_price"):
            if key in previous_filters:
                effective.setdefault(key, previous_filters[key])

    if current_location:
        # New location should reset old building/unit unless explicitly present now.
        effective["location"] = current_location

        if current_building:
            effective["building"] = current_building

        if current_unit_type:
            effective["unit_type"] = current_unit_type

        inherit_budget()

    elif current_building:
        # Building-only follow-up can still use previous location as context,
        # but if it gives zero results, search falls back to the building text.
        if previous_filters.get("location"):
            effective["location"] = previous_filters["location"]

        effective["building"] = current_building

        if current_unit_type:
            effective["unit_type"] = current_unit_type

        inherit_budget()

    elif current_unit_type:
        inherited_context = False

        if previous_filters.get("location"):
            effective["location"] = previous_filters["location"]
            inherited_context = True

        if previous_filters.get("building"):
            effective["building"] = previous_filters["building"]
            inherited_context = True

        effective["unit_type"] = current_unit_type
        inherit_budget()

        # If schema could not identify location/building previously,
        # still use the previous raw search text as base context.
        if not inherited_context and previous_search_text:
            search_text = previous_search_text

    elif is_show_all_reset_request(user_text) and (previous_filters or previous_search_text):
        # "show all" should usually remove the previous unit type filter,
        # but preserve location/building/budget.
        if previous_filters.get("location"):
            effective["location"] = previous_filters["location"]

        if previous_filters.get("building"):
            effective["building"] = previous_filters["building"]

        inherit_budget()

        search_text = previous_search_text or user_text

    elif looks_like_followup(user_text) and (previous_filters or previous_search_text):
        # "price?", "details?", "available?" should use the last filters.
        effective = {
            key: value
            for key, value in previous_filters.items()
            if key in PERSISTABLE_FILTER_KEYS and value is not None and value != ""
        }

        search_text = previous_search_text or user_text

    else:
        effective = {
            key: value
            for key, value in current_filters.items()
            if key in PERSISTABLE_FILTER_KEYS and value is not None and value != ""
        }

        inherit_budget()

    # A budget stated in THIS message always wins over an inherited one.
    for budget_key in ("min_price", "max_price"):
        if current_filters.get(budget_key) is not None:
            effective[budget_key] = current_filters[budget_key]

    if current_filters.get("available_only") or DEFAULT_AVAILABLE_ONLY:
        effective["available_only"] = True

    return effective, search_text


def unit_type_matches_record(
    record: Dict[str, Any],
    desired_unit_type: str,
    schema: Dict[str, str],
) -> bool:
    desired = canonical_unit_type(desired_unit_type)

    if not desired:
        return True

    unit_type_column = schema.get("unit_type", "")

    text_parts = []
    if unit_type_column:
        text_parts.append(clean_cell(record.get(unit_type_column, "")))

    text_parts.append(row_search_text(record))

    combined_normal = normalize_text(" ".join(text_parts))
    combined_search = searchable_text(" ".join(text_parts))
    compact = re.sub(r"[^a-z0-9]+", "", combined_normal)

    if desired.lower() == "studio":
        return "studio" in combined_search

    br_match = re.match(r"^([1-9])\s*BR$", desired, flags=re.IGNORECASE)
    if br_match:
        number = br_match.group(1)

        compact_patterns = [
            f"{number}br",
            f"{number}bhk",
            f"{number}bed",
            f"{number}beds",
            f"{number}bedroom",
            f"{number}bedrooms",
        ]

        if any(pattern in compact for pattern in compact_patterns):
            return True

        regex = rf"\b{number}\s*(?:br|bed|beds|bedroom|bedrooms|bhk)\b"
        if re.search(regex, combined_normal):
            return True

        # If unit-type column simply says "1" / "2".
        if unit_type_column:
            field_value = searchable_text(record.get(unit_type_column, ""))
            if field_value == number:
                return True

        return False

    # Non-bedroom types like office/shop/villa.
    if unit_type_column and phrase_in_text(desired, record.get(unit_type_column, "")):
        return True

    return phrase_in_text(desired, row_search_text(record))


def status_is_available(status_text: str) -> bool:
    """
    Keeps rows unless they are clearly unavailable.
    """
    status = normalize_text(status_text)

    if not status:
        return True

    negative_phrases = [
        "not available",
        "unavailable",
        "rented",
        "sold",
        "booked",
        "occupied",
        "blocked",
        "hold",
        "on hold",
        "leased",
    ]

    return not any(phrase in status for phrase in negative_phrases)


def filter_available_records(
    records: List[Dict[str, Any]],
    schema: Dict[str, str],
) -> List[Dict[str, Any]]:
    status_column = schema.get("status", "")

    if not status_column:
        return records

    return [
        record for record in records
        if status_is_available(record.get(status_column, ""))
    ]


def apply_hard_filters(
    records: List[Dict[str, Any]],
    filters: Dict[str, Any],
    schema: Dict[str, str],
) -> List[Dict[str, Any]]:
    candidates = list(records)

    location = filters.get("location", "")
    building = filters.get("building", "")
    unit_type = filters.get("unit_type", "")

    location_column = schema.get("location", "")
    building_column = schema.get("building", "")

    if location:
        candidates = [
            record for record in candidates
            if (
                location_column and phrase_in_text(location, record.get(location_column, ""))
            )
            or phrase_in_text(location, row_search_text(record))
        ]

    if building:
        candidates = [
            record for record in candidates
            if (
                building_column and phrase_in_text(building, record.get(building_column, ""))
            )
            or phrase_in_text(building, row_search_text(record))
        ]

    if unit_type:
        candidates = [
            record for record in candidates
            if unit_type_matches_record(record, unit_type, schema)
        ]

    min_price = filters.get("min_price")
    max_price = filters.get("max_price")
    price_column = schema.get("price", "")

    if (min_price is not None or max_price is not None) and price_column:
        def in_budget(record: Dict[str, Any]) -> bool:
            record_price = parse_money_value(record.get(price_column, ""))
            if record_price is None:
                # Row has no parseable price -- don't hide it just because
                # of a budget filter; a missing figure isn't evidence it's
                # out of budget.
                return True
            if min_price is not None and record_price < min_price:
                return False
            if max_price is not None and record_price > max_price:
                return False
            return True

        candidates = [record for record in candidates if in_budget(record)]

    if filters.get("available_only"):
        candidates = filter_available_records(candidates, schema)

    return candidates


def score_search_records(
    query: str,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Scored fallback search across the entire row text.

    This solves cases where column detection is imperfect or a user types
    only a location keyword like "Al Raffa".
    """
    query_search = searchable_text(query)
    tokens = [
        token for token in meaningful_tokens(query)
        if len(token) >= 3 or token.isdigit()
    ]

    if not query_search and not tokens:
        return []

    scored = []

    for index, record in enumerate(records):
        text = row_search_text(record)
        score = 0

        # Exact phrase boost. Very important for areas like "Al Raffa".
        if query_search and f" {query_search} " in f" {text} ":
            score += 1000

        # Token matching.
        for token in tokens:
            if re.search(rf"\b{re.escape(token)}\b", text):
                score += 30 if len(token) >= 4 else 15
            elif token in text:
                score += 5

        # If all meaningful tokens are present, boost.
        if tokens and all(token in text for token in tokens):
            score += 120

        if score > 0:
            scored.append((score, index, record))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [record for _, _, record in scored]


def dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = []
    seen = set()

    for record in records:
        key = tuple(
            (column, clean_cell(value))
            for column, value in record.items()
            if column != INTERNAL_SEARCH_KEY and clean_cell(value)
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(record)

    return unique


def has_specific_property_filter(filters: Dict[str, Any]) -> bool:
    return any(
        filters.get(key) is not None
        for key in ["location", "building", "unit_type", "min_price", "max_price"]
    )


def search_properties(
    user_text: str,
    records: List[Dict[str, Any]],
    schema: Dict[str, str],
    filters: Dict[str, Any],
    fallback_search_text: str = "",
) -> List[Dict[str, Any]]:
    if not records:
        return []

    fallback_search_text = clean_cell(fallback_search_text) or user_text

    only_unit_followup = (
        filters.get("unit_type")
        and not filters.get("location")
        and not filters.get("building")
        and searchable_text(fallback_search_text) != searchable_text(user_text)
    )

    # If user says "Studio" after "Al Raffa", but schema did not identify location,
    # first search previous text "Al Raffa", then filter studios within that.
    if only_unit_followup:
        base_records = score_search_records(fallback_search_text, records)

        if base_records:
            narrowed = [
                record for record in base_records
                if unit_type_matches_record(record, filters["unit_type"], schema)
            ]

            if filters.get("available_only"):
                narrowed = filter_available_records(narrowed, schema)

            if narrowed:
                return dedupe_records(narrowed)

    specific_filter_present = has_specific_property_filter(filters)

    if specific_filter_present or filters.get("available_only"):
        hard_filtered = apply_hard_filters(records, filters, schema)

        if hard_filtered:
            return dedupe_records(hard_filtered)

        # If hard filters gave no results, fallback to normal text search.
        fallback_results = score_search_records(user_text, records)

        if not fallback_results and fallback_search_text:
            fallback_results = score_search_records(fallback_search_text, records)

        if filters.get("unit_type"):
            unit_filtered = [
                record for record in fallback_results
                if unit_type_matches_record(record, filters["unit_type"], schema)
            ]
            if unit_filtered:
                fallback_results = unit_filtered

        if filters.get("available_only"):
            fallback_results = filter_available_records(fallback_results, schema)

        return dedupe_records(fallback_results)

    # No detected filters: score search the raw message.
    scored_results = score_search_records(fallback_search_text, records)

    if filters.get("available_only"):
        scored_results = filter_available_records(scored_results, schema)

    return dedupe_records(scored_results)


def is_likely_property_query(
    text: str,
    filters: Dict[str, Any],
    previous_state: Dict[str, Any],
) -> bool:
    if filters:
        return True

    if extract_unit_type_from_text(text):
        return True

    norm = normalize_text(text)

    if any(keyword in norm for keyword in PROPERTY_KEYWORDS):
        return True

    if looks_like_followup(text) and previous_state.get("last_filters"):
        return True

    return False


# ============================================================
# Responses API Tools / Google Places / Video Tour Helpers
# ============================================================

SYSTEM_INSTRUCTIONS = """
# ROLE
You are an elite, highly professional Real Estate Consultant operating in Dubai.
Provide a premium, VIP experience: polite, relaxing, confidence-inspiring.
Keep responses concise and scannable -- never large blocks of text.

# DATA GOVERNANCE -- 0% HALLUCINATION
- Only use data returned by the `search_listings` and `get_nearby_places` tools.
- Never invent prices, availability, rental yields, unit numbers, sizes, or neighborhood details.
- If information is not available from those tools, say so plainly and offer to connect the client with a human expert.

# LOCATION & AMENITY HIGHLIGHTS
- For any property, use `get_nearby_places` with the property's landmark keywords or coordinates.
- Present only amenities actually returned by that tool.
- Good phrasing:
  "This property offers exceptional convenience, located moments from {metro} and {school}."

# VIDEO TOURS
- If `search_listings` returns a video_link, do NOT send the link immediately.
- Offer it at the end:
  "I also have a stunning, high-quality video tour of this exact property -- would you like me to send it over?"
- Wait for an affirmative reply before sending the link.

# HONESTY
- If a client directly asks whether they are speaking with a human or AI, answer truthfully.
- Everything else about the premium tone stays the same.

# CONTACT
For viewing, booking, or expert assistance, refer the client to:
Mr. Zahid at +971562625777.
"""

TOOLS = [
    {
        "type": "function",
        "name": "search_listings",
        "description": (
            "Search the live property listings sheet for units matching "
            "the client's criteria. Returns price, availability, rental "
            "yield, landmark_keywords, and video_link when present."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "description": "Neighbourhood, community, building, or project, e.g. 'Dubai Marina'",
                },
                "property_type": {
                    "type": "string",
                    "enum": ["apartment", "villa", "townhouse", "penthouse"],
                },
                "bedrooms": {
                    "type": "integer",
                    "description": "Number of bedrooms, e.g. 2 for 2-bedroom",
                },
                "budget_min_aed": {
                    "type": "number",
                    "description": "Minimum budget in AED",
                },
                "budget_max_aed": {
                    "type": "number",
                    "description": "Maximum budget in AED",
                },
            },
            "required": ["area"],
        },
    },
    {
        "type": "function",
        "name": "get_nearby_places",
        "description": (
            "Look up verified nearby amenities for a property via Google Places: "
            "metro stations, schools, malls, parks, and supermarkets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "landmark_keywords": {
                    "type": "string",
                    "description": "Property landmark keywords, building name, or area, e.g. 'Dubai Marina Mall, JBR'",
                },
                "place_type": {
                    "type": "string",
                    "enum": [
                        "school",
                        "metro_station",
                        "shopping_mall",
                        "park",
                        "supermarket",
                    ],
                },
                "radius_meters": {
                    "type": "integer",
                    "default": 1500,
                },
                "latitude": {
                    "type": "number",
                    "description": "Optional property latitude if available",
                },
                "longitude": {
                    "type": "number",
                    "description": "Optional property longitude if available",
                },
            },
            "required": ["landmark_keywords", "place_type"],
        },
    },
]


CLIENT_HIDDEN_COLUMN_KEYWORDS = [
    "s n",
    "serial",
    "prop code",
    "property code",
    "moveout date",
    "vacant on",
    "ageing",
    "previous rent",
    "unique id",
    "property details",
    "video",
    "tour link",
    "youtube",
    "vimeo",
    "google drive",
    "location keywords",
    "landmark keywords",
    "nearby keywords",
]

CLIENT_HIDDEN_COLUMN_EXACT = {
    "latitude",
    "longitude",
    "lat",
    "lng",
    "long",
    "property latitude",
    "property longitude",
}


def should_hide_client_column(column: str) -> bool:
    """
    Hides internal/sensitive/technical columns from WhatsApp replies.
    Video links are intentionally hidden until the client says yes.
    """
    col = normalize_header(column)

    if not col:
        return True

    if "unnamed" in col:
        return True

    if re.match(r"^column\s*\d+$", col):
        return True

    if col in CLIENT_HIDDEN_COLUMN_EXACT:
        return True

    return any(
        normalize_header(hidden) in col
        for hidden in CLIENT_HIDDEN_COLUMN_KEYWORDS
        if normalize_header(hidden)
    )


def get_record_value_by_field(
    record: Dict[str, Any],
    schema: Dict[str, str],
    field: str,
) -> str:
    """
    Gets a value from a record using resolved schema first, then alias scan.
    """
    column = schema.get(field, "")

    if column:
        value = clean_cell(record.get(column, ""))
        if value:
            return value

    aliases = COLUMN_ALIASES.get(field, [])

    for column_name, value in record.items():
        if column_name == INTERNAL_SEARCH_KEY:
            continue

        normalized_column = normalize_header(column_name)

        for alias in aliases:
            normalized_alias = normalize_header(alias)

            if not normalized_alias:
                continue

            if normalized_column == normalized_alias:
                cleaned = clean_cell(value)
                if cleaned:
                    return cleaned

            if len(normalized_alias) >= 3 and normalized_alias in normalized_column:
                cleaned = clean_cell(value)
                if cleaned:
                    return cleaned

    return ""


def parse_float_cell(value: Any) -> Optional[float]:
    value = clean_cell(value)

    if not value:
        return None

    value = re.sub(r"[^\d.\-]", "", value)

    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    parsed = parse_money_value(value)
    return parsed


def build_listing_label(record: Dict[str, Any], schema: Dict[str, str]) -> str:
    parts = []

    building = get_record_value_by_field(record, schema, "building")
    unit_type = get_record_value_by_field(record, schema, "unit_type")
    unit_no = get_record_value_by_field(record, schema, "unit_no")

    if building:
        parts.append(building)

    if unit_type:
        parts.append(unit_type)

    if unit_no:
        parts.append(f"Unit {unit_no}")

    return " | ".join(parts) if parts else "Property"


def extract_video_urls(value: Any) -> List[str]:
    text = clean_cell(value)

    if not text:
        return []

    urls = re.findall(r"https?://[^\s<>()\[\]{}\"']+", text)

    if not urls and re.match(r"^www\.", text, flags=re.IGNORECASE):
        urls = [f"https://{text}"]

    cleaned_urls = []
    seen = set()

    for url in urls:
        url = url.rstrip(".,;)")
        key = url.strip()

        if key and key not in seen:
            seen.add(key)
            cleaned_urls.append(key)

    return cleaned_urls


def extract_video_links_from_records(
    records: List[Dict[str, Any]],
    schema: Dict[str, str],
    columns: List[str] = None,
) -> List[Dict[str, str]]:
    """
    Extracts video tour links from matching records.
    These links are stored in session and only sent after an affirmative reply.
    """
    links = []
    seen = set()

    for record in records:
        video_text = get_record_value_by_field(record, schema, "video_link")

        # Fallback scan if schema detection did not catch the video column.
        if not video_text:
            for column, value in record.items():
                if column == INTERNAL_SEARCH_KEY:
                    continue

                col_norm = normalize_header(column)

                if (
                    "video" in col_norm
                    or "tour link" in col_norm
                    or "youtube" in col_norm
                    or "vimeo" in col_norm
                ):
                    video_text = clean_cell(value)
                    if video_text:
                        break

        urls = extract_video_urls(video_text)

        for url in urls:
            if url in seen:
                continue

            seen.add(url)
            links.append({
                "label": build_listing_label(record, schema),
                "url": url,
            })

    return links


def set_pending_video_links(sender: str, links: List[Dict[str, str]]) -> None:
    """
    Stores video links in session, but does not send them yet.
    """
    normalized_links = []
    seen = set()

    for item in links or []:
        if isinstance(item, dict):
            url = clean_cell(item.get("url") or item.get("video_link") or "")
            label = clean_cell(item.get("label") or "Property video tour")
        else:
            url = clean_cell(item)
            label = "Property video tour"

        urls = extract_video_urls(url) or ([url] if url.startswith("http") else [])

        for clean_url in urls:
            if clean_url in seen:
                continue

            seen.add(clean_url)
            normalized_links.append({
                "label": label,
                "url": clean_url,
            })

    now = time.time()

    with sessions_lock:
        session = user_sessions.get(sender)

        if not normalized_links:
            if session:
                session.setdefault("state", {}).pop("pending_video_links", None)
                session.setdefault("state", {}).pop("pending_video_created_at", None)
            return

        if sender not in user_sessions:
            user_sessions[sender] = {
                "history": [],
                "state": {},
                "last_updated": now,
            }

        session = user_sessions[sender]
        session["last_updated"] = now
        session.setdefault("state", {})["pending_video_links"] = normalized_links[:5]
        session.setdefault("state", {})["pending_video_created_at"] = now


def set_pending_video_links_for_records(
    sender: str,
    records: List[Dict[str, Any]],
    schema: Dict[str, str],
    columns: List[str],
) -> None:
    links = extract_video_links_from_records(records, schema, columns)
    set_pending_video_links(sender, links)


def is_affirmative_video_request(text: str) -> bool:
    norm = normalize_text(text)

    affirmative_exact = {
        "yes",
        "yes please",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "please",
        "send",
        "send it",
        "send me",
        "send video",
        "send the video",
        "share",
        "share it",
        "share video",
        "share the video",
    }

    if norm in affirmative_exact:
        return True

    phrases = [
        "yes please",
        "send it",
        "send me",
        "send the video",
        "send video",
        "share it",
        "share the video",
        "share video",
        "video tour",
        "tour video",
    ]

    return any(phrase in norm for phrase in phrases)


def consume_pending_video_reply(sender: str, user_text: str) -> str:
    """
    If the user says yes after a video offer, send the stored video link(s).
    """
    if not is_affirmative_video_request(user_text):
        return ""

    with sessions_lock:
        session = user_sessions.get(sender)

        if not session:
            return ""

        state = session.setdefault("state", {})
        links = state.get("pending_video_links") or []
        created_at = state.get("pending_video_created_at", 0)

        if not links:
            return ""

        if created_at and time.time() - created_at > SESSION_TIMEOUT_SECONDS:
            state.pop("pending_video_links", None)
            state.pop("pending_video_created_at", None)
            return ""

        # Consume once.
        state.pop("pending_video_links", None)
        state.pop("pending_video_created_at", None)

    if len(links) == 1:
        return (
            "Of course — here is the high-quality video tour:\n\n"
            f"{links[0]['url']}\n\n"
            "Would you like me to arrange a viewing for you as well?"
        )

    lines = ["Of course — here are the available video tours:\n"]

    for index, item in enumerate(links, start=1):
        label = clean_cell(item.get("label", f"Property {index}"))
        url = clean_cell(item.get("url", ""))

        if not url:
            continue

        lines.append(f"{index}. *{label}*\n{url}")

    lines.append("\nWould you like me to arrange a viewing for any of these?")

    return "\n\n".join(lines)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distance between two latitude/longitude points in meters.
    """
    radius_earth_m = 6371000

    phi1 = lat1 * 3.141592653589793 / 180
    phi2 = lat2 * 3.141592653589793 / 180
    d_phi = (lat2 - lat1) * 3.141592653589793 / 180
    d_lambda = (lon2 - lon1) * 3.141592653589793 / 180

    a = (
        (math_sin(d_phi / 2) ** 2)
        + math_cos(phi1) * math_cos(phi2) * (math_sin(d_lambda / 2) ** 2)
    )

    c = 2 * math_atan2(a ** 0.5, (1 - a) ** 0.5)
    return radius_earth_m * c


def math_sin(value: float) -> float:
    return math.sin(value)


def math_cos(value: float) -> float:
    return math.cos(value)


def math_atan2(y: float, x: float) -> float:
    return math.atan2(y, x)


def places_cache_get(key: str) -> Any:
    now = time.time()

    with places_cache_lock:
        item = places_cache["items"].get(key)

        if not item:
            return None

        if now - item.get("loaded_at", 0) > GOOGLE_PLACES_CACHE_TTL_SECONDS:
            places_cache["items"].pop(key, None)
            return None

        return item.get("value")


def places_cache_set(key: str, value: Any) -> None:
    with places_cache_lock:
        places_cache["items"][key] = {
            "loaded_at": time.time(),
            "value": value,
        }


def make_places_cache_key(*parts: Any) -> str:
    return json.dumps(parts, ensure_ascii=False, sort_keys=False, default=str)


def resolve_landmark_to_coordinates(landmark_keywords: str) -> Optional[Dict[str, Any]]:
    """
    Converts landmark keywords/building/area text into coordinates using
    Google Places Find Place from Text.
    """
    landmark_keywords = clean_cell(landmark_keywords)

    if not GOOGLE_PLACES_API_KEY or not landmark_keywords:
        return None

    query = landmark_keywords

    if "dubai" not in normalize_text(query):
        query = f"{query}, Dubai, UAE"

    cache_key = make_places_cache_key("findplace", query)
    cached = places_cache_get(cache_key)

    if cached is not None:
        return cached

    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"

    params = {
        "key": GOOGLE_PLACES_API_KEY,
        "input": query,
        "inputtype": "textquery",
        "fields": "name,geometry,formatted_address,place_id",
    }

    try:
        response = http_session.get(url, params=params, timeout=20)
        data = response.json()
    except Exception as error:
        print("Google Places Find Place error:", error)
        traceback.print_exc()
        return None

    status = data.get("status")

    if status != "OK":
        if status not in {"ZERO_RESULTS"}:
            print("Google Places Find Place status:", status, data.get("error_message", ""))
        places_cache_set(cache_key, None)
        return None

    candidates = data.get("candidates") or []

    if not candidates:
        places_cache_set(cache_key, None)
        return None

    candidate = candidates[0]
    location = ((candidate.get("geometry") or {}).get("location") or {})

    lat = location.get("lat")
    lng = location.get("lng")

    if lat is None or lng is None:
        places_cache_set(cache_key, None)
        return None

    result = {
        "lat": float(lat),
        "lng": float(lng),
        "name": clean_cell(candidate.get("name", "")),
        "formatted_address": clean_cell(candidate.get("formatted_address", "")),
        "place_id": clean_cell(candidate.get("place_id", "")),
    }

    places_cache_set(cache_key, result)
    return result


GOOGLE_PLACE_TYPE_MAP = {
    "school": ["school"],
    "metro_station": ["subway_station", "train_station", "transit_station"],
    "shopping_mall": ["shopping_mall"],
    "park": ["park"],
    "supermarket": ["supermarket"],
}


def get_nearby_places(
    landmark_keywords: str,
    place_type: str,
    radius_meters: int = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Real Google Places Nearby Search implementation.

    Uses:
    1. Find Place from Text to resolve landmark keywords into coordinates.
    2. Nearby Search to find verified amenities within radius.
    """
    landmark_keywords = clean_cell(landmark_keywords)
    place_type = clean_cell(place_type)

    if not GOOGLE_PLACES_API_KEY:
        return []

    if place_type not in GOOGLE_PLACE_TYPE_MAP:
        return []

    try:
        radius = int(radius_meters or GOOGLE_PLACES_RADIUS_DEFAULT)
    except (TypeError, ValueError):
        radius = GOOGLE_PLACES_RADIUS_DEFAULT

    radius = max(1, min(radius, 50000))

    lat = parse_float_cell(latitude)
    lng = parse_float_cell(longitude)

    origin_name = landmark_keywords

    if lat is None or lng is None:
        origin = resolve_landmark_to_coordinates(landmark_keywords)

        if not origin:
            return []

        lat = origin["lat"]
        lng = origin["lng"]
        origin_name = origin.get("name") or landmark_keywords

    cache_key = make_places_cache_key(
        "nearby",
        round(float(lat), 6),
        round(float(lng), 6),
        place_type,
        radius,
    )

    cached = places_cache_get(cache_key)

    if cached is not None:
        return cached

    nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    results: List[Dict[str, Any]] = []
    seen = set()

    for google_type in GOOGLE_PLACE_TYPE_MAP[place_type]:
        params = {
            "key": GOOGLE_PLACES_API_KEY,
            "location": f"{lat},{lng}",
            "radius": radius,
            "type": google_type,
        }

        # Helps Dubai Metro results when Google classifies stations differently.
        if place_type == "metro_station":
            params["keyword"] = "Dubai Metro"

        try:
            response = http_session.get(nearby_url, params=params, timeout=20)
            data = response.json()
        except Exception as error:
            print("Google Places Nearby Search error:", error)
            traceback.print_exc()
            continue

        status = data.get("status")

        if status not in {"OK", "ZERO_RESULTS"}:
            print("Google Places Nearby Search status:", status, data.get("error_message", ""))
            continue

        for place in data.get("results", []) or []:
            name = clean_cell(place.get("name", ""))

            if not name:
                continue

            place_id = clean_cell(place.get("place_id", "")) or searchable_text(name)

            if place_id in seen:
                continue

            seen.add(place_id)

            place_location = ((place.get("geometry") or {}).get("location") or {})
            place_lat = place_location.get("lat")
            place_lng = place_location.get("lng")

            distance_m = None

            if place_lat is not None and place_lng is not None:
                distance_m = int(round(haversine_m(float(lat), float(lng), float(place_lat), float(place_lng))))

            results.append({
                "name": name,
                "type": place_type,
                "google_type": google_type,
                "distance_m": distance_m,
                "address": clean_cell(place.get("vicinity", "")),
                "place_id": clean_cell(place.get("place_id", "")),
                "origin": origin_name,
            })

    results.sort(
        key=lambda item: (
            item["distance_m"] is None,
            item["distance_m"] if item["distance_m"] is not None else 999999,
            item["name"],
        )
    )

    final_results = results[:MAX_NEARBY_PLACES_PER_TYPE]
    places_cache_set(cache_key, final_results)

    return final_results


AMENITY_TYPES_TO_SHOW = [
    "metro_station",
    "school",
    "shopping_mall",
    "park",
    "supermarket",
]

AMENITY_LABELS = {
    "metro_station": "Metro",
    "school": "School",
    "shopping_mall": "Mall",
    "park": "Park",
    "supermarket": "Supermarket",
}


def format_distance(distance_m: Any) -> str:
    if distance_m is None:
        return ""

    try:
        distance = float(distance_m)
    except (TypeError, ValueError):
        return ""

    if distance < 1000:
        return f"{int(round(distance))}m"

    return f"{distance / 1000:.1f}km"


def get_record_coordinates(
    record: Dict[str, Any],
    schema: Dict[str, str],
) -> Tuple[Optional[float], Optional[float]]:
    lat = parse_float_cell(get_record_value_by_field(record, schema, "latitude"))
    lng = parse_float_cell(get_record_value_by_field(record, schema, "longitude"))
    return lat, lng


def get_record_landmark_keywords(
    record: Dict[str, Any],
    schema: Dict[str, str],
) -> str:
    landmark = get_record_value_by_field(record, schema, "landmark_keywords")

    if landmark:
        return landmark

    building = get_record_value_by_field(record, schema, "building")
    location = get_record_value_by_field(record, schema, "location")

    parts = []

    if building:
        parts.append(building)

    if location and searchable_text(location) not in searchable_text(" ".join(parts)):
        parts.append(location)

    if parts:
        joined = ", ".join(parts)

        if "dubai" not in normalize_text(joined):
            joined += ", Dubai, UAE"

        return joined

    return ""


def build_nearby_amenity_lines(
    record: Dict[str, Any],
    schema: Dict[str, str],
) -> List[str]:
    """
    Builds verified nearby amenity lines from Google Places only.
    If Places is not configured or returns nothing, this stays silent.
    """
    if not GOOGLE_PLACES_API_KEY:
        return []

    landmark_keywords = get_record_landmark_keywords(record, schema)
    lat, lng = get_record_coordinates(record, schema)

    if not landmark_keywords and (lat is None or lng is None):
        return []

    amenity_lines = []

    for place_type in AMENITY_TYPES_TO_SHOW:
        try:
            places = get_nearby_places(
                landmark_keywords=landmark_keywords,
                place_type=place_type,
                radius_meters=GOOGLE_PLACES_RADIUS_DEFAULT,
                latitude=lat,
                longitude=lng,
            )
        except Exception as error:
            print("Amenity lookup error:", error)
            traceback.print_exc()
            places = []

        if not places:
            continue

        nearest = places[0]
        name = clean_cell(nearest.get("name", ""))
        distance = format_distance(nearest.get("distance_m"))

        if not name:
            continue

        label = AMENITY_LABELS.get(place_type, place_type)

        if distance:
            amenity_lines.append(f"{label}: {name} ({distance})")
        else:
            amenity_lines.append(f"{label}: {name}")

    if not amenity_lines:
        return []

    lines = ["   • *Nearby verified amenities:*"]

    for item in amenity_lines:
        lines.append(f"     - {item}")

    return lines


def listing_to_tool_dict(
    record: Dict[str, Any],
    schema: Dict[str, str],
    columns: List[str],
) -> Dict[str, Any]:
    """
    Converts a sheet row into a clean tool result for the Responses API.
    """
    raw_details = {}

    for column in ordered_columns_for_output(columns, schema):
        if column == INTERNAL_SEARCH_KEY:
            continue

        if should_hide_client_column(column):
            continue

        value = clean_cell(record.get(column, ""))

        if value:
            raw_details[column] = value

    listing_id = get_record_value_by_field(record, schema, "id")

    if not listing_id:
        fingerprint = json.dumps(raw_details, ensure_ascii=False, sort_keys=True)
        listing_id = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]

    price_display = get_record_value_by_field(record, schema, "price")
    price_aed = parse_money_value(price_display)

    status = get_record_value_by_field(record, schema, "status")
    rental_yield = get_record_value_by_field(record, schema, "rental_yield")
    landmark_keywords = get_record_landmark_keywords(record, schema)
    video_text = get_record_value_by_field(record, schema, "video_link")
    video_urls = extract_video_urls(video_text)

    lat, lng = get_record_coordinates(record, schema)

    result = {
        "id": listing_id,
        "building": get_record_value_by_field(record, schema, "building"),
        "area": get_record_value_by_field(record, schema, "location"),
        "unit_type": get_record_value_by_field(record, schema, "unit_type"),
        "unit_no": get_record_value_by_field(record, schema, "unit_no"),
        "price_aed": price_aed,
        "price_display": price_display,
        "availability": status,
        "rental_yield": rental_yield,
        "landmark_keywords": landmark_keywords,
        "video_link": video_urls[0] if video_urls else "",
        "raw_details": raw_details,
    }

    if lat is not None and lng is not None:
        result["latitude"] = lat
        result["longitude"] = lng

    return result


def search_listings(
    area: str,
    property_type: str = None,
    bedrooms: int = None,
    budget_min_aed: float = None,
    budget_max_aed: float = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Responses API tool backed by your existing Google Sheet loader/search.

    This is intentionally grounded:
    - loads the current cached sheet
    - applies your deterministic filters
    - returns only real sheet rows
    """
    records, schema, columns = get_properties()

    if not records:
        return []

    area = clean_cell(area)
    property_type = clean_cell(property_type)

    query_parts = []

    if bedrooms:
        query_parts.append(f"{int(bedrooms)} bedroom")

    if property_type:
        query_parts.append(property_type)

    if area:
        query_parts.append(f"in {area}")

    min_budget = to_optional_float(budget_min_aed)
    max_budget = to_optional_float(budget_max_aed)

    if min_budget is not None:
        query_parts.append(f"above {min_budget:,.0f} AED")

    if max_budget is not None:
        query_parts.append(f"under {max_budget:,.0f} AED")

    tool_query = " ".join(query_parts).strip() or area

    filters = extract_filters_from_text(
        user_text=tool_query,
        records=records,
        schema=schema,
    )

    # Area/building mapping using the exact area argument too.
    if area:
        area_filters = extract_filters_from_text(
            user_text=area,
            records=records,
            schema=schema,
        )

        for key in ("location", "building"):
            if area_filters.get(key):
                filters[key] = area_filters[key]

    if bedrooms:
        filters["unit_type"] = f"{int(bedrooms)} BR"

    if min_budget is not None:
        filters["min_price"] = min_budget

    if max_budget is not None:
        filters["max_price"] = max_budget

    if DEFAULT_AVAILABLE_ONLY:
        filters["available_only"] = True

    matches = search_properties(
        user_text=tool_query,
        records=records,
        schema=schema,
        filters=filters,
        fallback_search_text=area or tool_query,
    )

    # Apply stricter property type filtering for property types that should
    # not accidentally return normal apartments.
    ptype = normalize_text(property_type)

    if ptype in {"villa", "townhouse", "penthouse"}:
        matches = [
            record for record in matches
            if phrase_in_text(ptype, row_search_text(record))
        ]
    elif ptype == "apartment":
        strict_apartment_matches = [
            record for record in matches
            if phrase_in_text("apartment", row_search_text(record))
        ]

        # Many sheets imply apartments by bedroom/unit type and do not write
        # the word apartment. Only enforce if explicit apartment rows exist.
        if strict_apartment_matches:
            matches = strict_apartment_matches

    matches = dedupe_records(matches)
    matches = matches[:MAX_TOOL_LISTINGS_TO_RETURN]

    return [
        listing_to_tool_dict(record, schema, columns)
        for record in matches
    ]


FUNCTIONS = {
    "search_listings": search_listings,
    "get_nearby_places": get_nearby_places,
}


def ask_consultant(
    user_message: str,
    conversation_input: Optional[List[Dict[str, Any]]] = None,
    return_metadata: bool = False,
):
    """
    OpenAI Responses API consultant loop with real tool execution.

    Returns:
    - default: (reply_text, updated_conversation_input)
    - if return_metadata=True:
      (reply_text, updated_conversation_input, metadata)
    """
    if not client:
        fallback = (
            "Sorry, the AI service is not configured right now. "
            "For more information, please connect with Mr. Zahid at +971562625777."
        )
        if return_metadata:
            return fallback, conversation_input or [], {"video_links": []}
        return fallback, conversation_input or []

    input_items = list(conversation_input or [])
    input_items.append({
        "role": "user",
        "content": user_message,
    })

    metadata = {
        "video_links": [],
    }

    max_tool_rounds = 6

    for _ in range(max_tool_rounds):
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=input_items,
            tools=TOOLS,
        )

        tool_calls = [
            item for item in response.output
            if getattr(item, "type", "") == "function_call"
        ]

        if not tool_calls:
            reply = clean_cell(response.output_text)

            if reply:
                input_items.append({
                    "role": "assistant",
                    "content": reply,
                })

            if return_metadata:
                return reply, input_items, metadata

            return reply, input_items

        for item in tool_calls:
            try:
                input_items.append(item.model_dump(exclude_none=True))
            except Exception:
                input_items.append({
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                })

            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            try:
                result = FUNCTIONS[item.name](**args)
            except Exception as error:
                print(f"Tool execution error for {item.name}:", error)
                traceback.print_exc()
                result = {
                    "error": f"{item.name} failed. Please try again or contact Mr. Zahid.",
                }

            if item.name == "search_listings" and isinstance(result, list):
                for listing in result:
                    video_link = clean_cell(listing.get("video_link", ""))

                    for url in extract_video_urls(video_link):
                        metadata["video_links"].append({
                            "label": clean_cell(
                                listing.get("building")
                                or listing.get("unit_type")
                                or listing.get("id")
                                or "Property video tour"
                            ),
                            "url": url,
                        })

            input_items.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result, ensure_ascii=False, default=str),
            })

    fallback = (
        "I’m sorry, I could not complete the live lookup in time. "
        "For immediate assistance, please connect with Mr. Zahid at +971562625777."
    )

    if return_metadata:
        return fallback, input_items, metadata

    return fallback, input_items


# ============================================================
# Property Result Formatting
# ============================================================

def ordered_columns_for_output(columns: List[str], schema: Dict[str, str]) -> List[str]:
    preferred_fields = [
        "building",
        "location",
        "unit_type",
        "unit_no",
        "price",
        "size",
        "status",
        "description",
    ]

    ordered = []

    for field in preferred_fields:
        column = schema.get(field)
        if column and column in columns and column not in ordered:
            ordered.append(column)

    for column in columns:
        if column not in ordered:
            ordered.append(column)

    return ordered


def describe_filters(filters: Dict[str, Any]) -> str:
    parts = []

    if filters.get("location"):
        parts.append(str(filters["location"]))

    if filters.get("building"):
        parts.append(str(filters["building"]))

    if filters.get("unit_type"):
        parts.append(str(filters["unit_type"]))

    min_price = filters.get("min_price")
    max_price = filters.get("max_price")
    if min_price is not None and max_price is not None:
        parts.append(f"{min_price:,.0f}-{max_price:,.0f}")
    elif max_price is not None:
        parts.append(f"under {max_price:,.0f}")
    elif min_price is not None:
        parts.append(f"above {min_price:,.0f}")

    if filters.get("available_only"):
        parts.append("available units")

    return " / ".join(parts)


def format_property_results(
    records: List[Dict[str, Any]],
    schema: Dict[str, str],
    columns: List[str],
    filters: Dict[str, Any],
    user_text: str,
) -> str:
    records = dedupe_records(records)

    if not records:
        return (
            "I could not find an exact match in the current property sheet.\n\n"
            "For more information, please connect with Mr. Zahid at +971562625777."
        )

    total_count = len(records)
    visible_records = records[:MAX_PROPERTIES_TO_SHOW]
    count = len(visible_records)
    filter_description = describe_filters(filters)

    intro = f"Sure — I found *{total_count}* matching record"
    intro += "" if total_count == 1 else "s"

    if filter_description:
        intro += f" for *{filter_description}*"

    if total_count > count:
        intro += f".\n\nHere are the top *{count}*:"
    else:
        intro += ".\n\nHere are the complete details from our current sheet:"

    lines = [intro]

    building_column = schema.get("building", "")
    unit_type_column = schema.get("unit_type", "")
    unit_no_column = schema.get("unit_no", "")
    price_column = schema.get("price", "")

    grouped = OrderedDict()

    for record in visible_records:
        building_name = ""

        if building_column:
            building_name = clean_cell(record.get(building_column, ""))

        if not building_name:
            building_name = "Matching Properties"

        grouped.setdefault(building_name, []).append(record)

    output_columns = ordered_columns_for_output(columns, schema)
    global_index = 1

    for building_name, group_records in grouped.items():
        building_header = f"\n🏢 *{building_name}*"

        if len(group_records) > 1:
            building_header += f" — {len(group_records)} entries"

        lines.append(building_header)

        # Verified Google Places amenities, shown once per building/group.
        amenity_lines = build_nearby_amenity_lines(group_records[0], schema)

        if amenity_lines:
            lines.extend(amenity_lines)

        for record in group_records:
            title_parts = []

            if unit_type_column:
                unit_type_value = clean_cell(record.get(unit_type_column, ""))

                if unit_type_value:
                    title_parts.append(unit_type_value)

            if unit_no_column:
                unit_no_value = clean_cell(record.get(unit_no_column, ""))

                if unit_no_value:
                    title_parts.append(f"Unit {unit_no_value}")

            if price_column:
                price_value = clean_cell(record.get(price_column, ""))

                if price_value:
                    title_parts.append(price_value)

            title = " | ".join(title_parts) if title_parts else f"Record {global_index}"

            lines.append(f"\n{global_index}. *{title}*")

            for column in output_columns:
                col_header = str(column).strip()

                if should_hide_client_column(col_header):
                    continue

                value = clean_cell(record.get(column, ""))

                if not value:
                    continue

                col_norm = normalize_header(col_header)
                display_column = col_header

                if "new rent" in col_norm or "new price" in col_norm:
                    display_column = "💎 Best Price"

                lines.append(f"   • *{display_column}:* {value}")

            global_index += 1

    if total_count > count:
        lines.append(
            f"\nThere are *{total_count - count}* more matching properties I'm "
            "not showing here to keep this readable. Tell me your preferred "
            "budget, bedrooms, or move-in date and I'll narrow it down."
        )

    video_links = extract_video_links_from_records(visible_records, schema, columns)

    if video_links:
        if len(video_links) == 1:
            lines.append(
                "\n🎥 I also have a stunning, high-quality video tour of this exact "
                "property — would you like me to send it over so you can experience "
                "the full layout right from your phone?"
            )
        else:
            lines.append(
                f"\n🎥 I also have high-quality video tours for *{len(video_links)}* "
                "of these matching properties — would you like me to send them over?"
            )

    lines.append(
        "\nFor viewing, booking, or more information, please connect with "
        "*Mr. Zahid* at +971562625777."
    )

    return "\n".join(lines)


# ============================================================
# Hybrid Understanding: AI-Assisted Intent -> Deterministic Sheet Lookup
# ============================================================
#
# extract_filters_from_text() (regex/keyword based) handles clear, direct
# queries ("2BR in Al Raffa") fast and for free. But casual, broad, or
# complex phrasing ("something small and quiet for a young family, nothing
# crazy on price") doesn't reliably reduce to keywords. ai_understand_query()
# is the upgrade for that case: it asks the AI model to *interpret intent*
# and propose filters -- but it never gets to describe a property itself.
# Its output is validated against the real vocabulary in the sheet and then
# run back through the exact same search_properties()/format_property_results()
# pipeline the deterministic path uses, so whatever the client sees always
# comes from an actual sheet row, never from the model's imagination.

AI_FILTER_EXTRACTION_PROMPT = """
You extract structured search filters from a WhatsApp real estate lead's
message so a server can look up exact matches in a property spreadsheet.
The client's phrasing may be casual, broad, or roundabout -- interpret intent,
don't just look for keywords. Examples of implied filters:
- "near the beach" / "close to the metro" -> a location, if one in the
  vocabulary plausibly matches
- "under 70k", "budget is around 50000", "nothing above 1 lakh" -> max_price
- "starting from 80k", "at least 60000" -> min_price
- "cheapest you have", "most affordable" -> leave price null, but keep
  location/unit_type/available_only if mentioned; the app sorts for cheapest
- "small family", "couple", "just for me" -> a plausible unit_type ONLY if
  it clearly maps to one in the vocabulary (e.g. "just for me" -> Studio),
  otherwise leave unit_type null rather than guessing

Hard rules:
- Only use values that appear in the given vocabulary lists, verbatim or as
  a close match. NEVER invent a location, building, or unit type that isn't
  in the vocabulary. If nothing fits, leave that field null.
- Only set min_price/max_price when the client's own words imply a real
  number -- never invent a figure.
- Respond with ONLY a single JSON object, no other text, no markdown fences,
  matching exactly this shape:
{"location": string or null, "building": string or null, "unit_type": string or null,
 "min_price": number or null, "max_price": number or null, "available_only": boolean}
"""


def ai_understand_query(
    user_question: str,
    past_history: List[Dict[str, str]],
    records: List[Dict[str, Any]],
    schema: Dict[str, str],
) -> Dict[str, Any]:
    """
    Uses the AI model to interpret a casual/complex client message that the
    deterministic matcher couldn't confidently parse, and maps it onto the
    REAL vocabulary present in the sheet (actual locations, buildings, unit
    types) plus any implied budget.

    Returns a filter dict in the same shape extract_filters_from_text()
    produces -- this function never returns property details itself, only
    search filters, which the caller then runs through the normal
    search_properties()/format_property_results() pipeline. That's what
    keeps this grounded: the model proposes *where to look*, code decides
    *what's true*.
    """
    if not client or not records:
        return {}

    location_column = schema.get("location", "")
    building_column = schema.get("building", "")
    unit_type_column = schema.get("unit_type", "")

    known_locations = get_unique_column_values(records, location_column, split_values=True)[:80]
    known_buildings = get_unique_column_values(records, building_column, split_values=False)[:80]
    known_unit_types = get_unique_column_values(records, unit_type_column, split_values=False)[:40]

    # Nothing to map onto -- skip the AI call entirely rather than let it
    # guess with no real vocabulary to ground against.
    if not (known_locations or known_buildings or known_unit_types):
        return {}

    vocabulary = {
        "known_locations": known_locations,
        "known_buildings": known_buildings,
        "known_unit_types": known_unit_types,
    }

    user_content = (
        f"Vocabulary available in the sheet:\n{json.dumps(vocabulary, ensure_ascii=False)}\n\n"
        f"Recent conversation (oldest first):\n"
        f"{json.dumps(past_history[-6:], ensure_ascii=False)}\n\n"
        f"Client's latest message: {user_question}"
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": AI_FILTER_EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
    except Exception as error:
        print("ai_understand_query error:", error)
        traceback.print_exc()
        return {}

    filters: Dict[str, Any] = {}

    # Validate against the real vocabulary -- never trust the model's string
    # verbatim without confirming it actually resolves to a real sheet value.
    raw_location = parsed.get("location")
    if raw_location:
        matched_location = best_value_match(str(raw_location), known_locations, mode="location")
        if matched_location:
            filters["location"] = matched_location

    raw_building = parsed.get("building")
    if raw_building:
        matched_building = best_value_match(str(raw_building), known_buildings, mode="partial")
        if matched_building:
            filters["building"] = matched_building

    raw_unit_type = parsed.get("unit_type")
    if raw_unit_type:
        canonical = canonical_unit_type(str(raw_unit_type))
        if canonical:
            filters["unit_type"] = canonical

    for key in ("min_price", "max_price"):
        value = parsed.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            filters[key] = float(value)

    if parsed.get("available_only") is True:
        filters["available_only"] = True

    return filters


# ============================================================
# OpenAI General Conversation
# ============================================================

GENERAL_SYSTEM_PROMPT = """
You are a professional, premium real estate assistant in Dubai.

Style:
- Reply naturally like a high-end WhatsApp real estate consultant.
- Be polite, calm, concise, and confidence-inspiring.
- Keep replies scannable. Avoid large blocks of text.
- Use conversation history to understand follow-up questions.

Strict rules:
- Do not invent property availability, prices, unit numbers, sizes, yields, or locations.
- If the user asks about exact inventory, the software should search the sheet first.
- Use only provided company knowledge and sheet coverage summary.
- If information is not available, say so clearly and suggest:
  "For more information, please connect with Mr. Zahid at +971562625777."

Honesty:
- If the client directly asks whether they are speaking with a human or AI, answer truthfully:
  you are an AI assistant helping the real estate team.
"""


def summarize_sheet_for_ai(records: List[Dict[str, Any]], schema: Dict[str, str]) -> str:
    """
    A compact, cheap-to-send summary of what's actually in the sheet right
    now (coverage, not individual listings). Lets the general fallback
    reply be grounded ("we don't have anything in X, but we do cover Y/Z")
    instead of totally blind, without shipping the whole sheet as context
    or risking it inventing unit-level specifics.
    """
    if not records:
        return "The property sheet is currently empty or unavailable."

    location_column = schema.get("location", "")
    unit_type_column = schema.get("unit_type", "")
    price_column = schema.get("price", "")

    locations = get_unique_column_values(records, location_column, split_values=True)[:25]
    unit_types = get_unique_column_values(records, unit_type_column, split_values=False)[:15]

    prices = [
        parse_money_value(record.get(price_column, ""))
        for record in records
        if price_column
    ]
    prices = [p for p in prices if p is not None]

    lines = [f"Total listings currently in the sheet: {len(records)}"]

    if locations:
        lines.append(f"Locations covered: {', '.join(locations)}")

    if unit_types:
        lines.append(f"Unit types available: {', '.join(unit_types)}")

    if prices:
        lines.append(f"Price range across all listings: {min(prices):,.0f} - {max(prices):,.0f}")

    return "\n".join(lines)


def create_general_ai_reply(
    user_question: str,
    past_history: List[Dict[str, str]],
    records: List[Dict[str, Any]] = None,
    schema: Dict[str, str] = None,
) -> str:
    """
    Used only when no property match is found (deterministic or AI-assisted)
    and the client is having a more general conversation. Individual
    property replies stay fully deterministic (format_property_results) to
    avoid missing/invented rows -- this function only ever gets a coverage
    *summary*, never row-level data, so it can be grounded without being
    able to state specific unit facts it shouldn't.
    """
    if not client:
        return (
            "Hello! I can help you with Dubai real estate options. "
            "Please send me a location, building name, or unit type, and I will check the available details."
        )

    knowledge = get_knowledge()
    sheet_summary = summarize_sheet_for_ai(records or [], schema or {})

    messages = [
        {
            "role": "system",
            "content": GENERAL_SYSTEM_PROMPT,
        },
        {
            "role": "system",
            "content": (
                "Company knowledge JSON. Use only if relevant:\n"
                f"{json.dumps(knowledge, ensure_ascii=False)}"
            ),
        },
        {
            "role": "system",
            "content": (
                "Current sheet coverage summary (use this to answer general "
                "questions like 'what areas do you cover' or 'what's your price "
                "range' accurately -- but never state a specific unit's price, "
                "unit number, or availability from this summary; that always "
                "requires an actual sheet lookup):\n"
                f"{sheet_summary}"
            ),
        },
    ]

    messages.extend(past_history[-MAX_HISTORY_MESSAGES:])
    messages.append({
        "role": "user",
        "content": user_question,
    })

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=1,
            messages=messages,
        )

        reply = response.choices[0].message.content or ""
        return reply.strip()

    except Exception as error:
        print("OpenAI error:", error)
        traceback.print_exc()

        return (
            "Sorry, I had a small technical issue. "
            "Please send me the location or building name again, or connect with "
            "Mr. Zahid at +971562625777."
        )


# ============================================================
# Main Reply Creation
# ============================================================

def create_ai_reply(sender: str, user_question: str) -> str:
    user_question = clean_cell(user_question)

    if not user_question:
        return ""

    if is_reset_command(user_question):
        reset_session(sender)
        return (
            "Sure — I have reset our chat. "
            "Please send me a location, building name, or unit type, and I will check the details."
        )

    # If the previous property reply offered a video tour, only now send the link
    # when the client gives an affirmative response.
    video_reply = consume_pending_video_reply(sender, user_question)

    if video_reply:
        update_session(
            sender=sender,
            user_text=user_question,
            assistant_text=video_reply,
            filters={},
            search_text=user_question,
            matched=False,
        )
        return video_reply

    past_history, previous_state = get_session_snapshot(sender)

    # Optional full Responses API consultant mode.
    # Keep ENABLE_RESPONSES_CONSULTANT=false first, then test before enabling.
    if ENABLE_RESPONSES_CONSULTANT and client:
        try:
            response_history = [
                {
                    "role": item["role"],
                    "content": item["content"],
                }
                for item in past_history[-MAX_HISTORY_MESSAGES:]
                if item.get("role") in {"user", "assistant"} and clean_cell(item.get("content", ""))
            ]

            reply, _, metadata = ask_consultant(
                user_message=user_question,
                conversation_input=response_history,
                return_metadata=True,
            )

            if metadata.get("video_links"):
                set_pending_video_links(sender, metadata["video_links"])

            if reply:
                update_session(
                    sender=sender,
                    user_text=user_question,
                    assistant_text=reply,
                    filters={},
                    search_text=user_question,
                    matched=False,
                )
                return reply

        except Exception as error:
            print("Responses API consultant error, falling back to deterministic flow:", error)
            traceback.print_exc()

    # Whole-word check so words like "now" or "renovated" don't get mistaken
    # for "no"/"not" and wipe the session unexpectedly.
    neg_words = {"not", "no", "nahi", "other", "except", "change", "dusra"}

    if neg_words.intersection(meaningful_tokens(user_question)):
        with sessions_lock:
            if sender in user_sessions:
                user_sessions[sender]["state"] = {}
                previous_state = {}

    records, schema, columns = get_properties()

    current_filters = extract_filters_from_text(
        user_text=user_question,
        records=records,
        schema=schema,
    ) if records else {}

    if current_filters.get("location") or current_filters.get("building"):
        previous_state = {}

    effective_filters, search_text = build_effective_filters(
        current_filters=current_filters,
        previous_state=previous_state,
        user_text=user_question,
    )

    matches = search_properties(
        user_text=user_question,
        records=records,
        schema=schema,
        filters=effective_filters,
        fallback_search_text=search_text,
    ) if records else []

    if matches:
        visible_matches = dedupe_records(matches)[:MAX_PROPERTIES_TO_SHOW]
        set_pending_video_links_for_records(sender, visible_matches, schema, columns)

        reply = format_property_results(
            records=matches,
            schema=schema,
            columns=columns,
            filters=effective_filters,
            user_text=user_question,
        )

        update_session(
            sender=sender,
            user_text=user_question,
            assistant_text=reply,
            filters=effective_filters,
            search_text=search_text or user_question,
            matched=True,
        )

        return reply

    # Deterministic keyword/regex matching found nothing usable.
    # Let AI interpret broad/casual phrasing, but only into filters.
    ai_filters: Dict[str, Any] = {}

    if records:
        ai_filters = ai_understand_query(user_question, past_history, records, schema)

        if ai_filters:
            merged_filters = {**effective_filters, **ai_filters}

            ai_matches = search_properties(
                user_text=user_question,
                records=records,
                schema=schema,
                filters=merged_filters,
                fallback_search_text=search_text,
            )

            if ai_matches:
                visible_ai_matches = dedupe_records(ai_matches)[:MAX_PROPERTIES_TO_SHOW]
                set_pending_video_links_for_records(sender, visible_ai_matches, schema, columns)

                reply = format_property_results(
                    records=ai_matches,
                    schema=schema,
                    columns=columns,
                    filters=merged_filters,
                    user_text=user_question,
                )

                update_session(
                    sender=sender,
                    user_text=user_question,
                    assistant_text=reply,
                    filters=merged_filters,
                    search_text=search_text or user_question,
                    matched=True,
                )

                return reply

    if is_likely_property_query(user_question, effective_filters or current_filters, previous_state) or ai_filters:
        # New property query with no match; clear stale video links from previous result.
        set_pending_video_links(sender, [])

        reply = (
            "I could not find an exact match in the current property sheet for that query.\n\n"
            "For more information, please connect with Mr. Zahid at +971562625777."
        )

        update_session(
            sender=sender,
            user_text=user_question,
            assistant_text=reply,
            filters=effective_filters,
            search_text=search_text or user_question,
            matched=False,
        )

        return reply

    # General conversational response.
    reply = create_general_ai_reply(user_question, past_history, records, schema)

    update_session(
        sender=sender,
        user_text=user_question,
        assistant_text=reply,
        filters=current_filters,
        search_text=user_question,
        matched=False,
    )

    return reply


# ============================================================
# WhatsApp Sending
# ============================================================

def split_whatsapp_message(message: str, limit: int = WHATSAPP_TEXT_LIMIT) -> List[str]:
    """
    Splits long messages safely for WhatsApp.
    """
    text = clean_cell(message)

    if not text:
        return []

    if len(text) <= limit:
        return [text]

    chunks = []

    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)

        if cut < int(limit * 0.4):
            cut = text.rfind("\n", 0, limit)

        if cut < int(limit * 0.4):
            cut = text.rfind(" ", 0, limit)

        if cut < int(limit * 0.4):
            cut = limit

        chunks.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        chunks.append(text)

    if len(chunks) > 1:
        total = len(chunks)
        chunks = [
            f"({index + 1}/{total})\n{chunk}"
            for index, chunk in enumerate(chunks)
        ]

    return chunks


def send_whatsapp_message(to: str, message: str) -> None:
    if not META_TOKEN or not PHONE_NUMBER_ID:
        print("META_TOKEN or PHONE_NUMBER_ID is not configured. Cannot send WhatsApp message.")
        return

    chunks = split_whatsapp_message(message)

    if not chunks:
        return

    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {META_TOKEN}",
        "Content-Type": "application/json",
    }

    for chunk in chunks:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": chunk,
            },
        }

        try:
            result = http_session.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )

            print("WhatsApp send:", result.status_code, result.text[:1000])

        except Exception as error:
            print("WhatsApp send error:", error)
            traceback.print_exc()

        # Small pause so multipart messages arrive in order.
        time.sleep(0.2)


# ============================================================
# Webhook De-Duplication (persistent -- survives restarts)
# ============================================================
#
# Backed by SQLite instead of an in-memory dict. An in-memory dict is wiped
# on every restart, so a message Meta redelivers after the server was
# asleep/restarting would look brand new and get processed (and replied to)
# a second time. A row in this file surviving the restart is what actually
# prevents that. See REACTIVE-ONLY GUARANTEE near the top of this file.

def get_dedup_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DEDUP_DB_PATH, timeout=10)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_messages (
            message_id TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            status TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    return conn


def cleanup_seen_message_ids(conn: sqlite3.Connection, now: float) -> None:
    conn.execute(
        "DELETE FROM seen_messages WHERE ts < ?",
        (now - MESSAGE_ID_TTL_SECONDS,),
    )
    conn.commit()


def mark_message_seen(message_id: str) -> bool:
    """
    Returns:
    - True if message is new and should be processed.
    - False if it is a duplicate Meta retry (or a redelivery after a
      restart) and must be ignored.
    """
    now = time.time()

    with seen_lock:
        conn = get_dedup_connection()
        try:
            cleanup_seen_message_ids(conn, now)

            try:
                conn.execute(
                    "INSERT INTO seen_messages (message_id, ts, status, updated_at) "
                    "VALUES (?, ?, 'queued', ?)",
                    (message_id, now, now),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Already have this message_id -- genuine duplicate/retry.
                return False
        finally:
            conn.close()


def update_message_status(message_id: str, status: str) -> None:
    now = time.time()

    with seen_lock:
        conn = get_dedup_connection()
        try:
            conn.execute(
                "UPDATE seen_messages SET status = ?, updated_at = ? WHERE message_id = ?",
                (status, now, message_id),
            )
            conn.commit()
        finally:
            conn.close()


def stable_message_id(message: Dict[str, Any], sender: str, text: str) -> str:
    """
    WhatsApp usually provides message["id"].
    This fallback is only for safety.
    """
    message_id = clean_cell(message.get("id", ""))

    if message_id:
        return message_id

    timestamp = clean_cell(message.get("timestamp", ""))
    raw = f"{sender}|{timestamp}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()



# ============================================================
# Background Processing
# ============================================================

def process_message_background(sender: str, message_id: str, user_text: str) -> None:
    """
    Runs after the webhook has already returned 200 OK to Meta.

    This prevents Meta retry loops caused by slow:
    - Google Sheet loading
    - OpenAI API calls
    - WhatsApp send calls
    """
    user_lock = get_user_lock(sender)

    with user_lock:
        try:
            update_message_status(message_id, "processing")

            reply = create_ai_reply(sender, user_text)

            if reply:
                send_whatsapp_message(sender, reply)

            update_message_status(message_id, "done")

        except Exception as error:
            print("Background processing error:", error)
            traceback.print_exc()

            # Send one fallback message only. Do not remove message_id from seen cache,
            # otherwise a Meta retry could create duplicate replies.
            try:
                send_whatsapp_message(
                    sender,
                    (
                        "Sorry, I had a technical issue while checking that. "
                        "For quick assistance, please connect with Mr. Zahid at +971562625777."
                    ),
                )
            except Exception:
                traceback.print_exc()

            update_message_status(message_id, "failed")


# ============================================================
# FastAPI Routes
# ============================================================

@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "service": "WhatsApp Real Estate Bot",
    }


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge or "")

    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Critical behavior:
    - Parse incoming payload quickly.
    - Queue real processing in BackgroundTasks.
    - Immediately return 200 OK to Meta.
    - Deduplicate message IDs (persistently -- see mark_message_seen) to
      prevent retry-loop duplicate replies.
    - Drop anything older than MAX_MESSAGE_AGE_SECONDS with no reply, so a
      batch of delayed webhooks after downtime can never trigger outbound
      messages. See REACTIVE-ONLY GUARANTEE near the top of this file.
    """
    try:
        data = await request.json()
    except Exception as error:
        print("Invalid webhook JSON:", error)
        # Return 200 anyway so Meta does not keep retrying bad payloads.
        return {"status": "ignored_invalid_json"}

    queued = 0
    ignored = 0

    try:
        entries = data.get("entry", [])

        for entry in entries:
            changes = entry.get("changes", [])

            for change in changes:
                value = change.get("value", {})

                # Status updates for messages you sent appear here.
                # They do not contain inbound customer text, so ignore silently.
                messages = value.get("messages", []) or []

                if not messages:
                    ignored += 1
                    continue

                for message in messages:
                    message_type = message.get("type")

                    # Stay silent for images, audio, reactions, stickers, etc.
                    if message_type != "text":
                        ignored += 1
                        continue

                    sender = clean_cell(message.get("from", ""))
                    text_obj = message.get("text", {}) or {}
                    user_text = clean_cell(text_obj.get("body", ""))

                    if not sender or not user_text:
                        ignored += 1
                        continue

                    message_id = stable_message_id(message, sender, user_text)

                    if not mark_message_seen(message_id):
                        print(f"Duplicate webhook ignored: {message_id}")
                        ignored += 1
                        continue

                    # Reactive-only safeguard #2 (see REACTIVE-ONLY GUARANTEE
                    # near the top of this file): never reply to a message
                    # that's old enough to be a delayed/retried webhook --
                    # e.g. delivered in a batch right after the server woke
                    # up from being asleep for hours -- rather than a live,
                    # in-the-moment conversation. This check is independent
                    # of mark_message_seen() above, so it still protects the
                    # bot even if dedup history was ever lost.
                    msg_age_seconds = None
                    try:
                        msg_age_seconds = time.time() - float(message.get("timestamp"))
                    except (TypeError, ValueError):
                        pass  # missing/malformed timestamp -- don't block on it

                    if msg_age_seconds is not None and msg_age_seconds > MAX_MESSAGE_AGE_SECONDS:
                        print(
                            f"Ignoring stale message {message_id} "
                            f"(age={msg_age_seconds:.0f}s > {MAX_MESSAGE_AGE_SECONDS}s cutoff) "
                            "-- no reply sent."
                        )
                        update_message_status(message_id, "ignored_stale")
                        ignored += 1
                        continue

                    background_tasks.add_task(
                        process_message_background,
                        sender,
                        message_id,
                        user_text,
                    )

                    queued += 1

    except Exception as error:
        print("Webhook parsing error:", error)
        traceback.print_exc()
        # Still return 200 to avoid Meta retry loop.
        return {
            "status": "ok",
            "queued": queued,
            "ignored": ignored,
            "error_logged": True,
        }

    # Immediate 200 OK response to Meta.
    return {
        "status": "accepted",
        "queued": queued,
        "ignored": ignored,
    }
