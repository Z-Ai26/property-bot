import os
import json
import time
import re
import hashlib
import threading
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

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

# WhatsApp message de-duplication cache.
seen_message_ids: Dict[str, Dict[str, Any]] = {}
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
                if key in {"location", "building", "unit_type", "available_only"} and value
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
        "location",
        "building",
        "unit_type",
        "unit_no",
        "price",
        "size",
        "status",
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
    """
    previous_filters = previous_state.get("last_filters", {}) or {}
    previous_search_text = clean_cell(previous_state.get("last_search_text", ""))

    effective: Dict[str, Any] = {}
    search_text = user_text

    current_location = current_filters.get("location")
    current_building = current_filters.get("building")
    current_unit_type = current_filters.get("unit_type")

    if current_location:
        # New location should reset old building/unit unless explicitly present now.
        effective["location"] = current_location

        if current_building:
            effective["building"] = current_building

        if current_unit_type:
            effective["unit_type"] = current_unit_type

    elif current_building:
        # Building-only follow-up can still use previous location as context,
        # but if it gives zero results, search falls back to the building text.
        if previous_filters.get("location"):
            effective["location"] = previous_filters["location"]

        effective["building"] = current_building

        if current_unit_type:
            effective["unit_type"] = current_unit_type

    elif current_unit_type:
        inherited_context = False

        if previous_filters.get("location"):
            effective["location"] = previous_filters["location"]
            inherited_context = True

        if previous_filters.get("building"):
            effective["building"] = previous_filters["building"]
            inherited_context = True

        effective["unit_type"] = current_unit_type

        # If schema could not identify location/building previously,
        # still use the previous raw search text as base context.
        if not inherited_context and previous_search_text:
            search_text = previous_search_text

    elif is_show_all_reset_request(user_text) and (previous_filters or previous_search_text):
        # "show all" should usually remove the previous unit type filter,
        # but preserve location/building.
        if previous_filters.get("location"):
            effective["location"] = previous_filters["location"]

        if previous_filters.get("building"):
            effective["building"] = previous_filters["building"]

        search_text = previous_search_text or user_text

    elif looks_like_followup(user_text) and (previous_filters or previous_search_text):
        # "price?", "details?", "available?" should use the last filters.
        effective = {
            key: value
            for key, value in previous_filters.items()
            if key in {"location", "building", "unit_type", "available_only"} and value
        }

        search_text = previous_search_text or user_text

    else:
        effective = {
            key: value
            for key, value in current_filters.items()
            if key in {"location", "building", "unit_type", "available_only"} and value
        }

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
    return any(filters.get(key) for key in ["location", "building", "unit_type"])


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

    count = len(records)
    filter_description = describe_filters(filters)

    intro = f"Sure — I found *{count}* matching record"
    intro += "" if count == 1 else "s"

    if filter_description:
        intro += f" for *{filter_description}*"

    intro += ".\n\nHere are the complete details from our current sheet:"

    lines = [intro]

    building_column = schema.get("building", "")
    unit_type_column = schema.get("unit_type", "")
    unit_no_column = schema.get("unit_no", "")
    price_column = schema.get("price", "")

    grouped = OrderedDict()

    for record in records:
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

            # Print ALL non-empty sheet columns.
        for column in output_columns:
            col_header = str(column).strip()
            
            if not col_header:
                continue
                
            value = clean_cell(record.get(column, ""))
            if not value:
                continue

            col_lower = col_header.lower()

            # 1. Hide secret columns
            hidden_keywords = [
                "s.n", 
                "prop code", 
                "moveout date", 
                "vacant on",          
                "ageing", 
                "previous rent",
                "unique id",          
                "property details"    
            ]
            
            if any(hidden in col_lower for hidden in hidden_keywords):
                continue

            # 2. Rename specific columns
            display_column = col_header
            
            if "new rent" in col_lower or "new price" in col_lower:
                display_column = "💎 Best Price"
            elif "location keywords" in col_lower:
                display_column = "📍 Nearby Locations"

            # 3. Final message line for client
            lines.append(f"   • *{display_column}:* {value}")

    global_index += 1

    lines.append(
        "\nFor viewing, booking, or more information, please connect with "
        "*Mr. Zahid* at +971562625777."
    )

    return "\n".join(lines)


# ============================================================
# OpenAI General Conversation
# ============================================================

GENERAL_SYSTEM_PROMPT = """
You are a professional, friendly real estate assistant in Dubai.

Style:
- Reply naturally like a human WhatsApp assistant.
- Be clear, polite, and helpful.
- Keep replies concise unless the user asks for details.
- Use conversation history to understand follow-up questions.

Rules:
- Do not invent property availability, prices, unit numbers, or sizes.
- If the user asks about exact property inventory, the software will search the sheet first.
- If information is not available in the provided company knowledge, ask the user for clarification or say:
  "For more information, please connect with Mr. Zahid at +971562625777."
"""


def create_general_ai_reply(
    user_question: str,
    past_history: List[Dict[str, str]],
) -> str:
    """
    Used only when no property match is found and the user is having a general conversation.
    Property data replies are deterministic to avoid missing rows.
    """
    if not client:
        return (
            "Hello! I can help you with Dubai real estate options. "
            "Please send me a location, building name, or unit type, and I will check the available details."
        )

    knowledge = get_knowledge()

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

    past_history, previous_state = get_session_snapshot(sender)

    records, schema, columns = get_properties()

    current_filters = extract_filters_from_text(
        user_text=user_question,
        records=records,
        schema=schema,
    ) if records else {}

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

    if is_likely_property_query(user_question, effective_filters or current_filters, previous_state):
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
    reply = create_general_ai_reply(user_question, past_history)

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
# Webhook De-Duplication
# ============================================================

def cleanup_seen_message_ids_locked(now: float) -> None:
    expired = [
        message_id
        for message_id, meta in seen_message_ids.items()
        if now - meta.get("ts", 0.0) > MESSAGE_ID_TTL_SECONDS
    ]

    for message_id in expired:
        seen_message_ids.pop(message_id, None)


def mark_message_seen(message_id: str) -> bool:
    """
    Returns:
    - True if message is new and should be processed.
    - False if it is a duplicate Meta retry and must be ignored.
    """
    now = time.time()

    with seen_lock:
        cleanup_seen_message_ids_locked(now)

        if message_id in seen_message_ids:
            return False

        seen_message_ids[message_id] = {
            "ts": now,
            "status": "queued",
        }

        return True


def update_message_status(message_id: str, status: str) -> None:
    with seen_lock:
        if message_id in seen_message_ids:
            seen_message_ids[message_id]["status"] = status
            seen_message_ids[message_id]["updated_at"] = time.time()


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
    - Deduplicate message IDs to prevent retry-loop duplicate replies.
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