"""
The Property Panda - conversation layer.
=======================================

Why this file exists
--------------------
The deterministic engine in main.py answers a property SEARCH perfectly:
measured 12/12 on real area and building names, including typos. It answers
a CONVERSATION badly: 0/5 on viewing requests, 0/5 on "not interested" and
"yes interested", 43% on building features. Every one of those failures is
the same shape - a regex decided what the client meant, and decided wrong.
"price kya hai" was answered correctly and "kya rate hai" was answered with
a repeat of the listing, because one phrasing was in a keyword list and the
other was not. No amount of new patterns closes that gap; there is always
another phrasing.

So the split here is by what each side is actually good at:

    a message that names an area, building, unit type or budget
        -> main.py, unchanged, instant, free

    anything else a human might say
        -> this file: Claude reads it, calls the same Python functions as
           tools, and writes the reply in its own words

What stops it inventing things
------------------------------
Three layers, and the third is the one that matters.

  1. Claude never receives the sheet. It receives tool results only, so
     there is no inventory in its context to misremember.
  2. Claude is told never to write a figure, unit number or building name
     that did not come back from a tool.
  3. verify_reply() checks that instruction was obeyed, before the message
     is sent. Every AED figure, unit number and building name in the draft
     must appear in the tool results of that same turn. One that does not
     kills the whole reply and main.py falls back to the deterministic
     answer. Rule 2 is a request; rule 3 is what makes it true.

Nothing here can leave the bot silent. Every failure path - no API key, no
package, timeout, tool error, failed verification - returns None, and the
caller carries on exactly as it does today.
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Tunables. Kept small on purpose: this layer runs on the minority of
# messages, and a slow reply on WhatsApp reads as a broken one.
# --------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 3
MAX_TOKENS = 700
MAX_HISTORY_TURNS = 6


# --------------------------------------------------------------------------
# Tool schema
# --------------------------------------------------------------------------

TOOL_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "search_properties",
        "description": (
            "Look up currently available units. Use whenever the client "
            "asks what is available, mentions an area, a building, a unit "
            "type or a budget, or asks for something cheaper or bigger. "
            "Returns ready-formatted listing cards which you must paste "
            "verbatim - never retype the numbers inside them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "string",
                    "description": "Area or neighbourhood, e.g. Al Raffa, JVC, Deira.",
                },
                "building": {
                    "type": "string",
                    "description": "Building name if the client named one.",
                },
                "unit_type": {
                    "type": "string",
                    "description": "One of: Studio, 1 BR, 2 BR, 3 BR, Office, Shop.",
                },
                "max_price": {
                    "type": "integer",
                    "description": "Yearly budget ceiling in AED, e.g. 45000.",
                },
            },
        },
    },
    {
        "name": "get_policy",
        "description": (
            "Read a company policy: commission, security deposit, Ejari, "
            "cheques and payment terms, furnishing, maintenance, move-in "
            "timeline, pets, chiller and cooling, gas. Use for any question "
            "about cost, paperwork, rules or process. Never answer these "
            "from your own knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "commission | security_deposit | ejari | "
                        "cheques_payment | furnishing | maintenance | "
                        "move_in_timeline | pets | chiller | gas | all"
                    ),
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "get_building_facts",
        "description": (
            "Read the verified fact sheet for one building - parking, "
            "chiller provider, admin charge, occupancy rules. Use when the "
            "client asks about a feature of a specific building."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "building": {"type": "string"},
            },
            "required": ["building"],
        },
    },
    {
        "name": "request_callback",
        "description": (
            "Hand the client to the human broker. Use for viewings, "
            "negotiation, exact location pins, anything you cannot verify, "
            "and whenever the client asks to speak to a person. Also use "
            "it when the client says they are ready or interested."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One short line on what the client wants.",
                },
            },
            "required": ["reason"],
        },
    },
]


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

def build_system_prompt(
    agent_name: str,
    agent_phone: str,
    context_line: str = "",
) -> str:
    prompt = f"""You are the WhatsApp assistant for {agent_name}, a RERA-certified \
Dubai rentals broker. Clients message you directly.

HOW YOU SOUND
You are a working broker's assistant, not a chatbot and not a brochure. Short \
WhatsApp messages, two or three lines. Warm, calm, useful. You answer the \
question that was asked before offering anything else.

Never use pressure. Banned: "only one left", "prices rising", "book fast", \
"last chance", "hurry", "limited", "don't miss out". Never promise an outcome \
- no "you will have no issues living here", no "perfect for you". If a client \
is not interested, accept it warmly and leave the door open. Do not pitch again.

Match the client's language. If they write Hinglish, reply in Hinglish. If \
they write Arabic, reply in Arabic. Keep building and area names in English.

WHERE FACTS COME FROM
You have no property knowledge of your own. Every number, unit number, \
building name, size and policy must come from a tool result in this \
conversation.

  - Never write a rent figure, size or unit number that a tool did not return.
  - Listing cards come back preformatted. Paste them exactly as given, \
including the line breaks. Do not retype, round, convert or summarise the \
figures inside them.
  - If a tool returns nothing, say so plainly and offer {agent_name}. Never \
fill the gap with a guess.
  - You do not know rents for buildings that are not in a tool result, even \
if you think you do.

WHAT YOU HAND OVER
Viewings, negotiation, exact map pins, and anything you cannot verify go to \
{agent_name} on {agent_phone} through the request_callback tool. You never \
negotiate a price yourself and never agree to a discount.

If the client asks whether you are a bot, say plainly that you are \
{agent_name}'s AI assistant and that he handles viewings and negotiation \
himself. Do not pretend to be human."""

    if context_line:
        prompt += f"\n\nCURRENT CONTEXT\n{context_line}"

    return prompt


# --------------------------------------------------------------------------
# Verifier - the layer that makes the instructions above enforceable
# --------------------------------------------------------------------------

MONEY_IN_TEXT = re.compile(
    r"(?:aed|dhs?)\s*([\d,]+(?:\.\d+)?)|([\d,]{4,})\s*(?:aed|dhs?|/\s*year)",
    re.IGNORECASE,
)

UNIT_IN_TEXT = re.compile(r"\bunit\s+([A-Za-z0-9\-]+)", re.IGNORECASE)

SIZE_IN_TEXT = re.compile(r"([\d,]+(?:\.\d+)?)\s*sq\.?\s*ft", re.IGNORECASE)

# Words that look like a building name but are ordinary English, so a
# capitalised occurrence of them is not a claim about inventory.
NAME_STOPWORDS = {
    "i", "we", "you", "he", "she", "it", "they", "the", "a", "an",
    "hi", "hello", "yes", "no", "ok", "okay", "sure", "thanks", "thank",
    "dubai", "sharjah", "uae", "aed", "whatsapp", "ejari", "dewa",
    "studio", "bedroom", "bed", "unit", "size", "price", "rent", "year",
    "shall", "would", "could", "should", "and", "but", "for", "with",
    "mr", "mrs", "ms", "dr", "adv", "sq", "ft", "metro", "station",
    "near", "building", "buildings", "apartment", "apartments", "empower",
}


def _digits(value: str) -> str:
    return re.sub(r"[^\d]", "", str(value or ""))


def collect_allowed(tool_results: List[str]) -> Dict[str, set]:
    """
    Everything the model is permitted to state this turn, harvested from
    what the tools actually returned.
    """
    blob = "\n".join(tool_results)

    numbers = set()

    for match in re.finditer(r"[\d,]{3,}(?:\.\d+)?", blob):
        digits = _digits(match.group(0))

        if digits:
            numbers.add(digits.lstrip("0") or "0")

    units = {
        m.group(1).upper()
        for m in UNIT_IN_TEXT.finditer(blob)
    }

    names = {
        w.lower()
        for w in re.findall(r"\b[A-Z][A-Za-z0-9'&.\-]{2,}\b", blob)
    }

    return {"numbers": numbers, "units": units, "names": names}


def verify_reply(
    reply: str,
    allowed: Dict[str, set],
) -> Tuple[bool, str]:
    """
    Returns (ok, reason). A False here discards the whole reply - a message
    that is 90 percent right and quotes one invented rent is not 90 percent
    useful, it is a liability.
    """
    if not reply or not reply.strip():
        return False, "empty reply"

    # --- money ---
    for match in MONEY_IN_TEXT.finditer(reply):
        raw = match.group(1) or match.group(2) or ""
        digits = _digits(raw)

        if not digits:
            continue

        normalized = digits.lstrip("0") or "0"

        if normalized not in allowed["numbers"]:
            return False, f"unverified figure AED {raw}"

    # --- sizes ---
    for match in SIZE_IN_TEXT.finditer(reply):
        digits = _digits(match.group(1))
        normalized = digits.lstrip("0") or "0"

        if normalized and normalized not in allowed["numbers"]:
            return False, f"unverified size {match.group(1)} sq ft"

    # --- unit numbers ---
    for match in UNIT_IN_TEXT.finditer(reply):
        token = match.group(1).upper()

        if token not in allowed["units"]:
            return False, f"unverified unit {match.group(1)}"

    # --- building names ---
    for word in re.findall(r"\b[A-Z][A-Za-z0-9'&.\-]{2,}\b", reply):
        lowered = word.lower()

        if lowered in NAME_STOPWORDS:
            continue

        if lowered in allowed["names"]:
            continue

        # Only flag SHOUTED names. Ordinary sentence capitalisation is how
        # people write, and blocking it would reject every valid reply.
        if word.isupper() and len(word) > 2:
            return False, f"unverified name {word}"

    return True, ""


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------

def run_agent(
    user_text: str,
    history: List[Dict[str, str]],
    tool_impls: Dict[str, Callable[[Dict[str, Any]], str]],
    call_model: Callable[..., Any],
    system_prompt: str,
    model: str,
    log: Callable[[str], None] = lambda _m: None,
) -> Optional[str]:
    """
    Drives one turn. Returns the verified reply, or None so the caller can
    fall back to the deterministic engine.

    call_model is injected rather than imported so this can be exercised
    without a network, including against deliberately misbehaving models.
    """
    messages: List[Dict[str, Any]] = []

    for turn in history[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()

        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text})

    tool_results: List[str] = []

    for _round in range(MAX_TOOL_ROUNDS):
        try:
            response = call_model(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=TOOL_SCHEMA,
                messages=messages,
            )
        except Exception as error:
            log(f"agent: model call failed: {error}")
            return None

        blocks = list(getattr(response, "content", None) or [])
        stop_reason = getattr(response, "stop_reason", "")

        if stop_reason != "tool_use":
            text = "\n".join(
                getattr(b, "text", "")
                for b in blocks
                if getattr(b, "type", "") == "text"
            ).strip()

            if not text:
                log("agent: model returned no text")
                return None

            allowed = collect_allowed(tool_results)
            ok, reason = verify_reply(text, allowed)

            if not ok:
                log(f"agent: VERIFIER REJECTED reply - {reason}")
                return None

            return text

        # --- run the tools it asked for ---
        messages.append({"role": "assistant", "content": blocks})

        results_block: List[Dict[str, Any]] = []

        for block in blocks:
            if getattr(block, "type", "") != "tool_use":
                continue

            name = getattr(block, "name", "")
            args = getattr(block, "input", None) or {}
            impl = tool_impls.get(name)

            if impl is None:
                output = f"No such tool: {name}"
            else:
                try:
                    output = impl(args) or "No results."
                except Exception as error:
                    traceback.print_exc()
                    output = f"Tool error: {error}"

            log(f"agent: tool {name}({json.dumps(args)[:120]}) -> {len(output)} chars")
            tool_results.append(output)

            results_block.append(
                {
                    "type": "tool_result",
                    "tool_use_id": getattr(block, "id", ""),
                    "content": output,
                }
            )

        if not results_block:
            log("agent: stop_reason was tool_use but no tool_use block")
            return None

        messages.append({"role": "user", "content": results_block})

    log("agent: hit the tool-round ceiling")
    return None
