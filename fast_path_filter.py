"""
DRAFT — review the wordlist before merging into pipeline.py.

Conservative fast-path filter: skips retrieval only for queries that are
BOTH short AND contain zero domain vocabulary. Anything else still goes
through normal embedding-based retrieval, which remains the real relevance
authority.
"""

import re

# Cast a wide net deliberately - false negatives here (missing a real keyword)
# are worse than false positives (letting a junk query through to retrieval,
# which just costs ~0.5s and gets caught by DISTANCE_THRESHOLD anyway).
_DOMAIN_KEYWORDS = frozenset({
    # Fares / booking
    "fare", "fares", "ticket", "tickets", "book", "booking", "reservation",
    "reservations", "price", "fee", "fees", "cost", "pay", "payment",
    "value", "flexi", "xtra",

    # Baggage
    "baggage", "bag", "bags", "luggage", "carry", "checked", "weight",
    "kg", "allowance", "excess", "lost", "damaged",

    # Flights / schedule
    "flight", "flights", "fly", "flying", "departure", "arrival",
    "delay", "delayed", "cancel", "cancelled", "cancellation",
    "domestic", "international", "route", "destination", "schedule",

    # Check-in / airport
    "checkin", "check-in", "boarding", "airport", "counter", "pnr",
    "gate", "terminal",

    # Refunds / changes
    "refund", "refunds", "exchange", "exchanges", "change", "voucher",
    "credit", "reschedule",

    # Loyalty program
    "bluemiles", "miles", "loyalty", "rewards", "redeem", "redemption",
    "membership", "member",

    # Passengers / policy
    "passenger", "passengers", "guest", "guests", "child", "infant",
    "visa", "documents", "policy", "rules", "rights", "medical",
    "assistance", "wheelchair", "pregnant",

    # Contact / support
    "contact", "support", "helpline", "complaint", "feedback", "agent",

    # Generic but relevant verbs/nouns that show real intent
    "seat", "seats", "meal", "meals", "online", "app", "website",
})

_WORD_PATTERN = re.compile(r"[a-zA-Z]+")
_FAST_PATH_MAX_WORDS = 6


def _has_domain_keyword(query: str) -> bool:
    words = {w.lower() for w in _WORD_PATTERN.findall(query)}
    return bool(words & _DOMAIN_KEYWORDS)


def _is_fast_path_irrelevant(query: str) -> bool:
    """
    Return True only for queries that are short AND contain zero domain
    vocabulary - a narrow, conservative trap for obvious junk/small-talk.
    Anything longer or containing any domain word falls through to normal
    retrieval, which remains the real relevance judge.
    """
    word_count = len(query.split())
    if word_count > _FAST_PATH_MAX_WORDS:
        return False  # let retrieval handle longer queries regardless of keywords

    return not _has_domain_keyword(query)


if __name__ == "__main__":
    # Known junk / off-topic - SHOULD fast-path (True)
    should_block = [
        "kia haal hai",
        "do you know mr.beast",
        "what is the capital of france",
        "write me a poem about the ocean",
        "what's up",
        "tell me a joke",
    ]

    # Real queries from today's session - should NOT fast-path (False),
    # retrieval must get a chance to handle these
    should_pass = [
        "what are bluemiles",
        "baggage allowance?",
        "how much does extra baggage cost",
        "compare value and xtra fare",
        "what about flexi",
        "all fare types?",
        "explain in detail baggage+travelinfo+fares",
        "how do i request a refund",
        "what about xtra fare?",
        "how much for a business class ticket",
        "what if i miss my plane",          # KNOWN RISK CASE - short, no exact keyword
        "what happens if i miss my flight",  # same idea but has "flight" keyword
        "what's the deal if airblue cancels on me",
        "how do i earn them",                # follow-up, very short, no keyword - KNOWN RISK CASE
        "how much for ticket",
        "will my ticket include meals",
        "hello! i need help",
        "would you like to join me on my value fare flight",
    ]

    print("=== Should BLOCK (fast-path to fallback) ===")
    for query in should_block:
        result = _is_fast_path_irrelevant(query)
        status = "OK" if result else "MISMATCH"
        print(f"[{status}] '{query}' -> blocked={result}")

    print("\n=== Should PASS THROUGH (let retrieval decide) ===")
    for query in should_pass:
        result = _is_fast_path_irrelevant(query)
        status = "OK" if not result else "MISMATCH - WOULD BE WRONGLY BLOCKED"
        print(f"[{status}] '{query}' -> blocked={result}")