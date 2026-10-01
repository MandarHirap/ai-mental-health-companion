# safety.py

NORMAL = "normal"
DISTRESS = "distress"
HIGH_RISK = "high_risk"


# High-risk phrases.
# This is an initial deterministic safety layer.
HIGH_RISK_PATTERNS = [
    "kill myself",
    "killing myself",
    "suicide",
    "suicidal",
    "end my life",
    "take my own life",
    "want to die",
    "wanna die",
    "don't want to live",
    "dont want to live",
    "no reason to live",
    "better off dead",
    "hurt myself",
    "harm myself",
    "self harm",
    "self-harm",
]


# Distress indicators that don't necessarily indicate
# immediate danger.
DISTRESS_PATTERNS = [
    "very stressed",
    "really stressed",
    "extremely stressed",
    "overwhelmed",
    "hopeless",
    "lonely",
    "feeling alone",
    "panic",
    "anxious",
    "anxiety",
    "depressed",
    "depression",
    "can't cope",
    "cannot cope",
    "falling apart",
    "feel helpless",
]


def classify_message(message: str) -> str:

    text = message.lower().strip()

    # Check high-risk patterns first
    for pattern in HIGH_RISK_PATTERNS:

        if pattern in text:
            return HIGH_RISK

    # Check general distress
    for pattern in DISTRESS_PATTERNS:

        if pattern in text:
            return DISTRESS

    return NORMAL


def get_safety_response() -> str:

    return (
        "I'm really sorry you're going through this. "
        "I can't provide emergency or crisis care, but you don't "
        "have to handle this alone. Please reach out to someone "
        "you trust and seek immediate help from a local emergency "
        "service or crisis-support service. If you're in immediate "
        "danger, please go to the nearest emergency department."
    )