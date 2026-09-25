"""
Map English, Swahili and mixed-language symptom text
to canonical respiratory symptom codes.

This module uses the validated English and Swahili aliases
stored in the NLP folder.
"""

import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENGLISH_ALIASES_PATH = (
    PROJECT_ROOT
    / "nlp"
    / "english_aliases.json"
)

SWAHILI_ALIASES_PATH = (
    PROJECT_ROOT
    / "nlp"
    / "swahili_aliases.json"
)

FUZZY_THRESHOLD = 82

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_text(text):
    """
    Convert text to lowercase and normalize whitespace.
    """

    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def build_alias_index():
    """
    Combine English and Swahili aliases into one searchable index.
    """

    english_aliases = load_json(
        ENGLISH_ALIASES_PATH
    )

    swahili_aliases = load_json(
        SWAHILI_ALIASES_PATH
    )

    alias_index = []

    for code, aliases in english_aliases.items():
        for alias in aliases:
            alias_index.append(
                {
                    "alias": normalize_text(alias),
                    "code": code,
                    "language": "en",
                }
            )

    for code, aliases in swahili_aliases.items():
        for alias in aliases:
            alias_index.append(
                {
                    "alias": normalize_text(alias),
                    "code": code,
                    "language": "sw",
                }
            )

    # Longer phrases are checked first.
    # Example:
    # "homa ya mafua" should be checked before "homa".
    alias_index.sort(
        key=lambda item: len(
            item["alias"]
        ),
        reverse=True,
    )

    return alias_index

def find_fuzzy_matches(
    normalized_text,
    alias_index,
    matched_codes,
    occupied_spans,
):
    """
    Find misspelled or near-matching symptom phrases.

    Fuzzy matching is only used after exact alias matching.
    This reduces the chance of replacing a correct exact match
    with a weaker approximate match.
    """

    fuzzy_matches = []

    words = list(
        re.finditer(
            r"\b[\w']+\b",
            normalized_text,
        )
    )

    for item in alias_index:
        code = item["code"]

        if code in matched_codes:
            continue

        alias = item["alias"]

        alias_word_count = len(
            alias.split()
        )

        if alias_word_count == 0:
            continue

        if len(words) < alias_word_count:
            continue

        best_score = 0
        best_candidate = None
        best_span = None

        for start_index in range(
            len(words)
            - alias_word_count
            + 1
        ):
            end_index = (
                start_index
                + alias_word_count
                - 1
            )

            start = words[
                start_index
            ].start()

            end = words[
                end_index
            ].end()

            overlaps = any(
                start < existing_end
                and end > existing_start
                for (
                    existing_start,
                    existing_end,
                )
                in occupied_spans
            )

            if overlaps:
                continue

            candidate = (
                normalized_text[
                    start:end
                ]
            )

            score = fuzz.ratio(
                candidate,
                alias,
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_span = (
                    start,
                    end,
                )

        is_close_transposition = (
            best_candidate is not None
            and len(best_candidate) == len(alias)
            and len(alias) >= 5
            and best_candidate[:2] == alias[:2]
            and best_score >= 80
        )

        if (
            best_candidate is not None
            and (
                best_score >= FUZZY_THRESHOLD
                or is_close_transposition
            )
        ):
            fuzzy_matches.append(
                {
                    "code": code,
                    "matched_alias": (
                        best_candidate
                    ),
                    "canonical_alias": alias,
                    "language": item[
                        "language"
                    ],
                    "match_type": "fuzzy",
                    "score": round(
                        best_score,
                        1,
                    ),
                    "_span": best_span,
                }
            )

            matched_codes.add(code)

            occupied_spans.append(
                best_span
            )

    return fuzzy_matches

def find_symptoms(text):
    """
    Find canonical symptoms in English, Swahili,
    or mixed-language free text.

    Exact matching is attempted first.
    Fuzzy matching is then used for remaining
    misspelled or near-matching symptom phrases.
    """

    normalized_text = normalize_text(
        text
    )

    alias_index = build_alias_index()

    matches = []
    matched_codes = set()
    occupied_spans = []

    # -------------------------
    # Exact matching
    # -------------------------

    for item in alias_index:
        alias = item["alias"]

        pattern = (
            r"(?<!\w)"
            + re.escape(alias)
            + r"(?!\w)"
        )

        for match in re.finditer(
            pattern,
            normalized_text,
        ):
            start, end = match.span()

            overlaps = any(
                start < existing_end
                and end > existing_start
                for (
                    existing_start,
                    existing_end,
                )
                in occupied_spans
            )

            if overlaps:
                continue

            code = item["code"]

            if code in matched_codes:
                continue

            matches.append(
                {
                    "code": code,
                    "matched_alias": alias,
                    "canonical_alias": alias,
                    "language": item[
                        "language"
                    ],
                    "match_type": "exact",
                    "score": 100.0,
                }
            )

            matched_codes.add(
                code
            )

            occupied_spans.append(
                (
                    start,
                    end,
                )
            )

    # -------------------------
    # Fuzzy matching
    # -------------------------

    fuzzy_matches = (
        find_fuzzy_matches(
            normalized_text,
            alias_index,
            matched_codes,
            occupied_spans,
        )
    )

    for match in fuzzy_matches:
        match.pop(
            "_span",
            None,
        )

    matches.extend(
        fuzzy_matches
    )

    return matches

def main():
    if len(sys.argv) < 2:
        print(
            "Usage:"
        )
        print(
            'python nlp/symptom_mapper.py '
            '"I have fever na kikohozi"'
        )
        return

    text = " ".join(
        sys.argv[1:]
    )

    matches = find_symptoms(
        text
    )

    print()
    print(
        "Input:",
        text,
    )

    print()
    print(
        "Recognised symptoms:"
    )

    if not matches:
        print(
            "No supported symptoms found."
        )
        return

    for match in matches:
        print(
            "-",
            match["code"],
            "|",
            match["matched_alias"],
            "|",
            match["language"],
        )


if __name__ == "__main__":
    main()