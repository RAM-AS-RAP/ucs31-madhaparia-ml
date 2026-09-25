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


def find_symptoms(text):
    """
    Find canonical symptoms in English, Swahili,
    or mixed-language free text.
    """

    normalized_text = normalize_text(
        text
    )

    alias_index = build_alias_index()

    matches = []
    matched_codes = set()
    occupied_spans = []

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
                    "language": item[
                        "language"
                    ],
                }
            )

            matched_codes.add(code)

            occupied_spans.append(
                (
                    start,
                    end,
                )
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