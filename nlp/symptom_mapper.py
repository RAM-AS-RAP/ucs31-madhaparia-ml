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

from rapidfuzz import fuzz, process


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

CANONICAL_SYMPTOMS_PATH = (
    PROJECT_ROOT
    / "nlp"
    / "canonical_symptoms.json"
)

FUZZY_THRESHOLD = 90
SUGGESTION_THRESHOLD = 70

FILLER_WORDS = {
    "i",
    "im",
    "i'm",
    "have",
    "has",
    "having",
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "my",
    "me",
    "am",
    "is",
    "are",
    "feel",
    "feeling",
    "with",
    "also",
    "nina",
    "niko",
    "na",
    "lakini",
    "kuwa",
    "sana",
}

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
                occupied_spans.append((start, end))
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

def find_unmatched_text(
    text,
    matches,
):
    """
    Return meaningful words or phrases that were not
    recognised as supported symptoms.
    """

    remaining_text = normalize_text(
        text
    )

    # Remove all recognised phrases, including alternative
    # aliases that refer to the same canonical symptom.
    matched_codes = {
        match["code"] for match in matches
    }

    aliases_to_remove = {
        normalize_text(match["matched_alias"])
        for match in matches
    }

    if matched_codes:
        aliases_to_remove.update(
            item["alias"]
            for item in build_alias_index()
            if item["code"] in matched_codes
        )

    # Remove longer phrases first to avoid partial matches.
    for alias in sorted(
        aliases_to_remove,
        key=len,
        reverse=True
    ):
        pattern = (
            r"(?<!\w)"
            + re.escape(alias)
            + r"(?!\w)"
        )

        remaining_text = re.sub(
            pattern,
            " ",
            remaining_text,
        )

    # Treat common English/Swahili connectors as separators.
    remaining_text = re.sub(
        r"\b(and|but|or|na|lakini)\b",
        ",",
        remaining_text,
    )

    segments = re.split(
        r"[,;.!?]+",
        remaining_text,
    )

    unmatched = []

    for segment in segments:
        words = re.findall(
            r"\b[\w']+\b",
            segment,
        )

        meaningful_words = [
            word
            for word in words
            if word not in FILLER_WORDS
        ]

        if not meaningful_words:
            continue

        phrase = " ".join(
            meaningful_words
        )

        if phrase not in unmatched:
            unmatched.append(
                phrase
            )

    return unmatched

def suggest_symptoms(
    unmatched_phrases,
    alias_index,
    limit=3,
):
    """
    Suggest supported symptoms for unmatched wording.

    Suggestions are for user review only and are not
    automatically treated as confirmed symptoms.
    """

    suggestions = []

    alias_lookup = {
        item["alias"]: item
        for item in alias_index
    }

    alias_choices = list(
        alias_lookup.keys()
    )

    for phrase in unmatched_phrases:

        # Prevent unsupported symptoms from
        # being incorrectly suggested.
        unsupported_terms = {
            "sneezing",
            "sneeze",
            "sneezes",
        }

        if phrase.strip().lower() in unsupported_terms:
            suggestions.append({
                "unmatched_text": phrase,
                "suggestions": [],
            })
            continue

        results = process.extract(
            phrase,
            alias_choices,
            scorer=fuzz.ratio,
            limit=limit,
        )

        phrase_suggestions = []

        seen_codes = set()

        for (
            alias,
            score,
            _,
        ) in results:

            if score < SUGGESTION_THRESHOLD:
                continue

            item = alias_lookup[
                alias
            ]

            code = item["code"]

            if code in seen_codes:
                continue

            phrase_suggestions.append(
                {
                    "code": code,
                    "suggested_alias": alias,
                    "language": item[
                        "language"
                    ],
                    "score": round(
                        score,
                        1,
                    ),
                }
            )

            seen_codes.add(
                code
            )

        suggestions.append(
            {
                "unmatched_text": phrase,
                "suggestions": (
                    phrase_suggestions
                ),
            }
        )

    return suggestions


def infer_symptom_status(text, matched_alias):
    """
    Determine whether a recognised symptom is
    present, absent or unknown.
    """
    normalized = normalize_text(text)

    pattern = (
        r"(?<!\w)"
        + re.escape(normalize_text(matched_alias))
        + r"(?!\w)"
    )

    occurrences = list(re.finditer(pattern, normalized))

    if not occurrences:
        return "unknown"

    statuses = []

    for occurrence in occurrences:
        before = normalized[:occurrence.start()]


        after = normalized[occurrence.end():]

        # Check whether a symptom belongs to another person.
        subject_clause = re.split(
            r"\b(?:but|however|lakini)\b|[.!?;]",
            before,
        )[-1]

        other_person = re.search(
            r"\bmy\s+(?:sister|brother|mother|father|"
            r"mom|dad|son|daughter|child|friend|wife|husband)\b",
            subject_clause,
        )

        patient_mentioned = (
            other_person is not None
            and re.search(
                r"\b(?:i|me|myself)\b",
                subject_clause[other_person.end():],
            )
        )

        if other_person and not patient_mentioned:
            denied = re.match(
                r"\s*[,;]?\s*(?:but\s+)?not\s+me\b",
                after,
            )

            statuses.append(
                "absent" if denied else "unknown"
            )
            continue

        # Check whether a past symptom has resolved.
        if re.search(
            r"\bi\s+had\s+(?:a\s+|an\s+)?$",
            before,
        ):
            if re.search(
                r"\b(?:but\s+not\s+now|not\s+anymore|no\s+longer)\b",
                after,
            ):
                statuses.append("absent")

            elif re.search(
                r"\bstill\s+have\s+it\b",
                after,
            ):
                statuses.append("present")

            else:
                statuses.append("unknown")

            continue


        clause = re.split(
            r"\b(?:but|lakini|however)\b"
            r"|\b(?:and|na)\s+(?=(?:i\s+)?"
            r"(?:am|have|feel|can|cannot|can't|was|nina|sina)\b)"
            r"|[.!?;]",
            before
        )[-1]


        uncertain = re.search(
            r"\b(?:not\s+sure|unsure|don'?t\s+know|sijui|maybe|perhaps)\b",
            clause
        )

        negations = list(re.finditer(
            r"\b(?:no|not|never|without|don'?t|doesn'?t|didn'?t|sina|hana|hakuna)\b",
            clause
        ))

        if uncertain:
            statuses.append("unknown")

        elif negations:
            last_negation = negations[-1]
            remainder = clause[last_negation.end():]

            restarted = re.search(
                r"\b(?:i\s+have|i\s+am|nina)\b",
                remainder
            )

            statuses.append(
                "present" if restarted else "absent"
            )

        else:
            statuses.append("present")

    if len(set(statuses)) == 1:
        return statuses[0]

    return "unknown"


def infer_symptom_subject(text, matched_alias):
    """
    Identify clear reports about another person.
    Unclear cases remain eligible for safety review.
    """
    normalized = normalize_text(text)
    alias = normalize_text(matched_alias)

    pattern = (
        r"(?<!\w)"
        + re.escape(alias)
        + r"(?!\w)"
    )

    occurrences = list(re.finditer(pattern, normalized))

    if not occurrences:
        return "unclear"

    subjects = []

    for occurrence in occurrences:
        before = normalized[:occurrence.start()]

        clause = re.split(
            r"[.!?;]|\b(?:but|however|lakini)\b",
            before
        )[-1]

        other_people = list(re.finditer(
            r"\bmy\s+(?:sister|brother|mother|father|"
            r"mom|dad|son|daughter|child|friend|"
            r"wife|husband)\b",
            clause
        ))

        patient_references = list(re.finditer(
            r"\b(?:i|me|myself)\b",
            clause
        ))

        if (
            other_people
            and (
                not patient_references
                or other_people[-1].start()
                > patient_references[-1].start()
            )
        ):
            subjects.append("other_person")
        else:
            subjects.append("patient_or_unspecified")

    if all(subject == "other_person" for subject in subjects):
        return "other_person"

    return "patient_or_unspecified"



def infer_repeated_symptom_context(
    text, matched_alias, code=None, alias_index=None
):
    """
    Combine different phrases for the same symptom code,
    while keeping patient and other-person reports separate.
    """
    normalized = normalize_text(text)
    aliases = {normalize_text(matched_alias)}

    if code is not None and alias_index is not None:
        aliases.update(
            item["alias"]
            for item in alias_index
            if item["code"] == code
        )

    def matching_aliases(fragment):
        found = []
        occupied = []

        # Longer phrases take priority over overlapping ones.
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = (
                r"(?<!\w)"
                + re.escape(alias)
                + r"(?!\w)"
            )

            for occurrence in re.finditer(pattern, fragment):
                overlaps = any(
                    occurrence.start() < end
                    and occurrence.end() > start
                    for start, end in occupied
                )

                if overlaps:
                    continue

                found.append(alias)
                occupied.append(occurrence.span())

        return found

    # Preserve existing behaviour for one symptom mention.
    if len(matching_aliases(normalized)) < 2:
        return (
            infer_symptom_status(text, matched_alias),
            infer_symptom_subject(text, matched_alias),
        )

    clauses = re.split(
        r"\b(?:but|however|lakini)\b"
        r"|\b(?:and|na)\s+(?=(?:i|my|am|nina|sina)\b)"
        r"|,\s*(?=(?:i|my)\b)"
        r"|[.!?;]",
        normalized,
    )

    mentions = []

    for clause in clauses:
        for alias in matching_aliases(clause):
            mentions.append((
                infer_symptom_status(clause, alias),
                infer_symptom_subject(clause, alias),
            ))

    if not mentions:
        return "unknown", "patient_or_unspecified"

    patient_states = {
        status
        for status, subject in mentions
        if subject != "other_person"
    }

    if not patient_states:
        return "unknown", "other_person"

    if len(patient_states) == 1:
        return patient_states.pop(), "patient_or_unspecified"

    # Conflicting patient reports require confirmation.
    return "unknown", "patient_or_unspecified"


def extract_symptom_duration_days(text, matched_alias):
    """
    Extract a clear symptom duration in days.
    Returns None when duration is not understood.
    """
    normalized = normalize_text(text)
    alias = normalize_text(matched_alias)

    pattern = (
        r"(?<!\w)"
        + re.escape(alias)
        + r"(?!\w)"
    )

    number_words = {
        "a": 1, "an": 1,
        "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8,
        "nine": 9, "ten": 10,
    }

    for occurrence in re.finditer(pattern, normalized):
        following = normalized[occurrence.end():]

        # Only inspect wording associated with this symptom.
        local_text = re.split(
            r"[,;.!?]|\b(?:and|but|lakini|na)\b",
            following,
            maxsplit=1,
        )[0][:80]

        duration_match = re.search(
            r"\bfor\s+"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)"
            r"\s+(days?|weeks?)\b",
            local_text,
        )

        if duration_match:
            quantity_text = duration_match.group(1)
            unit = duration_match.group(2)

            if quantity_text.isdigit():
                quantity = int(quantity_text)
            else:
                quantity = number_words[quantity_text]

            if quantity < 1:
                continue

            if unit.startswith("week"):
                return quantity * 7

            return quantity

        if re.search(
            r"^\s+since\s+yesterday\b",
            local_text,
        ):
            return 1

    return None


def extract_patient_duration_days(
    text, matched_alias, code, alias_index
):
    """
    Extract symptom duration from patient statements only.
    Return None if patient durations conflict or are unclear.
    """
    normalized = normalize_text(text)

    aliases = {normalize_text(matched_alias)}
    aliases.update(
        item["alias"]
        for item in alias_index
        if item["code"] == code
    )

    clauses = re.split(
        r"\b(?:but|however|lakini)\b"
        r"|\b(?:and|na)\s+(?=(?:i|my|am|nina|sina)\b)"
        r"|,\s*(?=(?:i|my)\b)"
        r"|[.!?;]",
        normalized,
    )

    durations = set()

    for clause in clauses:
        for alias in aliases:
            pattern = (
                r"(?<!\w)"
                + re.escape(alias)
                + r"(?!\w)"
            )

            if not re.search(pattern, clause):
                continue

            if infer_symptom_subject(
                clause, alias
            ) == "other_person":
                continue

            if infer_symptom_status(
                clause, alias
            ) != "present":
                continue

            days = extract_symptom_duration_days(
                clause, alias
            )

            if days is not None:
                durations.add(days)

    if len(durations) == 1:
        return next(iter(durations))

    return None

def build_confirmation_output(text):
    """
    Build the structured output that can be sent to the
    frontend for symptom review and confirmation.
    """

    matches = find_symptoms(
        text
    )

    alias_index = build_alias_index()

    unmatched_phrases = (
        find_unmatched_text(
            text,
            matches,
        )
    )

    suggestions = (
        suggest_symptoms(
            unmatched_phrases,
            alias_index,
        )
    )

    canonical_symptoms = load_json(
        CANONICAL_SYMPTOMS_PATH
    )

    canonical_index = {
        entry["code"]: entry
        for entry in canonical_symptoms
    }

    recognised_symptoms = []

    for match in matches:
        code = match["code"]

        if code not in canonical_index:
            raise KeyError(
                f"Canonical symptom not found: "
                f"{code}"
            )

        symptom = canonical_index[
            code
        ]

        status, reported_subject = infer_repeated_symptom_context(
            text,
            match["matched_alias"],
            code=code,
            alias_index=alias_index,
        )

        duration_days = None

        if status == "present":
            duration_days = extract_patient_duration_days(
                text,
                match["matched_alias"],
                code,
                alias_index,
            )

        recognised_symptoms.append(
            {
                "code": code,
                "display_name": symptom[
                    "display_name"
                ],
                "features": symptom[
                    "features"
                ],
                "kind": symptom[
                    "kind"
                ],
                "is_red_flag": symptom[
                    "is_red_flag"
                ],
                "matched_text": match[
                    "matched_alias"
                ],
                "language": match[
                    "language"
                ],
                "match_type": match[
                    "match_type"
                ],
                "match_score": match[
                    "score"
                ],

                "status": status,
                "reported_subject": reported_subject,
                "duration_min_days": duration_days,
                "duration_max_days": duration_days,
                "confirmed": False,
            }
        )

    has_red_flag = any(
        symptom["is_red_flag"]
        and symptom["status"] != "absent"
        and symptom["reported_subject"] != "other_person"
        for symptom in recognised_symptoms
    )

    return {
        "input_text": text,
        "recognised_symptoms": (
            recognised_symptoms
        ),
        "unmatched_wording": suggestions,
        "has_red_flag": has_red_flag,
        "requires_confirmation": (
            bool(recognised_symptoms)
            or bool(suggestions)
        ),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            'python nlp/symptom_mapper.py '
            '"I have fever and a sore throat"'
        )
        return

    text = " ".join(
        sys.argv[1:]
    )

    matches = find_symptoms(
        text
    )

    alias_index = build_alias_index()

    unmatched_phrases = (
        find_unmatched_text(
            text,
            matches,
        )
    )

    suggestions = (
        suggest_symptoms(
            unmatched_phrases,
            alias_index,
        )
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

    if matches:
        for match in matches:
            print(
                "-",
                match["code"],
                "|",
                match["matched_alias"],
                "|",
                match["language"],
                "|",
                match["match_type"],
                "|",
                match["score"],
            )
    else:
        print(
            "No supported symptoms found."
        )

    print()
    print(
        "Unmatched wording:"
    )

    if not suggestions:
        print(
            "None."
        )

    for item in suggestions:
        print(
            "-",
            item["unmatched_text"],
        )

        if not item["suggestions"]:
            print(
                "  No close supported "
                "symptom suggestion."
            )
            continue

        print(
            "  Suggestions:"
        )

        for suggestion in (
            item["suggestions"]
        ):
            print(
                "  -",
                suggestion["code"],
                "|",
                suggestion[
                    "suggested_alias"
                ],
                "|",
                suggestion["language"],
                "|",
                suggestion["score"],
            )


if __name__ == "__main__":
    main()