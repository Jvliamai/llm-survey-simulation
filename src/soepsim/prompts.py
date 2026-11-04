"""
Prompt templates.

The German templates are close translations of the ATP templates
An English variant of the same prompts is provided for the language ablation.
"""

from __future__ import annotations

from typing import Literal

Language = Literal["de", "en"]

# Response generation
SYSTEM_DE = (
    "Sie sind Teilnehmerin bzw. Teilnehmer einer Umfrage. Nehmen Sie das folgende "
    "Profil an: {demographic_profile} Antworten Sie als Sie selbst in der ersten "
    "Person. Waehlen Sie genau eine Option aus der Liste. Geben Sie ausschliesslich "
    "einen einzelnen Grossbuchstaben aus [{letters}] aus. Keine Woerter, keine "
    "Satzzeichen, keine Erklaerungen, keine Einschraenkungen. Diskutieren Sie nicht "
    "ueber Ethik, Studiendesign oder darueber, was die meisten Menschen tun wuerden."
)

USER_DE = "{question}\n\nAntwortmoeglichkeiten:\n{choices}"

SYSTEM_EN = (
    "You are a survey respondent. Adopt this profile: {demographic_profile} Answer as "
    "yourself in first person. Pick exactly one option from the list. Output only one "
    "uppercase letter from [{letters}]. No words, no punctuation, no explanations, no "
    "qualifiers. Do not discuss ethics, study design, or what most people would do."
)

USER_EN = "{question}\n\nChoices:\n{choices}"

# Appended to the system message under persona-guided synthesis. 
PERSONA_SUFFIX_DE = "\n\nZusaetzliche Profilinformationen:\n{persona}"
PERSONA_SUFFIX_EN = "\n\nAdditional profile information:\n{persona}"


def format_choices(choices: list[str] | tuple[str, ...]) -> str:
    """``['Viel hoeher', ...]`` -> ``"A: Viel hoeher\\nB: ..."``."""
    return "\n".join(f"{chr(ord('A') + i)}: {c}" for i, c in enumerate(choices))


def build_prompt(
    demographic_profile: str,
    question: str,
    choices: list[str] | tuple[str, ...],
    language: Language = "de",
    persona: str | None = None,
) -> list[dict[str, str]]:
    """
    Build the chat messages for one respondent.

    :param demographic_profile: rendered profile text
    :param question: focal question
    :param choices: answer options in prompt order
    :param language: "de" or "en"
    :param persona: persona text; None for demo-only
    :return: list of system and user messages
    """
    letters = ", ".join(chr(ord("A") + i) for i in range(len(choices)))
    if language == "de":
        system = SYSTEM_DE.format(demographic_profile=demographic_profile, letters=letters)
        user = USER_DE.format(question=question, choices=format_choices(choices))
        suffix = PERSONA_SUFFIX_DE
    elif language == "en":
        system = SYSTEM_EN.format(demographic_profile=demographic_profile, letters=letters)
        user = USER_EN.format(question=question, choices=format_choices(choices))
        suffix = PERSONA_SUFFIX_EN
    else:
        raise ValueError(f"language must be 'de' or 'en', got {language!r}")

    if persona:
        system = system + suffix.format(persona=persona)

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# Persona synthesis
PERSONA_SYSTEM_DE = (
    "Sie sind Expertin bzw. Experte darin, aus demografischen und "
    "verhaltensbezogenen Daten detaillierte Personenbeschreibungen zu "
    "synthetisieren. Ihre Ausgabe muss ein einziger, zusammenhaengender Absatz sein."
)

PERSONA_USER_DE = (
    "Hier sind die demografischen Angaben einer Person und ein Protokoll ihrer "
    "frueheren Antworten.\n\n"
    "Demografie:\n{demo}\n\n"
    "Historie:\n{context}\n\n"
    "Ihre Aufgabe: Schreiben Sie auf Basis dieser Informationen einen einzigen, "
    "detaillierten Absatz, der die Gewohnheiten, die Lebensweise und die "
    "Persoenlichkeit dieser Person allgemein beschreibt. Ihre Beschreibung soll eine "
    "zusammenhaengende Erzaehlung sein, die alle verfuegbaren Hinweise zusammenfuehrt. "
    "Konzentrieren Sie sich darauf, Muster, Routinen und Rahmenbedingungen "
    "abzuleiten, die durch die vorliegenden Informationen gestuetzt werden. Verwenden "
    "Sie keine Listen, Aufzaehlungszeichen oder Bewertungen. Fuegen Sie keine "
    "zusaetzlichen Erklaerungen oder Begruendungen hinzu. Geben Sie ausschliesslich "
    "die narrative Personenbeschreibung aus."
)

PERSONA_SYSTEM_EN = (
    "You are an expert at synthesizing detailed persona descriptions from demographic "
    "and behavioral data. Your output must be a single, coherent paragraph."
)

PERSONA_USER_EN = (
    "Here is a person's demographic information and a log of what they answered in the "
    "past.\n\nDemographics\n{demo}\n\nHistory\n{context}\n\n"
    "Your task: Based on this information, write a single, detailed paragraph "
    "describing this person's habits, lifestyle, and personality in general. Your "
    "description should be a coherent narrative that synthesizes all the available "
    "evidence. Focus on inferring patterns, routines, and constraints that are "
    "supported by the provided information. Do not use lists, bullet points, or "
    "scores. Do not include extra explanations or reasoning. Just provide the "
    "narrative persona description."
)


def build_persona_prompt(
    demo: str, context: str, language: Language = "de"
) -> list[dict[str, str]]:
    """Chat messages for the auxiliary persona-writing model."""
    if language == "de":
        return [
            {"role": "system", "content": PERSONA_SYSTEM_DE},
            {"role": "user", "content": PERSONA_USER_DE.format(demo=demo, context=context)},
        ]
    if language == "en":
        return [
            {"role": "system", "content": PERSONA_SYSTEM_EN},
            {"role": "user", "content": PERSONA_USER_EN.format(demo=demo, context=context)},
        ]
    raise ValueError(f"language must be 'de' or 'en', got {language!r}")


# Domain-FT
def build_ft_example(
    demographic_profile: str,
    question: str,
    choices: list[str] | tuple[str, ...],
    answer_letter: str,
    language: Language = "de",
) -> dict:
    """
    One supervised fine-tuning example, in the chat format axolotl expects.

    Domain-FT learns from (question, demographics, response) triplets drawn from
    the history waves and never from wave T. 
    """
    msgs = build_prompt(demographic_profile, question, choices, language=language)
    return {
        "system": msgs[0]["content"],
        "conversations": [
            {"role": "user", "content": msgs[1]["content"]},
            {"role": "assistant", "content": answer_letter},
        ],
    }


# ICL
# Retrieved examples are *other* real respondents' profile + true wave-T
ICL_EXAMPLE_DE = "Profil: {demographic_profile}\nAntwort: {answer}\n"
ICL_EXAMPLE_EN = "Profile: {demographic_profile}\nAnswer: {answer}\n"

ICL_PREFIX_DE = "Hier sind die Antworten anderer Personen auf dieselbe Frage:\n\n"
ICL_PREFIX_EN = "Here are other people's answers to the same question:\n\n"

ICL_SUFFIX_DE = "\n---\n\nUnd nun Ihre eigene Antwort:\n\n"
ICL_SUFFIX_EN = "\n---\n\nAnd now your own answer:\n\n"


def build_icl_block(examples: list[tuple[str, str]], language: Language = "de") -> str:
    """
    Render the ICL examples as a text block placed before the user message.

    :param examples: [(profile, answer), ...] of other respondents
    :return: block text, or "" if there are no examples
    """
    if language not in ("de", "en"):
        raise ValueError(f"language must be 'de' or 'en', got {language!r}")
    if not examples:
        return ""
    example_tpl = ICL_EXAMPLE_DE if language == "de" else ICL_EXAMPLE_EN
    prefix = ICL_PREFIX_DE if language == "de" else ICL_PREFIX_EN
    suffix = ICL_SUFFIX_DE if language == "de" else ICL_SUFFIX_EN

    body = "\n".join(
        example_tpl.format(demographic_profile=demographic_profile, answer=answer)
        for demographic_profile, answer in examples
    )
    return prefix + body + suffix