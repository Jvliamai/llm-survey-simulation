"""Central registry of SOEP variables used in this project."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal

# SOEP encodes item non-response with negative integers.
MISSING_CODES: dict[int, str] = {
    -1: "keine Angabe / weiss nicht",
    -2: "trifft nicht zu",
    -3: "unplausibler Wert",
    -4: "unzulaessige Mehrfachantwort",
    -5: "in Fragebogenversion nicht enthalten",
    -6: "Fragebogenversion mit geaenderter Filterfuehrung",
    -7: "nur in weniger eingeschraenkter Edition verfuegbar",
    -8: "Frage in diesem Jahr nicht Teil des Frageprogramms",
    -9: "fehlend wegen abgebrochenem Interview",
}

# Wave letters of the SOEP raw files, for the window used here.
WAVE_LETTERS: dict[int, str] = {
    2018: "bi", 2019: "bj", 2020: "bk", 2021: "bl",
}

# Target wave (T) and history window (waves 1..T-1), mirroring ATP waves 146-149.
TARGET_YEAR: int = 2021
HISTORY_YEARS: tuple[int, ...] = (2018, 2019, 2020)
ALL_YEARS: tuple[int, ...] = HISTORY_YEARS + (TARGET_YEAR,)

NETTO_INTERVIEW_RANGE: tuple[int, int] = (10, 19)

# SOEP sub-samples that are *not* part of the representative core.
NON_REPRESENTATIVE_PSAMPLES: tuple[int, ...] = (
    2, 4, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 
    20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 33,
)

# Covariates
Kind = Literal["categorical", "numeric", "derived"]


@dataclass(frozen=True)
class Covariate:
    name: str
    source: str
    label_de: str
    kind: Kind = "categorical"
    value_labels: dict[int, str] = field(default_factory=dict)
    formatter: Callable[[float], str] | None = None
    verified: bool = True
    note: str = ""

def _fmt_years(v: float) -> str:
    return f"{v:.0f} Jahre"

def _fmt_eur_month(v: float) -> str:
    return f"{v:,.0f} Euro brutto im Monat".replace(",", ".")

def _fmt_eur_year(v: float) -> str:
    return f"{v:,.0f} Euro im Jahr".replace(",", ".")

def _fmt_count(v: float) -> str:
    return f"{v:.0f}"

def _fmt_hours(v: float) -> str:
    return f"{v:.1f} Stunden pro Woche".replace(".", ",")

# The 25-variable demographic frame.
COVARIATES: tuple[Covariate, ...] = (
    Covariate(
        "age", "derived", "Alter", "numeric",
        formatter=_fmt_years,
    ),
    Covariate(
        "sex", "ppathl", "Geschlecht", "categorical",
        value_labels={1: "maennlich", 2: "weiblich"},
    ),
    Covariate(
        "pgcasmin", "pgen", "Hoechster Bildungsabschluss", "categorical",
        value_labels={
            0: "noch in schulischer Ausbildung",
            1: "ohne ausreichenden Abschluss",
            2: "Volks-/Hauptschulabschluss ohne Berufsausbildung",
            3: "Volks-/Hauptschulabschluss mit Berufsausbildung",
            4: "mittlere Reife ohne Berufsausbildung",
            5: "mittlere Reife mit Berufsausbildung",
            6: "Abitur ohne Berufsausbildung",
            7: "Abitur mit Berufsausbildung",
            8: "Fachhochschulabschluss",
            9: "Hochschulabschluss",
        },
    ),
    Covariate(
        "pgbilzeit", "pgen", "Bildungsjahre", "numeric",
        formatter=_fmt_years,
    ),
    Covariate(
        "pgemplst", "pgen", "Erwerbsstatus", "categorical",
        value_labels={
            1: "voll erwerbstaetig",
            2: "teilzeitbeschaeftigt",
            3: "in beruflicher Ausbildung oder Lehre",
            4: "geringfuegig oder unregelmaessig beschaeftigt",
            5: "nicht erwerbstaetig",
            6: "in einer Werkstatt fuer behinderte Menschen",
            7: "in Kurzarbeit",
        },
    ),
    Covariate(
        "pgstib_group", "derived", "Berufliche Stellung", "categorical",
        value_labels={
            1: "nicht erwerbstaetig / in Ausbildung",
            2: "Arbeiter(in)",
            3: "Selbstaendige(r)",
            4: "Angestellte(r)",
            5: "Beamte(r)",
        },
    ),
    Covariate(
        "pgisco08_major", "derived", "Berufsgruppe", "categorical",
        value_labels={
            0: "Angehoerige der Streitkraefte",
            1: "Fuehrungskraft",
            2: "akademischer Beruf",
            3: "Techniker(in) oder gleichrangiger Beruf",
            4: "Buerokraft oder verwandter Beruf",
            5: "Dienstleistungsberuf oder Verkauf",
            6: "Fachkraft in Land-/Forstwirtschaft oder Fischerei",
            7: "Handwerks- oder verwandter Beruf",
            8: "Bediener(in) von Anlagen und Maschinen",
            9: "Hilfsarbeitskraft",
        },
        # First digit of pgen/pgisco08 (major groups).
    ),
Covariate(
    "pgnace_section", "derived", "Branche", "categorical",
    value_labels={
        1: "Land- und Forstwirtschaft, Fischerei",
        2: "Bergbau und Gewinnung von Steinen und Erden",
        3: "Verarbeitendes Gewerbe",
        4: "Energieversorgung",
        5: "Wasserversorgung, Abwasser- und Abfallentsorgung",
        6: "Baugewerbe",
        7: "Handel; Instandhaltung und Reparatur von Kraftfahrzeugen",
        8: "Verkehr und Lagerei",
        9: "Gastgewerbe",
        10: "Information und Kommunikation",
        11: "Finanz- und Versicherungsdienstleistungen",
        12: "Grundstuecks- und Wohnungswesen",
        13: "Freiberufliche, wissenschaftliche und technische Dienstleistungen",
        14: "Sonstige wirtschaftliche Dienstleistungen",
        15: "Oeffentliche Verwaltung, Verteidigung, Sozialversicherung",
        16: "Erziehung und Unterricht",
        17: "Gesundheits- und Sozialwesen",
        18: "Kunst, Unterhaltung und Erholung",
        19: "Sonstige Dienstleistungen",
        20: "Private Haushalte mit Hauspersonal",
        21: "Exterritoriale Organisationen und Koerperschaften",
    },
    verified=False,
),
    Covariate(
        "pglabgro", "pgen", "Bruttoerwerbseinkommen", "numeric",
        formatter=_fmt_eur_month,
    ),
    Covariate(
        "i11102", "pequiv", "Verfuegbares Haushaltseinkommen", "numeric",
        formatter=_fmt_eur_year,
    ),
    Covariate(
        "pgfamstd", "pgen", "Familienstand", "categorical",
        value_labels={
            1: "verheiratet und mit Ehepartner(in) zusammenlebend",
            2: "verheiratet, aber getrennt lebend",
            3: "ledig",
            4: "geschieden",
            5: "verwitwet",
            6: "verheiratet, Ehepartner(in) im Ausland",
            7: "in eingetragener gleichgeschlechtlicher Lebenspartnerschaft zusammenlebend",
            8: "in eingetragener gleichgeschlechtlicher Lebenspartnerschaft, getrennt lebend",
        },
    ),
    Covariate(
        "pgpartz", "pgen", "Partner(in) im Haushalt", "categorical",
        value_labels={
            0: "kein Partner",
            1: "Ehepartner(in) / eingetragene(r) Partner(in)",
            2: "Lebenspartner(in)",
            3: "wahrscheinlich Ehepartner(in) / eingetragene(r) Partner(in)",
            4: "wahrscheinlich Lebenspartner(in)",
            5: "Partnerschaft unklar",
       },
        verified=False,
    ),
    Covariate("d11106", "pequiv", "Personen im Haushalt", "numeric", formatter=_fmt_count),
    Covariate("d11107", "pequiv", "Kinder im Haushalt", "numeric", formatter=_fmt_count),
    Covariate(
       "pli0096_h", "pl", "Ehrenamtliches Engagement", "categorical",
       value_labels={
           1: "taeglich",
           2: "mindestens 1x pro Woche",
           3: "mindestens 1x pro Monat",
           4: "seltener",
           5: "nie",
      },
      verified=True,
    ),
    Covariate(
        "hgowner", "hgen", "Wohnsituation", "categorical",
        value_labels={
            1: "Eigentuemer(in) der Wohnung bzw. des Hauses",
            2: "Hauptmieter(in)",
            3: "Untermieter(in)",
            4: "Mieter(in)",
            5: "in einem Heim oder einer Gemeinschaftsunterkunft lebend",
        },
    ),
    Covariate(
        "bula_h", "hbrutto", "Bundesland", "categorical",
        value_labels={
            1: "Schleswig-Holstein", 2: "Hamburg", 3: "Niedersachsen", 4: "Bremen",
            5: "Nordrhein-Westfalen", 6: "Hessen", 7: "Rheinland-Pfalz",
            8: "Baden-Wuerttemberg", 9: "Bayern", 10: "Saarland", 11: "Berlin",
            12: "Brandenburg", 13: "Mecklenburg-Vorpommern", 14: "Sachsen",
            15: "Sachsen-Anhalt", 16: "Thueringen",
        },
        verified=False,
    ),
    Covariate(
        "sampreg", "ppathl", "Region", "categorical",
        value_labels={1: "Westdeutschland", 2: "Ostdeutschland"},
    ),
    Covariate(
        "migback", "ppathl", "Migrationshintergrund", "categorical",
        value_labels={
            1: "kein Migrationshintergrund",
            2: "direkter Migrationshintergrund",
            3: "indirekter Migrationshintergrund",
        },
    ),
    Covariate(
        "pgnation", "pgen", "Staatsangehoerigkeit", "categorical",
        value_labels={1: "deutsch"},
        verified=False,
    ),
    Covariate(
        "plh0258_h", "pl", "Konfession", "categorical",
        value_labels={
            1: "roemisch-katholisch",
            2: "evangelisch",
            3: "andere christliche Religionsgemeinschaft",
            4: "islamisch",
            5: "andere Religionsgemeinschaft",
            6: "keiner Religionsgemeinschaft zugehoerig",
            7: "christlich-orthodox",
            8: "islamisch",
            9: "schiitisch",
            10: "alevitisch",
            11: "Mehrfachnennung",
            12: "juedisch",
        },
        verified=True,
        note="Codes 4 and 8 both read 'islamisch' by design (harmonised '_h' "
             "variable, two questionnaire versions/waves), not by error.",
    ),
    Covariate(
        "pli0098_h", "pl", "Besuch religioeser Veranstaltungen", "categorical",
        value_labels={
            1: "taeglich", 2: "mindestens einmal pro Woche",
            3: "mindestens einmal pro Monat", 4: "seltener", 5: "nie",
        },
    ),
    Covariate(
        "ple0008", "pl", "Gesundheitszustand", "categorical",
        value_labels={
            1: "sehr gut", 2: "gut", 3: "zufriedenstellend",
            4: "weniger gut", 5: "schlecht",
        },
    ),
    Covariate(
        "plh0011_h", "pl", "Parteineigung vorhanden", "categorical",
        value_labels={1: "ja", 2: "nein", 3: "weiss nicht"},
    ),
    Covariate(
        "plh0012_h", "pl", "Partei", "categorical",
        value_labels={
            1: "SPD", 2: "CDU", 3: "CSU", 4: "FDP", 5: "Buendnis 90/Die Gruenen",
            6: "Die Linke", 7: "NPD / Die Rechte / Republikaner",
            26: "Piratenpartei", 27: "AfD", 32: "Freie Waehler",
        },
        verified=False,
    ),
)

COVARIATE_NAMES: tuple[str, ...] = tuple(c.name for c in COVARIATES)
COVARIATES_BY_NAME: dict[str, Covariate] = {c.name: c for c in COVARIATES}


def covariates_from(source: str) -> list[Covariate]:
    """All covariates read from a given SOEP dataset file."""
    return [c for c in COVARIATES if c.source == source]

# Responso history items 
@dataclass(frozen=True)
class HistoryItem:
    name: str
    question_de: str
    value_labels: dict[int, str] = field(default_factory=dict)
    scale_hint: str = ""
    leaks_q1: bool = False
    leaks_q2: bool = False

_WORRY_LABELS = {1: "grosse Sorgen", 2: "einige Sorgen", 3: "keine Sorgen"}
_SAT_HINT = "Skala 0 (ganz und gar unzufrieden) bis 10 (ganz und gar zufrieden)"

HISTORY_ITEMS: tuple[HistoryItem, ...] = (
    HistoryItem(
        "plh0007", "Wie stark interessieren Sie sich fuer Politik?",
        {1: "sehr stark", 2: "stark", 3: "nicht so stark", 4: "ueberhaupt nicht"},
        leaks_q2=True,
    ),
    HistoryItem(
        "plh0011_h", "Neigen Sie einer bestimmten Partei zu?",
        {1: "ja", 2: "nein", 3: "weiss nicht"},
    ),
    HistoryItem(
    "plh0012_h", "Welcher Partei neigen Sie zu?",
    {
        1: "SPD", 2: "CDU", 3: "CSU", 4: "FDP", 5: "Buendnis 90/Die Gruenen",
        6: "Die Linke", 7: "NPD / Die Rechte / Republikaner",
        26: "Piratenpartei", 27: "AfD", 32: "Freie Waehler",
    },
),
    HistoryItem(
        "plh0032", "Sorgen um die allgemeine wirtschaftliche Entwicklung",
        _WORRY_LABELS, leaks_q1=True,
    ),
    HistoryItem(
        "plh0033", "Sorgen um die eigene wirtschaftliche Situation",
        _WORRY_LABELS, leaks_q1=True,
    ),
    HistoryItem("plh0036", "Sorgen um den Schutz der Umwelt", _WORRY_LABELS),
    HistoryItem("plh0038", "Sorgen um die Erhaltung des Friedens", _WORRY_LABELS),
    HistoryItem("plh0040", "Sorgen um die Entwicklung der Kriminalitaet", _WORRY_LABELS),
    HistoryItem("plh0042", "Sorgen um die Sicherheit des Arbeitsplatzes", _WORRY_LABELS),
    HistoryItem("plh0335", "Sorgen um die eigene Altersversorgung", _WORRY_LABELS, leaks_q1=True),
    HistoryItem(
        "plh0336", "Sorgen um den sozialen Zusammenhalt der Gesellschaft",
        _WORRY_LABELS, leaks_q2=True,
    ),
    HistoryItem("plj0046", "Sorgen ueber die Zuwanderung nach Deutschland", _WORRY_LABELS),
    HistoryItem(
        "plj0047", "Sorgen ueber Auslaenderfeindlichkeit und Fremdenhass",
        _WORRY_LABELS,
    ),
    HistoryItem("plh0171", "Zufriedenheit mit der Gesundheit", {}, _SAT_HINT),
    HistoryItem(
        "plh0175", "Zufriedenheit mit dem Haushaltseinkommen", {}, _SAT_HINT,
        leaks_q1=True,
    ),
    HistoryItem(
        "plh0176", "Zufriedenheit mit dem persoenlichen Einkommen", {}, _SAT_HINT,
        leaks_q1=True,
    ),
    HistoryItem("plh0182", "Zufriedenheit mit dem Leben insgesamt", {}, _SAT_HINT),
    HistoryItem(
        "ple0008", "Gegenwaertiger Gesundheitszustand",
        {1: "sehr gut", 2: "gut", 3: "zufriedenstellend", 4: "weniger gut", 5: "schlecht"},
    ),
)

HISTORY_NAMES: tuple[str, ...] = tuple(h.name for h in HISTORY_ITEMS)
HISTORY_BY_NAME: dict[str, HistoryItem] = {h.name: h for h in HISTORY_ITEMS}


def history_items(variant: str, focal: str) -> list[HistoryItem]:
    """
    Select history items.

    :param variant: ``"full"`` keeps everything; ``"clean"`` drops items that
        are semantically close to the focal question (leakage control).
    :param focal: ``"q1"`` or ``"q2"``.
    """
    if variant == "full":
        return list(HISTORY_ITEMS)
    if variant != "clean":
        raise ValueError(f"history variant must be 'full' or 'clean', got {variant!r}")
    attr = "leaks_q1" if focal == "q1" else "leaks_q2"
    return [h for h in HISTORY_ITEMS if not getattr(h, attr)]


# Focal questions
@dataclass(frozen=True)
class FocalQuestion:
    key: str
    variable: str
    source: str
    syear: int
    domain: str
    question_de: str
    choices_de: tuple[str, ...]
    soep_code_to_letter: dict[int, str]
    response_map: dict[str, float]
    atp_counterpart: str
    atp_theta_star: float
    note: str = ""

import math  

_NAN = math.nan

Q1 = FocalQuestion(
    key="q1",
    variable="plh0427i03",
    source="pl",
    syear=2021,
    domain="economics",
    question_de=(
        "Wie hoch ist Ihr Lebensstandard im Vergleich zu dem Ihrer Eltern, "
        "als sie in Ihrem Alter waren?"
    ),
    choices_de=(
        "Viel hoeher als der Ihrer Eltern",
        "Etwas hoeher als der Ihrer Eltern",
        "Etwa gleich wie der Ihrer Eltern",
        "Etwas niedriger als der Ihrer Eltern",
        "Viel niedriger als der Ihrer Eltern",
        "Person(en) nicht vorhanden",
    ),
    soep_code_to_letter={5: "A", 4: "B", 3: "C", 2: "D", 1: "E", 6: "F"},
    response_map={"A": 4, "B": 3, "C": 2, "D": 1, "E": 0, "F": _NAN},
    atp_counterpart="ATP Q1 (standard of living vs. parents)",
    atp_theta_star=3.16,
)

_Q2_LETTERS = "ABCDEFGHIJK"

Q2 = FocalQuestion(
    key="q2",
    variable="plh0421i02",
    source="pl",
    syear=2021,
    domain="politics",
    question_de=(
        "Bitte geben Sie an, wie sehr Sie persoenlich dem Rechtssystem vertrauen. "
        "Verwenden Sie dazu eine Skala von 0 bis 10. Dabei bedeutet 0, dass Sie dem "
        "Rechtssystem ueberhaupt nicht vertrauen, und 10, dass Sie ihm voll und ganz "
        "vertrauen."
    ),
    choices_de=(
        "10 - vertraue ich voll und ganz", "9", "8", "7", "6", "5",
        "4", "3", "2", "1", "0 - vertraue ich ueberhaupt nicht",
    ),
    soep_code_to_letter={
        10: "A", 9: "B", 8: "C", 7: "D", 6: "E", 5: "F",
        4: "G", 3: "H", 2: "I", 1: "J", 0: "K",
    },
    response_map={
        "A": 10.0, "B": 9.0, "C": 8.0, "D": 7.0, "E": 6.0, "F": 5.0,
        "G": 4.0, "H": 3.0, "I": 2.0, "J": 1.0, "K": 0.0,
    },
    atp_counterpart="ATP Q2 (Supreme Court job rating)",
    atp_theta_star=3.57,
)

FOCAL_QUESTIONS: dict[str, FocalQuestion] = {"q1": Q1, "q2": Q2}


def letters_for(fq: FocalQuestion) -> list[str]:
    """Prompt letters A, B, C, ... for a focal question, in choice order."""
    return [chr(ord("A") + i) for i in range(len(fq.choices_de))]

# Subgroup variables
SUBGROUP_VARS: dict[str, str] = {
    "sex": "Geschlecht",
    "edu3": "Bildung (3 Stufen, aus pgcasmin)",
    "income_quintile": "Einkommensquintil",
    "sampreg": "West-/Ostdeutschland",
    "age3": "Altersgruppe (3 Stufen)",
    "migback": "Migrationshintergrund",
}

# Reading data
ID_COLS: tuple[str, ...] = ("pid", "hid", "syear")

READ_PLAN: dict[str, tuple[str, ...]] = {
    "ppathl": ("pid", "hid", "cid", "syear", "sex", "gebjahr", "gebmonat", "migback",
               "sampreg", "psample", "netto", "phrf"),
    "pgen": ("pid", "hid", "syear", "pgcasmin", "pgbilzeit", "pgemplst", "pgstib",
             "pgisco08", "pgnace2", "pglabgro", "pgfamstd", "pgpartz", "pgnation"),
    "pequiv": ("pid", "hid", "syear", "d11101", "d11106", "d11107", "i11102"),
    "hgen": ("hid", "syear", "hgowner"),
    "hbrutto": ("hid", "syear", "bula_h"),
    "pl": ("pid", "hid", "syear", Q1.variable, Q2.variable,
           "plh0428",
           *sorted({*HISTORY_NAMES, "plh0258_h", "pli0098_h", "ple0008",
                    "plh0011_h", "plh0012_h", "pli0096_h"})),
}

if __name__ == "__main__":  
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the variable registry.")
    parser.add_argument("--audit", action="store_true",
                        help="List entries whose labels are not verified.")
    args = parser.parse_args()

    if args.audit:
        print("Unverified covariate labels — check against your SOEP codebook:\n")
        for c in COVARIATES:
            if not c.verified:
                print(f"  {c.name:20s} ({c.source:8s})  {c.note}")
    else:
        print(f"Target wave T = {TARGET_YEAR}; history = {HISTORY_YEARS}")
        print(f"{len(COVARIATES)} covariates, {len(HISTORY_ITEMS)} history items")
        for key, fq in FOCAL_QUESTIONS.items():
            print(f"  {key}: {fq.variable} ({fq.domain}), "
                  f"{len(fq.choices_de)} options, syear={fq.syear}")
