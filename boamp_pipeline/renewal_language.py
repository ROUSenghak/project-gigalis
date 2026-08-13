"""Detection of self-declared contract succession in BOAMP notice text.

BOAMP publishes no renewal identifier, so the strongest available evidence that
one procurement replaces another is the buyer *saying so* in the notice. That
evidence sits on the successor, not the predecessor, which is why the benchmark
mines successor-to-predecessor rather than the other way round.

Patterns are declared as data rather than inlined, so every family reports its
own hit count into the run summary and can be unit-tested against real corpus
strings. Measured frequencies over all 1,620,712 notices:

===================================  ======  =========================
family                                 n     note
===================================  ======  =========================
``renouvellement`` + contract noun    1,514  1,017 of them in ``objet``
``arrivant a echeance`` + relaunch      570  often names the expiry date
``marche/titulaire actuel|precedent`` 1,359  predecessor unnamed
``depenses du precedent marche``        539  quantifies the predecessor
``reprise du personnel``                538  implies a running contract
``relance``/``infructueux``          43,550  contaminant, see below
===================================  ======  =========================

Three findings shape the rules below.

**Asset renewal is the dominant false friend.** The bare word
``renouvellement`` appears 28,475 times, but the head nouns following it are
overwhelmingly physical: ``urbain`` 6,179, ``reseau`` 4,844, ``reseaux`` 3,917,
``equipements`` 1,628, ``canalisations`` 1,272 -- against ``contrat`` 1,009 and
``marche`` 744. Only about a tenth concern a contract. Requiring a contract head
noun near the trigger, and no asset noun between them, is what reduces 28,475
to a usable 1,514.

**``reconduction`` is not succession.** It appears 43,807 times but only 129 of
those are in ``objet``; every sampled instance is the current contract's own
extension option ("reconductible 3 fois par periode successive de 1 an"). It is
extracted here only as evidence about a contract's own expected end, never as a
link to a predecessor.

**Re-tender after failure is a contaminant, not a renewal.** ``relance``,
``infructueux`` and ``sans suite`` genuinely reference a prior procedure, but
one that never produced a contract and is usually weeks old. It gets its own
class so it can be measured and excluded rather than silently inflating recall.

All matching runs on **raw** text. ``linkage.normalize_text`` strips accents and
punctuation, which would destroy the character offsets that the annotation layer
needs to verify each quotation as an exact substring of its source field.
Patterns are therefore accent-tolerant (``march[eé]``) and case-insensitive.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

RENEWAL_DECLARATION = "RENEWAL_DECLARATION"
EXPIRY_DECLARATION = "EXPIRY_DECLARATION"
INCUMBENT_MENTION = "INCUMBENT_MENTION"
PRIOR_SPEND = "PRIOR_SPEND"
STAFF_TRANSFER = "STAFF_TRANSFER"
RETENDER_AFTER_FAILURE = "RETENDER_AFTER_FAILURE"

#: Classes that assert a predecessor contract exists. An anchor is only
#: recruited from these.
POSITIVE_CLASSES = (
    RENEWAL_DECLARATION,
    EXPIRY_DECLARATION,
    INCUMBENT_MENTION,
    PRIOR_SPEND,
    STAFF_TRANSFER,
)

#: Classes recorded so their volume is measurable, but never used to recruit.
CONTAMINANT_CLASSES = (RETENDER_AFTER_FAILURE,)

#: Why a candidate match was thrown away. Exclusions are counted so the
#: filtering rate is reportable rather than invisible.
EXCLUDED_ASSET_RENEWAL = "ASSET_RENEWAL"
EXCLUDED_OWN_OPTION = "OWN_EXTENSION_OPTION"
EXCLUDED_MODIFICATION_BOILERPLATE = "MODIFICATION_BOILERPLATE"
EXCLUDED_ADVISORY_MISSION = "ADVISORY_MISSION"
EXCLUDED_CEMETERY = "CEMETERY_CONCESSION"
EXCLUDED_ACCOUNTING_TERM = "ACCOUNTING_TERM"
EXCLUDED_APPLICATION = "ADMINISTRATIVE_APPLICATION"
EXCLUDED_NEGATED = "NEGATED_RENEWAL"
EXCLUDED_FUTURE_CONTRACT = "FUTURE_CONTRACT"
EXCLUDED_SCOPE_STATEMENT = "SCOPE_STATEMENT"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Nouns naming a contractual instrument. Presence of one of these near a
#: renewal trigger is what separates contract renewal from asset renewal.
CONTRACT_HEAD = (
    r"march[eé]s?|accords?[-\s]cadres?|contrats?|conventions?|concessions?|"
    r"d[eé]l[eé]gations?\s+de\s+service\s+public|DSP|baux|bail|lots?|"
    r"consultations?|proc[eé]dures?|prestations?\s+actuelles?"
)

#: Physical things that get "renewed" in French procurement language. If one of
#: these sits between the trigger and a contract noun, the phrase is about
#: replacing equipment, not re-procuring a contract.
ASSET_HEAD = (
    r"r[eé]seaux?|canalisations?|parcs?|conduites?|voiries?|[eé]clairages?|"
    r"chauss[eé]es?|menuiseries?|toitures?|mobiliers?|v[eé]hicules?|mat[eé]riels?|"
    r"installations?|[eé]quipements?|branchements?|compteurs?|luminaires?|"
    r"colonnes?|b[aâ]timents?|urbain|urbaine|ascenseurs?|chaudi[eè]res?|"
    r"sols?|rev[eê]tements?|cl[oô]tures?|portails?|serrureries?|huisseries?|"
    r"tuyauteries?|c[aâ]bles?|postes?|transformateurs?|licences?|logiciels?|"
    r"serveurs?|ordinateurs?|postes\s+de\s+travail|flottes?|escaliers?|"
    r"ascenseurs?|trottoirs?|r[eé]sines?|peintures?|stores?|climatisations?|"
    r"biens\b|locaux\b|consommables?|poteaux?|bouches?|certificats?|ouvrages?|"
    r"fournitures?|denr[eé]es?|produits?|pi[eè]ces?|accessoires?|a[eé]rodromes?|"
    r"bornes?|abris?|conteneurs?|bacs?|extincteurs?|filtres?|pompes?|"
    r"voies?|ballast|vid[eé]osurveillances?|jeux|supervisions?|automates?|"
    r"rails?|traverses?|sc[eé]nographiques?|signalisations?|horodateurs?|"
    r"plateformes?|couches?\s+de\s+roulement|[eé]crans?|syst[eè]mes?|"
    r"moteurs?\s+de\s+recherche|progiciels?|applicatifs?"
)

#: An asset named just before a possessive trigger: "maintenance des
#: equipements scenographiques, et leur renouvellement". The asset sits before
#: the trigger rather than between it and the contract noun, so the ordinary
#: between-check cannot see it.
_POSSESSIVE_TRIGGER = re.compile(r"(?:leurs?|sons?|sa|ces?|cette|cet)\s+$", re.IGNORECASE)

#: "En cas de non-renouvellement du marche" and "Clauses de non-renouvellement"
#: describe what happens if this contract is *not* extended.
_NEGATED_RENEWAL = re.compile(r"non[-\s]?$", re.IGNORECASE)

#: "montage et renouvellement des marches a venir" concerns contracts that do
#: not exist yet, so there is no predecessor to resolve.
_FUTURE_CONTRACTS = re.compile(r"[àa]\s+venir|futur\w*|[àa]\s+lancer", re.IGNORECASE)

#: "l'entretien et le renouvellement sont exclus du present marche" is a scope
#: exclusion, not a re-procurement. Distinct from "le present marche a pour
#: objet le renouvellement des contrats", which is a genuine declaration.
_SCOPE_EXCLUSION = re.compile(r"\bexclu\w*\s+d[ue]|\bhors\s+d[ue]|ne\s+comprend\s+pas", re.IGNORECASE)

#: The contract noun can be reached through a temporal or scope phrase rather
#: than being the object of the renewal: "le renouvellement, pendant toute la
#: duree du marche, des consommables" renews consumables, not the contract.
_HEAD_IN_SCOPE_PHRASE = re.compile(
    r"(?:pendant|durant|au\s+titre|dans\s+le\s+cadre|pr[eé]vu\w*\s+au|au\s+cours|"
    r"sur\s+la\s+dur[eé]e|jusqu[’']?\s*[àa]\s+la\s+fin)\s+"
    r"(?:toute\s+)?(?:l[ae]\s+)?(?:dur[eé]e\s+)?(?:d[ue]\s+|des\s+|de\s+la\s+)?$",
    re.IGNORECASE,
)

#: A notice can announce a renewal that is *somebody else's*. An assistance,
#: advisory or study contract for the renewal of a service contract declares a
#: real succession -- but this notice is the adviser's contract, not the
#: successor. Audited on 60 random declarations, this was the single largest
#: false-positive source: 16 of 40.
ADVISORY_HEAD = (
    r"assistan\w+|assister|AMO\b|ATMO\b|ma[iî]trise\s+d[’']?ouvrage|"
    r"accompagnement|accompagner|conseils?\b|consultan\w+|audit\w*|"
    r"[eé]tudes?\b|diagnostics?|[eé]valuations?|expertises?|"
    r"candidature|mandat\s+de\s+ma[iî]trise|ma[iî]trise\s+d[’']?\s*(?:oeuvre|œuvre)|"
    r"\bMOE\b|\bAMO\b|\bATMO\b|programmation\s+des\s+travaux|appuis?\b|"
    r"\bAMU\b|assistance\s+technique\s+et\s+juridique"
)

#: "Concession" in French is both a public service delegation and a burial
#: plot. "Reprise de concessions funeraires echues" matched every expiry
#: pattern and concerns cemetery administration.
CEMETERY_CONTEXT = (
    r"fun[eé]raires?|cimeti[eè]res?|caveaux?|columbariums?|cin[eé]raires?|"
    r"s[eé]pultures?|tombes?|d[eé]funts?|ossuaires?|[eé]tat\s+d[’']?abandon|"
    r"reprise\s+d[eu]s?\s+concessions?"
)

#: A contract's own extension option, phrased with "renouvellement" rather than
#: "reconduction": "Renouvellement possible du marche deux fois un an de facon
#: tacite".
OWN_OPTION_CONTEXT = (
    r"tacite\w*|reconduction\w*|reconductible\w*|reconduits?|reconduire|"
    r"renouvellement\s+ou\s+non|(?:march[eé]|accord[-\s]cadre|contrat)\s+initial|"
    r"en\s+cas\s+de\s+renouvellement|possibilit[eé]\s+de\s+renouvellement|"
    r"p[eé]riode\s+initiale|dur[eé]e\s+initiale|au\s+sens\s+du\s+droit\s+communautaire|"
    r"tranche\s+(?:optionnelle|conditionnelle)"
)

#: "mois echu", "terme echu" are payment-schedule idioms, not contract expiry.
ACCOUNTING_ECHU = re.compile(
    r"(?:mois|trimestres?|semestres?|termes?|annuit[eé]s?|p[eé]riodes?)\s+[eé]chu",
    re.IGNORECASE,
)

#: An application to an authority for a concession, not a procurement.
APPLICATION_CONTEXT = re.compile(
    r"demande\s+de\s+renouvellement|sollicite\w*\s+le\s+renouvellement", re.IGNORECASE
)

#: Words marking the contract as the *incumbent* one rather than the one being
#: awarded. Requiring one of these is what separates "le marche actuel arrive a
#: echeance" (a predecessor) from "l'accord-cadre prendra fin le 31 decembre
#: 2026" (this contract's own term) -- 14 of the 40 audited false positives.
#: "initial" and "present" are deliberately absent: they mark the current
#: contract.
INCUMBENT_MARK = (
    r"actuel\w*|en\s+cours|existant\w*|pr[eé]c[eé]dent\w*|sortant\w*|"
    r"en\s+place|ant[eé]rieur\w*|pr[eé]cit[eé]\w*"
)

#: Renewal of physical works, phrased as a works item.
_WORKS_PREFIX = re.compile(r"(?:travaux|op[eé]rations?|programmes?|campagnes?\s+de\s+travaux)\s+(?:de\s+)?$", re.IGNORECASE)

#: Phrases announcing that a replacement procurement is being launched. These
#: carry the succession claim when the sentence never names the instrument --
#: "les abonnes arrivant a echeance en Mars 2022, une nouvelle consultation est
#: lancee" states a renewal without once saying "marche".
RELAUNCH = (
    r"nouvelle\s+consultation|nouveau\s+march[eé]|nouvelle\s+proc[eé]dure|"
    r"nouvelle\s+convention|nouvel\s+accord[-\s]cadre|relanc\w+|"
    r"remise\s+en\s+concurrence|remettre\s+en\s+concurrence|"
    r"assurer\s+la\s+continuit[eé]|renouveler\s+(?:ce|le|son|leur)"
)

#: Characters that end a sentence. Matching is confined within a sentence so a
#: trigger cannot bind to a noun several clauses away.
_SENTENCE_SPLIT = re.compile(r"[.;:!?\n\r]")

#: How far after a `renouvellement` trigger a contract noun may sit.
CONTRACT_HEAD_WINDOW = 60

#: Widened window used only if the strict pass under-delivers (gate G1).
CONTRACT_HEAD_WINDOW_WIDE = 120

#: `marche en cours` is usually "modifications du marche en cours d'execution",
#: boilerplate about amending the current contract under article R2194-1.
_MODIFICATION_BOILERPLATE = re.compile(
    r"modifications?\s+d[ue]\s+march[eé]\s+en\s+cours\s+d[’']?\s*ex[eé]cution",
    re.IGNORECASE,
)
_MODIFICATION_NEARBY = re.compile(r"modification|avenant|article\s+R\s*2194", re.IGNORECASE)
MODIFICATION_WINDOW = 40


@dataclass(frozen=True)
class Pattern:
    """One declaration family."""

    name: str
    regex: re.Pattern[str]
    klass: str
    #: Whether a contract head noun must be found near the match for it to
    #: count. Only the `renouvellement` family needs this.
    requires_contract_head: bool = False


@dataclass(frozen=True)
class Match:
    """A located declaration, carrying everything the annotator needs to check."""

    pattern: str
    klass: str
    field: str
    start: int
    end: int
    matched_text: str
    snippet: str
    excluded_reason: str = ""

    @property
    def is_excluded(self) -> bool:
        return bool(self.excluded_reason)


PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "renouvellement_contrat",
        re.compile(r"renouvell?ement\b", re.IGNORECASE),
        RENEWAL_DECLARATION,
        requires_contract_head=True,
    ),
    Pattern(
        # The instrument must be marked as the incumbent one. Without that
        # constraint this family matched every notice stating its own end date
        # ("l'accord-cadre prendra fin le 31 decembre 2026"), which was 14 of
        # 40 false positives in the first audit.
        "arrive_echeance",
        re.compile(
            # The marker need not touch the noun: "le marche d'exploitation
            # actuel du Reseau radio arrive a echeance le 15 juillet 2024".
            rf"(?:{CONTRACT_HEAD})[^.;:!?\n]{{0,60}}?\b(?:{INCUMBENT_MARK})"
            r"[^.;:!?\n]{0,100}?"
            r"(?:arriv\w*\s+[àa]\s+(?:son\s+)?(?:[eé]ch[eé]ance|terme)"
            r"|arriv\w*\s+[àa]\s+expiration"
            r"|prend\s+fin|prendra\s+fin|expire\w*|vient\s+[àa]\s+[eé]ch[eé]ance"
            r"|[eé]chu\w*|se\s+termine\w*|se\s+terminera)",
            re.IGNORECASE,
        ),
        EXPIRY_DECLARATION,
    ),
    Pattern(
        # An expiry stated without naming the instrument: "les abonnes arrivant
        # a echeance en Mars 2022, une nouvelle consultation est lancee". The
        # contract noun is absent, so the relaunch phrase carries the claim
        # instead -- which is what makes the sentence a succession declaration
        # rather than a stray deadline.
        "echeance_avec_relance",
        re.compile(
            r"(?:(?:arriv\w*\s+[àa]\s+(?:son\s+)?(?:[eé]ch[eé]ance|terme)|prend\s+fin"
            r"|prendra\s+fin|expire\w*|[eé]chu\w*)[^.;:!?\n]{0,160}?"
            rf"(?:{RELAUNCH})"
            r"|(?:" + RELAUNCH + r")[^.;:!?\n]{0,160}?"
            r"(?:arriv\w*\s+[àa]\s+(?:son\s+)?(?:[eé]ch[eé]ance|terme)|prend\s+fin"
            r"|prendra\s+fin|expire\w*|[eé]chu\w*))",
            re.IGNORECASE,
        ),
        EXPIRY_DECLARATION,
    ),
    Pattern(
        # A predecessor named without any expiry verb: "compte-tenu du marche
        # existant encore en cours de validite". The incumbent marker alone
        # carries the claim.
        "marche_existant",
        re.compile(
            rf"(?:compte[-\s]tenu|en\s+raison|du\s+fait)\s+d[ue]\s+(?:{CONTRACT_HEAD})"
            rf"[^.;:!?\n]{{0,60}}?\b(?:{INCUMBENT_MARK})",
            re.IGNORECASE,
        ),
        INCUMBENT_MENTION,
    ),
    Pattern(
        "titulaire_actuel_precedent",
        re.compile(
            rf"(?:{CONTRACT_HEAD}|titulaires?|prestataires?|d[eé]l[eé]gataires?|"
            r"fournisseurs?|exploitants?)\s+"
            r"(?:actuel\w*|pr[eé]c[eé]dent\w*|en\s+cours|sortant\w*|"
            r"existant\w*|en\s+place)",
            re.IGNORECASE,
        ),
        INCUMBENT_MENTION,
    ),
    Pattern(
        "precedent_marche",
        re.compile(
            r"(?:pr[eé]c[eé]dent\w*|ancien\w*|derni[eè]r\w*)\s+"
            rf"(?:{CONTRACT_HEAD})",
            re.IGNORECASE,
        ),
        INCUMBENT_MENTION,
    ),
    Pattern(
        "depenses_precedent",
        re.compile(
            r"(?:d[eé]penses?|montants?|quantit[eé]s?|commandes?|chiffres?\s+d[’']?affaires?|"
            r"consommations?)\s?[^.;:!?\n]{0,80}?"
            r"(?:ann[eé]es?\s+pr[eé]c[eé]dentes?|pr[eé]c[eé]dent\s+march[eé]|"
            r"march[eé]\s+pr[eé]c[eé]dent|derniers?\s+exercices?)",
            re.IGNORECASE,
        ),
        PRIOR_SPEND,
    ),
    Pattern(
        "reprise_personnel",
        re.compile(r"reprise\s+d[ue]\s+personnel", re.IGNORECASE),
        STAFF_TRANSFER,
    ),
    Pattern(
        "retender_after_failure",
        re.compile(
            r"(?:d[eé]clar[eé]\w*\s+(?:sans\s+suite|infructueu\w+)"
            r"|proc[eé]dure\s+infructueuse|appel\s+d[’']?offres?\s+infructueu\w+"
            r"|relance\s+d[ue]\s+(?:march[eé]|consultation|proc[eé]dure|MP\b|AO\b)"
            r"|nouvelle\s+consultation\s+suite\s+[àa])",
            re.IGNORECASE,
        ),
        RETENDER_AFTER_FAILURE,
    ),
)

PATTERNS_BY_NAME = {pattern.name: pattern for pattern in PATTERNS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fold(text: str) -> str:
    """Lowercase and strip accents, for window checks only -- never for offsets."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def sentence_span(text: str, position: int) -> tuple[int, int]:
    """Bounds of the sentence containing ``position``."""
    start = 0
    for separator in _SENTENCE_SPLIT.finditer(text, 0, position):
        start = separator.end()
    end_match = _SENTENCE_SPLIT.search(text, position)
    return start, end_match.start() if end_match else len(text)


def snippet_for(text: str, start: int, end: int, radius: int = 250) -> str:
    """Context around a match, kept as an exact substring of the source."""
    return text[max(0, start - radius): min(len(text), end + radius)]


_CONTRACT_HEAD_RE = re.compile(rf"\b(?:{CONTRACT_HEAD})\b", re.IGNORECASE)
_ASSET_HEAD_RE = re.compile(rf"\b(?:{ASSET_HEAD})\b", re.IGNORECASE)


def contract_head_after(
    text: str, position: int, window: int = CONTRACT_HEAD_WINDOW
) -> re.Match[str] | None:
    """First contract noun within ``window`` characters after ``position``.

    The search never leaves the sentence, so ``renouvellement du reseau. Le
    marche ...`` cannot bind ``marche`` to ``renouvellement``.
    """
    _, sentence_end = sentence_span(text, position)
    limit = min(position + window, sentence_end, len(text))
    if limit <= position:
        return None
    return _CONTRACT_HEAD_RE.search(text, position, limit)


def asset_head_between(text: str, start: int, end: int) -> re.Match[str] | None:
    """An asset noun sitting between the trigger and the contract noun."""
    if end <= start:
        return None
    return _ASSET_HEAD_RE.search(text, start, end)


def classify_renouvellement(
    text: str,
    match_end: int,
    window: int = CONTRACT_HEAD_WINDOW,
    match_start: int | None = None,
) -> tuple[bool, str]:
    """Decide whether a ``renouvellement`` trigger concerns a contract.

    Returns ``(accepted, exclusion_reason)``.

    >>> classify_renouvellement("renouvellement du marche de nettoyage", 15)[0]
    True
    >>> classify_renouvellement("renouvellement des canalisations d eau", 15)[0]
    False
    """
    # "maintenance des equipements scenographiques, et leur renouvellement":
    # the possessive points back at an asset, not forward at a contract.
    trigger_start = match_end if match_start is None else match_start
    if _POSSESSIVE_TRIGGER.search(text[max(0, trigger_start - 12): trigger_start]) \
            and _ASSET_HEAD_RE.search(text[max(0, trigger_start - 70): trigger_start]):
        return False, EXCLUDED_ASSET_RENEWAL

    head = contract_head_after(text, match_end, window)
    if head is None:
        return False, EXCLUDED_ASSET_RENEWAL
    blocker = asset_head_between(text, match_end, head.start())
    if blocker is not None:
        return False, EXCLUDED_ASSET_RENEWAL
    # "renouvellement des marches a venir" has no predecessor to point at.
    if _FUTURE_CONTRACTS.search(text, head.end(), min(len(text), head.end() + 30)):
        return False, EXCLUDED_FUTURE_CONTRACT
    # "le renouvellement est exclu du present marche" states scope, not intent.
    if _SCOPE_EXCLUSION.search(text, match_end, min(len(text), match_end + 40)):
        return False, EXCLUDED_SCOPE_STATEMENT
    if _HEAD_IN_SCOPE_PHRASE.search(text[max(0, head.start() - 45): head.start()]):
        return False, EXCLUDED_SCOPE_STATEMENT
    return True, ""


_ADVISORY_RE = re.compile(rf"\b(?:{ADVISORY_HEAD})", re.IGNORECASE)
_CEMETERY_RE = re.compile(rf"\b(?:{CEMETERY_CONTEXT})", re.IGNORECASE)
_OWN_OPTION_RE = re.compile(rf"\b(?:{OWN_OPTION_CONTEXT})", re.IGNORECASE)

#: How far back an advisory head may sit and still govern the trigger.
ADVISORY_LOOKBACK = 160

#: Window either side of a trigger for own-option wording. Wide enough to
#: reach "reconductible expressement trois fois ... Le renouvellement du marche
#: par le pouvoir adjudicateur prendra la forme d'un courrier".
OWN_OPTION_WINDOW = 120


#: Advisory lookback uses a looser boundary than the general one. Titles
#: routinely read "AMO: Renouvellement du marche X" or "Appui juridique -
#: renouvellement de la DSP", and treating ":" or ";" as a sentence end would
#: hide the advisory head from the trigger it governs.
_HARD_SENTENCE_SPLIT = re.compile(r"[.!?\n\r]")


def advisory_mission_before(text: str, position: int,
                            lookback: int = ADVISORY_LOOKBACK) -> re.Match[str] | None:
    """An advisory head governing the trigger, inside the same sentence.

    "Assistance a maitrise d'ouvrage pour le renouvellement du marche X"
    declares a genuine renewal of X, but the notice being read is the adviser's
    own contract. Recruiting it as the successor of X would be wrong.
    """
    start = 0
    for separator in _HARD_SENTENCE_SPLIT.finditer(text, 0, position):
        start = separator.end()
    low = max(start, position - lookback)
    if position <= low:
        return None
    return _ADVISORY_RE.search(text, low, position)


def cemetery_context(text: str, position: int) -> bool:
    """Whether the sentence is about burial plots rather than public contracts."""
    start, end = sentence_span(text, position)
    return _CEMETERY_RE.search(text, start, end) is not None


def own_option_context(text: str, start: int, end: int,
                       window: int = OWN_OPTION_WINDOW) -> bool:
    """Whether a `renouvellement` is this contract's own extension option."""
    low = max(0, start - window)
    high = min(len(text), end + window)
    return _OWN_OPTION_RE.search(text, low, high) is not None


def works_renewal(text: str, start: int) -> bool:
    """Whether the trigger is a works item ("travaux de renouvellement")."""
    return _WORKS_PREFIX.search(text[max(0, start - 40): start]) is not None


def is_modification_boilerplate(text: str, start: int, end: int) -> bool:
    """Whether a match sits inside amendment boilerplate rather than a decision.

    ``marche en cours`` is overwhelmingly "modifications du marche en cours
    d'execution", the standard clause about amending the current contract.
    """
    if _MODIFICATION_BOILERPLATE.search(
        text, max(0, start - MODIFICATION_WINDOW), min(len(text), end + MODIFICATION_WINDOW)
    ):
        return True
    fragment = text[max(0, start - MODIFICATION_WINDOW): min(len(text), end + MODIFICATION_WINDOW)]
    if "en cours" in _fold(text[start:end]) and _MODIFICATION_NEARBY.search(fragment):
        return True
    return False


def find_matches(
    text: Any,
    field: str,
    patterns: Sequence[Pattern] = PATTERNS,
    window: int = CONTRACT_HEAD_WINDOW,
    keep_excluded: bool = True,
) -> list[Match]:
    """Locate every declaration in one field of one notice.

    Excluded matches are returned by default with their reason set, so the
    exclusion rate can be measured instead of assumed.
    """
    if text is None:
        return []
    raw = str(text)
    if not raw.strip():
        return []

    found: list[Match] = []
    for pattern in patterns:
        for hit in pattern.regex.finditer(raw):
            reason = ""
            if pattern.requires_contract_head:
                if _NEGATED_RENEWAL.search(raw[max(0, hit.start() - 5): hit.start()]):
                    reason = EXCLUDED_NEGATED
                elif works_renewal(raw, hit.start()):
                    reason = EXCLUDED_ASSET_RENEWAL
                elif own_option_context(raw, hit.start(), hit.end()):
                    reason = EXCLUDED_OWN_OPTION
                elif APPLICATION_CONTEXT.search(
                    raw, max(0, hit.start() - 40), min(len(raw), hit.end() + 40)
                ):
                    reason = EXCLUDED_APPLICATION
                else:
                    accepted, reason = classify_renouvellement(
                        raw, hit.end(), window, match_start=hit.start()
                    )
            # Guards that apply to every family. Order matters only for which
            # reason is reported; each is sufficient on its own.
            if not reason and advisory_mission_before(raw, hit.start()):
                reason = EXCLUDED_ADVISORY_MISSION
            if not reason and cemetery_context(raw, hit.start()):
                reason = EXCLUDED_CEMETERY
            if not reason and ACCOUNTING_ECHU.search(
                raw, max(0, hit.start() - 30), min(len(raw), hit.end() + 30)
            ):
                reason = EXCLUDED_ACCOUNTING_TERM
            if not reason and is_modification_boilerplate(raw, hit.start(), hit.end()):
                reason = EXCLUDED_MODIFICATION_BOILERPLATE
            if reason and not keep_excluded:
                continue
            found.append(
                Match(
                    pattern=pattern.name,
                    klass=pattern.klass,
                    field=field,
                    start=hit.start(),
                    end=hit.end(),
                    matched_text=raw[hit.start(): hit.end()],
                    snippet=snippet_for(raw, hit.start(), hit.end()),
                    excluded_reason=reason,
                )
            )
    return found


# ---------------------------------------------------------------------------
# Predecessor attributes stated in the text
# ---------------------------------------------------------------------------

_MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12,
}

_DATE_TEXTUAL = re.compile(
    r"\b(\d{1,2})(?:\s*(?:er|ER))?\s+"
    r"(janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|"
    r"octobre|novembre|d[eé]cembre)\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_NUMERIC = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b")
_DATE_MONTH_YEAR = re.compile(
    r"\b(?:fin|d[eé]but|courant|mi)?\s*"
    r"(janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|"
    r"octobre|novembre|d[eé]cembre)\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_YEAR_ONLY = re.compile(r"\b(?:fin|courant|d[eé]but)\s+(20\d{2})\b", re.IGNORECASE)

#: How far from an expiry trigger a date may sit and still be taken as that
#: contract's end. Beyond this the association is not credible.
DATE_BINDING_WINDOW = 120


@dataclass(frozen=True)
class DeclaredDate:
    value: date
    precision: str  # "day" | "month" | "year"
    start: int
    end: int
    snippet: str


def _month_number(name: str) -> int:
    return _MONTHS[_fold(name)]


def extract_dates(text: str, near: int | None = None,
                  window: int = DATE_BINDING_WINDOW) -> list[DeclaredDate]:
    """Dates stated in the text, optionally confined near an expiry trigger.

    Precision is recorded because it drives how tightly a declared expiry may
    be matched against a candidate predecessor's computed end date.

    >>> [d.value.isoformat() for d in extract_dates("arrive a echeance le 15 juillet 2024")]
    ['2024-07-15']
    """
    if near is not None:
        sentence_start, sentence_end = sentence_span(text, near)
        low = max(sentence_start, near - window)
        high = min(sentence_end, near + window)
    else:
        low, high = 0, len(text)
    if high <= low:
        return []
    region = text[low:high]

    found: list[DeclaredDate] = []
    seen: set[tuple[int, int]] = set()

    for hit in _DATE_TEXTUAL.finditer(region):
        day, month, year = int(hit.group(1)), _month_number(hit.group(2)), int(hit.group(3))
        try:
            value = date(year, month, day)
        except ValueError:
            continue
        span = (low + hit.start(), low + hit.end())
        seen.add(span)
        found.append(DeclaredDate(value, "day", *span, snippet_for(text, *span, radius=80)))

    for hit in _DATE_NUMERIC.finditer(region):
        day, month, year = int(hit.group(1)), int(hit.group(2)), int(hit.group(3))
        try:
            value = date(year, month, day)
        except ValueError:
            continue
        span = (low + hit.start(), low + hit.end())
        seen.add(span)
        found.append(DeclaredDate(value, "day", *span, snippet_for(text, *span, radius=80)))

    for hit in _DATE_MONTH_YEAR.finditer(region):
        span = (low + hit.start(), low + hit.end())
        if any(span[0] >= s and span[1] <= e for s, e in seen):
            continue
        month, year = _month_number(hit.group(1)), int(hit.group(2))
        # Month precision resolves to the last day of that month, since these
        # phrases ("fin decembre 2023") describe a contract ending.
        following = date(year + (month == 12), (month % 12) + 1, 1)
        value = date.fromordinal(following.toordinal() - 1)
        found.append(DeclaredDate(value, "month", *span, snippet_for(text, *span, radius=80)))

    for hit in _DATE_YEAR_ONLY.finditer(region):
        span = (low + hit.start(), low + hit.end())
        if any(span[0] >= s and span[1] <= e for s, e in seen):
            continue
        year = int(hit.group(1))
        found.append(
            DeclaredDate(date(year, 12, 31), "year", *span, snippet_for(text, *span, radius=80))
        )

    return sorted(found, key=lambda item: item.start)


#: A reference marker is "n" followed by a degree sign ("n° 2024007"), the
#: abbreviation "no" ("no 15-144845"), or -- when neither is typed -- directly
#: by a digit. The digit lookahead is what keeps the bare-"n" branch safe:
#: without it, "marche nettoyage" would yield a reference of "ettoyage".
_REFERENCE = re.compile(
    r"(?:march[eé]|consultation|proc[eé]dure|accord[-\s]cadre|affaire|dossier)"
    r"\s*(?:public\s*)?"
    r"n\s*(?:[°ºo]\s*|(?=\d))"
    r"([A-Za-z0-9][A-Za-z0-9\-_/\.]{2,30})",
    re.IGNORECASE,
)


def extract_references(text: str, near: int | None = None,
                       window: int = DATE_BINDING_WINDOW) -> list[tuple[str, int, int]]:
    """Explicit predecessor reference numbers stated in the text.

    >>> extract_references("fait suite a la procedure n° 2024007 declaree sans suite")[0][0]
    '2024007'
    """
    if near is not None:
        low = max(0, near - window)
        high = min(len(text), near + window)
    else:
        low, high = 0, len(text)
    return [
        (hit.group(1).strip(" .,;"), low + hit.start(1), low + hit.end(1))
        for hit in _REFERENCE.finditer(text[low:high])
    ]


def objet_is_advisory(objet: Any) -> bool:
    """Whether a notice's own title shows it is an advisory contract.

    The per-match guard only sees one sentence, so an advisory head stated in
    the title can be missed by a trigger further down the body: "prestation
    d'Assistance a Maitrise d'Ouvrage pour le renouvellement de contrats de
    concession. Dans le cadre du renouvellement ..." -- the second sentence
    survives on its own. A notice's title is the reliable statement of what the
    contract *is*, so when the title's own renewal trigger is governed by an
    advisory head, every match in that notice belongs to an adviser's contract.

    Deliberately not a bare keyword test: "renouvellement du marche
    d'assistance technique" is a genuine renewal whose subject matter happens
    to be assistance, and the advisory word follows the trigger there.
    """
    if not objet:
        return False
    raw = str(objet)
    # Checked against the advisory context directly rather than against the
    # reported exclusion reason: a trigger can be filtered as asset renewal
    # first, which would hide the advisory head behind the other verdict.
    return any(
        advisory_mission_before(raw, match.start) is not None
        for match in find_matches(raw, "objet")
    )


def declaration_strength(matches: Iterable[Match]) -> int:
    """How many distinct positive families a notice carries.

    A notice saying both "renouvellement du marche" and "arrive a echeance le
    ..." is far better evidence than one matching a single weak family, so the
    count drives sampling priority.
    """
    return len({
        match.klass for match in matches
        if not match.is_excluded and match.klass in POSITIVE_CLASSES
    })
