"""Numeric-claim grounding: is every number in an answer backed by a tool result?

Pipeline
--------
1. **Citations.** ``[r:<id>]`` tags (several ids may share one bracket, separated
   by commas) are located, then blanked so that digits inside ids never count
   as numbers. Adjacent tags separated only by spaces, commas or semicolons form
   one group.
2. **Numbers.** Signed decimals, thousands separators and a leading ``$`` are
   recognized, with the units ``%``/``percent``/``pct``, percentage points
   (``pp``, ``ppt``, ``percentage points``), ``bp``/``bps``/``basis points``
   and multiples (``1.8x``), and the magnitude suffixes ``k``, ``M``/``mm``,
   ``B``/``bn``, ``T``/``tn`` and ``thousand``/``million``/``billion``/
   ``trillion`` (``$12.5k``, ``9.9M shares``). Minus signs may be ASCII,
   U+2212, an en or figure dash, or a full-width or small hyphen-minus.
   An unsigned number takes a **verbal sign** from an adjacent cue: ``down``,
   ``fell``, ``a loss of``, ``declined by`` or a trailing ``decline``/``lower``
   make it negative; ``up``, ``rose``, ``gained``, ``a gain of`` make it
   positive. Cues must touch the number, allowing only filler words such as
   ``by`` or ``of``, so ``fell to $105`` (a level) takes no sign.
3. **Exclusions.** A number is ignored, with the reason recorded, when it is
   * part of a date: ISO ``2023-06-30``, ``2023/06/30``, ``6/30/2023``, or a
     month-name form (``June 30, 2023``, ``30 June 2023``, ``Jun 2023``). A
     clock time ``10:30`` is also ignored;
   * a year: a bare four-digit integer from 1900 to 2100 with no sign, unit,
     separator or decimals;
   * a window or horizon: an integer followed by a horizon noun
     (``20-day``, ``63 trading days``, ``252 sessions``, ``3 months``,
     ``60 returns``) or preceded by ``window``/``lookback``/``horizon``;
   * a parameter written like a function argument: ``vol(20)``, ``MA(50)``;
   * an identifier: an integer attached to letters, underscores, ``/`` or
     ``:``, or attached through a hyphen or plus sign to a letter
     (``SYN01``, ``1st``, ``Q2``, ``t+1``, ``p-3f2a``);
   * an ordinal or list marker: ``rank 1``, ``#2``, ``No. 3``, or ``1.`` at
     the start of a line;
   * a convention constant: ``sqrt(252)``, ``√252``;
   * a restatement of the question: the same value and unit (sign and trailing
     zeros ignored) appears in the question text.
   A **decimal** number glued to an unrecognized letter suffix (``7.3zz``,
   ``1.5e-3``), or a digit run longer than 18 digits, is not ignored: it is an
   ``unparsed`` claim, which counts as a claim and is never supported.
   Every remaining number is a **claim**.
4. **Attribution.** A claim belongs to the first citation group that follows it
   in the same sentence. This is the "number, then ``[r:id]``" style the system
   prompt requires, and it means each citation covers the claims before it. If
   no group follows the claim in its sentence, the nearest group before it in
   the same sentence is used ("per [r:id], ..."). Otherwise the claim is uncited.
   Sentences end at a newline or at ``.``/``!``/``?`` followed by whitespace.
5. **Support.** A cited claim must match a numeric output
   (:attr:`Provenance.outputs`) of one of the results it cites. An uncited claim
   may match any result of the episode. A claim that cites only unknown ids is
   ``unknown_citation`` (fabricated or wrong id), which counts as unsupported.
   The match is rounding-aware. With ``d`` decimals shown and magnitude
   multiplier ``m``, the claim matches output ``v`` when
   ``|s*v - c| <= 0.5 * 10**-d * m`` (plus float noise), so the claim is a
   correct rounding of the value under any tie rule. The scale ``s`` is 100
   for percent and percentage points (tool outputs are fractions), 10,000 for
   basis points, 1 for dollars and multiples, and 1 or 100 for plain numbers,
   since models often drop the ``%`` sign.
6. **Sign.** An explicit or verbal sign must agree with the output. An
   unsigned number matches only a non-negative output, except for outputs
   conventionally reported as a magnitude (``max_drawdown``: "a drawdown of
   18.20%"). A number that would match only with the opposite sign is
   unsupported with the note ``sign_mismatch``.
7. **Binding.** Daily-bar outputs, where one result holds several fields and
   rows, are bound to the text. A row-level output such as
   ``close[2023-06-29]`` supports a claim only if its date is named in the
   claim's sentence or in the question. A daily-bar output with a bar field
   (the ``close``/``open``/``high``/``low``/``volume`` of a row, or
   ``first_close``, ``last_close``, ``max_high``, ``min_low``,
   ``mean_volume``) supports a claim only if the sentence, or failing that the
   question, names no bar field or names that field. ``first_close`` and
   ``last_close`` are also bound to their end of the range: when the sentence
   (or else the question) speaks only of the latest ("most recent", "last",
   ...) or only of the earliest ("first", "initial", ...) session, the other
   end is out of scope. A number that matches only an out-of-scope output is
   unsupported with the note ``date_mismatch``, ``field_mismatch`` or
   ``position_mismatch``.

Known limits (see ``tests/test_grounding.py``): numbers written in words are
not extracted. A derived number that no tool reported (for example a difference
of two returns, in percentage points) is unsupported by construction. A
four-digit quantity from 1900 to 2100 ("2000 shares") is treated as a year.
Binding covers bar fields, dates and range ends of daily-bar results only: a
claim may still match an output of the right type for the wrong symbol or
statistic (a volatility reported under another symbol's name, or a simple
return that happens to round to the log return of the same result). Coarsely
rounded claims (``5%``, a correlation of ``0.4``) and small counts can match
an unrelated output by chance; ``scripts/verifier_stress.py`` measures how often.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import math
import re
from typing import Any, Literal


Unit = Literal["plain", "percent", "pp", "bp", "usd", "multiple", "unparsed"]
ClaimStatus = Literal["supported", "unsupported", "unknown_citation", "unparsed"]
SignSource = Literal["explicit", "verbal"]

MAX_DIGITS = 18
MAGNITUDE_OUTPUTS = frozenset({"max_drawdown"})
# Daily-bar outputs, where several bar fields and rows of one result compete for the same number.
FIELD_OF_OUTPUT = {
    "first_close": "close",
    "last_close": "close",
    "min_low": "low",
    "max_high": "high",
    "mean_volume": "volume",
}
BAR_FIELDS = frozenset({"open", "high", "low", "close", "volume"})
# Summary outputs that belong to one end of the requested range.
POSITION_OF_OUTPUT = {"first_close": "first", "last_close": "last"}

_MINUS = r"\-" + "\u2212\u2013\u2012\ufe63\uff0d"  # hyphen escaped: a bare "-" inside [...] makes a range
_CITATION = re.compile(r"\[r:([^\]]*)\]")
_ID_SPLIT = re.compile(r"[\s,;]+")
_GROUP_GAP = re.compile(r"^[\s,;]*$")
_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("date", re.compile(r"(?<!\d)(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?(?!\d)")),
    ("date", re.compile(r"(?<!\d)(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})(?!\d)")),
    ("date", re.compile(r"(?<![\d/])(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{2,4})(?![\d/])")),
    ("date", re.compile(rf"\b(?P<mon>{_MONTH})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{{4}}))?\b")),
    ("date", re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<mon>{_MONTH})\.?(?:,?\s+(?P<y>\d{{4}}))?\b")),
    ("date", re.compile(rf"\b(?P<mon>{_MONTH})\.?,?\s+(?P<y>\d{{4}})\b")),
    ("time", re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")),
)
_NUMBER = re.compile(
    rf"(?P<sign>[+{_MINUS}])?(?P<cur>\$)?"
    r"(?P<body>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<mult>(?:[kKMBT]|mm|MM|bn|tn)(?![A-Za-z0-9_])|\s(?i:thousand|million|billion|trillion)\b)?"
    r"(?P<unit>\s?%|\s?(?i:percentage\s+points?|ppts?|pp)\b|\s?(?i:percent|pct)\b"
    r"|\s?(?i:bps?|basis\s+points?)\b|x(?![A-Za-z0-9_])|×)?"
)
_MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "t": 1e12, "tn": 1e12, "trillion": 1e12,
}
_HORIZON_AFTER = re.compile(
    r"^[\s\-]*(?:(?:trading|calendar|business|consecutive|daily|weekly|monthly)[\s\-]+)?"
    r"(?:day|session|week|month|quarter|year|return|observation|obs|bar|period|lag)s?\b",
    re.IGNORECASE,
)
_HORIZON_BEFORE = re.compile(r"(?:window|lookback|horizon)(?:\s+(?:of|length|size))?\s*[=:]?\s*$", re.IGNORECASE)
_ORDINAL_BEFORE = re.compile(r"(?:\brank(?:ed)?|\bno\.|#|\bnumber)\s*$", re.IGNORECASE)
_CONSTANT_BEFORE = re.compile(r"(?:sqrt\s*\(|√)\s*$", re.IGNORECASE)
_PARAMETER_BEFORE = re.compile(r"[A-Za-z_]\w*\(\s*$")
_LIST_MARKER = re.compile(r"^\s*$")
_SENTENCE_END = re.compile(r"\n|[.!?](?=\s|$)")
_FILLER = r"(?:\s+(?:by|of|about|around|roughly|approximately|nearly|almost|some|just|over|under|an?|another|further))*"
_CUE_NOUN = r"(?:\s+(?:return|correlation|change|move|performance))?"
_NEG_WORDS = (
    r"down|fell|falls?|fallen|falling|dropp(?:ed|ing)|drops?|declin(?:e|ed|es|ing)|decreas(?:e|ed|es|ing)|"
    r"loss(?:es)?|lost|losing|lower|negative|minus|slid|slipped|sank|sunk|slump(?:ed|s)?|"
    r"plung(?:e|ed|es)|tumbl(?:e|ed|es)|shed|shrank|contracted"
)
_POS_WORDS = (
    r"up|rose|rises?|risen|rising|gain(?:ed|s|ing)?|increas(?:e|ed|es|ing)|grew|grow(?:s|n|ing)?|"
    r"climb(?:ed|s|ing)?|advanc(?:e|ed|es|ing)|rall(?:y|ied|ies)|added|higher|positive|plus|jumped|surged"
)
_NEG_BEFORE = re.compile(rf"\b(?:{_NEG_WORDS}){_CUE_NOUN}{_FILLER}\s*$", re.IGNORECASE)
_POS_BEFORE = re.compile(rf"\b(?:{_POS_WORDS}){_CUE_NOUN}{_FILLER}\s*$", re.IGNORECASE)
_NEG_AFTER = re.compile(r"^\s*(?:decline|drop|loss|decrease|fall|lower|down)\b", re.IGNORECASE)
_POS_AFTER = re.compile(r"^\s*(?:gain|increase|rise|higher|up|advance)\b", re.IGNORECASE)
_FIELD_WORD = re.compile(r"\b(open|opening|opened|high|highs|low|lows|close|closes|closed|closing|volume|volumes)\b", re.I)
_FIELD_OF_WORD = {
    "open": "open", "opening": "open", "opened": "open",
    "high": "high", "highs": "high", "low": "low", "lows": "low",
    "close": "close", "closes": "close", "closed": "close", "closing": "close",
    "volume": "volume", "volumes": "volume",
}
_LAST_WORD = re.compile(r"\b(?:most\s+recent|latest|last|current|final|ending)\b", re.IGNORECASE)
_FIRST_WORD = re.compile(r"\b(?:first|earliest|initial|starting|beginning)\b", re.IGNORECASE)
_ROW_OUTPUT = re.compile(r"^(?P<field>open|high|low|close|volume)\[(?P<date>\d{4}-\d{2}-\d{2})\]$")
_FLOAT_NOISE = 1e-9


@dataclass(frozen=True)
class NumericClaim:
    """A number as written in the answer."""

    text: str
    value: float
    decimals: int
    unit: Unit
    signed: bool
    start: int
    end: int
    sentence: int
    multiplier: float = 1.0
    sign_source: SignSource | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "value": self.value if math.isfinite(self.value) else None,
            "decimals": self.decimals,
            "unit": self.unit,
            "signed": self.signed,
            "sign_source": self.sign_source,
            "multiplier": self.multiplier,
            "start": self.start,
        }


@dataclass(frozen=True)
class IgnoredNumber:
    """A number excluded from grounding, with the heuristic that excluded it."""

    text: str
    reason: str
    start: int

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "reason": self.reason, "start": self.start}


@dataclass(frozen=True)
class CitationGroup:
    """One or more adjacent ``[r:id]`` tags."""

    result_ids: tuple[str, ...]
    start: int
    end: int
    sentence: int


@dataclass(frozen=True)
class ClaimCheck:
    """Verdict for one claim. ``note`` explains a near miss (sign, date or field)."""

    claim: NumericClaim
    cited_ids: tuple[str, ...]
    status: ClaimStatus
    matched_result: str | None = None
    matched_output: str | None = None
    note: str | None = None

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    @property
    def cited(self) -> bool:
        return bool(self.cited_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.claim.to_dict(),
            "cited_ids": list(self.cited_ids),
            "status": self.status,
            "matched_result": self.matched_result,
            "matched_output": self.matched_output,
            "note": self.note,
        }


@dataclass(frozen=True)
class GroundingReport:
    """Claim-level grounding of one answer against one episode's results.

    ``error`` is set only when the verifier itself failed; such an episode is
    treated as not grounded.
    """

    checks: tuple[ClaimCheck, ...]
    ignored: tuple[IgnoredNumber, ...]
    cited_ids: tuple[str, ...]
    unknown_citations: tuple[str, ...]
    error: str | None = None

    @classmethod
    def failure(cls, error: str) -> "GroundingReport":
        return cls((), (), (), (), error)

    @property
    def n_claims(self) -> int:
        return len(self.checks)

    @property
    def n_supported(self) -> int:
        return sum(check.supported for check in self.checks)

    @property
    def n_cited(self) -> int:
        return sum(check.cited for check in self.checks)

    @property
    def n_cited_supported(self) -> int:
        return sum(check.cited and check.supported for check in self.checks)

    @property
    def n_unparsed(self) -> int:
        return sum(check.status == "unparsed" for check in self.checks)

    @property
    def fully_grounded(self) -> bool | None:
        """True if every claim is supported; None when there is nothing to judge."""

        if self.error is not None:
            return False
        return None if not self.checks else self.n_supported == self.n_claims

    @property
    def grounding_rate(self) -> float | None:
        """Share of claims that match a tool output (cited result, or any result if uncited)."""

        return self.n_supported / self.n_claims if self.checks else None

    @property
    def citation_rate(self) -> float | None:
        """Share of claims attributed to a citation."""

        return self.n_cited / self.n_claims if self.checks else None

    @property
    def cited_grounding_rate(self) -> float | None:
        """Share of claims that are cited *and* match the result they cite."""

        return self.n_cited_supported / self.n_claims if self.checks else None

    @property
    def unsupported(self) -> tuple[ClaimCheck, ...]:
        return tuple(check for check in self.checks if not check.supported)

    def notes(self) -> dict[str, int]:
        return dict(sorted(Counter(check.note for check in self.checks if check.note).items()))

    def summary(self) -> dict[str, Any]:
        return {
            "n_claims": self.n_claims,
            "n_supported": self.n_supported,
            "n_cited": self.n_cited,
            "n_cited_supported": self.n_cited_supported,
            "n_unparsed": self.n_unparsed,
            "grounding_rate": self.grounding_rate,
            "citation_rate": self.citation_rate,
            "cited_grounding_rate": self.cited_grounding_rate,
            "cited_ids": list(self.cited_ids),
            "unknown_citations": list(self.unknown_citations),
            "unsupported": [check.claim.text for check in self.unsupported],
            "notes": self.notes(),
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "checks": [check.to_dict() for check in self.checks],
            "ignored": [item.to_dict() for item in self.ignored],
        }


# Extraction -------------------------------------------------------------------------


@dataclass(frozen=True)
class _DateMark:
    start: int
    iso: str | None
    month_day: str | None


def _mask(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _sentence_starts(text: str) -> list[int]:
    return [match.end() for match in _SENTENCE_END.finditer(text)]


def _sentence_of(boundaries: list[int], position: int) -> int:
    return bisect_right(boundaries, position)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    bounds = _sentence_starts(text)
    starts = [0, *bounds]
    ends = [*bounds, len(text)]
    return list(zip(starts, ends))


def find_citations(text: str) -> list[CitationGroup]:
    """Locate citation groups (adjacent ``[r:...]`` tags merge into one group)."""

    boundaries = _sentence_starts(text)
    groups: list[CitationGroup] = []
    for match in _CITATION.finditer(text):
        ids = tuple(
            token[2:] if token.lower().startswith("r:") else token
            for token in _ID_SPLIT.split(match.group(1).strip())
            if token
        )
        previous = groups[-1] if groups else None
        if previous is not None and _GROUP_GAP.match(text[previous.end : match.start()]):
            merged = previous.result_ids + tuple(i for i in ids if i not in previous.result_ids)
            groups[-1] = CitationGroup(merged, previous.start, match.end(), previous.sentence)
        else:
            groups.append(CitationGroup(ids, match.start(), match.end(), _sentence_of(boundaries, match.start())))
    return groups


def _unit_of(raw: str | None, currency: bool) -> Unit:
    if currency:
        return "usd"
    if not raw:
        return "plain"
    token = raw.strip().lower()
    if token in {"%", "percent", "pct"}:
        return "percent"
    if token in {"x", "×"}:
        return "multiple"
    if token.startswith("percentage") or token.startswith("pp"):
        return "pp"
    return "bp"


def _multiplier_of(raw: str | None) -> float:
    if not raw:
        return 1.0
    token = raw.strip()
    return _MULTIPLIERS[token.lower()] if token.lower() in _MULTIPLIERS else 1.0


def _normalize_date(match: re.Match[str]) -> tuple[str | None, str | None]:
    """ISO date and ``MM-DD`` for a date match (``None`` for parts that are not stated)."""

    groups = match.groupdict()
    try:
        month = _MONTH_NAMES[groups["mon"][:3].lower()] if groups.get("mon") else int(groups["m"])
        day = int(groups["d"]) if groups.get("d") else None
    except (KeyError, TypeError, ValueError):
        return None, None
    if day is None:
        return None, None
    year_text = groups.get("y")
    month_day = f"{month:02d}-{day:02d}"
    if not year_text:
        return None, month_day
    year = int(year_text) + (2000 if len(year_text) == 2 else 0)
    try:
        return date(year, month, day).isoformat(), month_day
    except ValueError:
        return None, None


def _verbal_sign(masked: str, start: int, end: int) -> int:
    prefix = masked[max(0, start - 60) : start]
    suffix = masked[end : end + 24]
    if _NEG_BEFORE.search(prefix) or _NEG_AFTER.match(suffix):
        return -1
    if _POS_BEFORE.search(prefix) or _POS_AFTER.match(suffix):
        return 1
    return 0


@dataclass(frozen=True)
class _Scan:
    claims: list[NumericClaim]
    ignored: list[IgnoredNumber]
    dates: list[_DateMark]


def _scan(text: str, question: str | None) -> _Scan:
    ignored: list[IgnoredNumber] = []
    marks: list[_DateMark] = []
    citation_spans = [(m.start(), m.end()) for m in _CITATION.finditer(text)]
    masked = _mask(text, citation_spans)
    for reason, pattern in _DATE_PATTERNS:
        spans = []
        for match in pattern.finditer(masked):
            ignored.append(IgnoredNumber(match.group(0), reason, match.start()))
            spans.append((match.start(), match.end()))
            if reason == "date":
                iso, month_day = _normalize_date(match)
                marks.append(_DateMark(match.start(), iso, month_day))
        masked = _mask(masked, spans)

    quoted = _question_keys(question) if question else set()
    boundaries = _sentence_starts(text)
    claims: list[NumericClaim] = []
    for match in _NUMBER.finditer(masked):
        sign, body = match.group("sign"), match.group("body")
        unit_raw, mult_raw = match.group("unit"), match.group("mult")
        start, end = match.start(), match.end()
        before = masked[start - 1] if start > 0 else ""
        after = masked[end] if end < len(masked) else ""
        after2 = masked[end + 1] if end + 1 < len(masked) else ""
        if sign and before and (before.isalnum() or before == "_"):
            if before.isdigit():  # a range such as "5-10%": the hyphen is not a sign
                sign = None
                start = match.start("cur") if match.group("cur") else match.start("body")
                before = masked[start - 1]
            else:
                ignored.append(IgnoredNumber(match.group(0), "identifier", match.start()))
                continue
        literal = text[start:end].strip()
        if before and (before.isalnum() or before in "_/:."):
            ignored.append(IgnoredNumber(literal, "identifier", start))
            continue
        digits = body.replace(",", "").replace(".", "")
        glued = not unit_raw and not mult_raw and after and (after.isalpha() or after == "_")
        sentence = _sentence_of(boundaries, start)
        if len(digits) > MAX_DIGITS or (glued and "." in body):
            claims.append(NumericClaim(literal, math.nan, 0, "unparsed", sign is not None, start, end, sentence))
            continue
        if not unit_raw and not mult_raw and after and (
            after.isalnum() or after in "_/:" or (after == "." and after2.isdigit())
        ):
            ignored.append(IgnoredNumber(literal, "identifier", start))
            continue
        currency = bool(match.group("cur"))
        unit = _unit_of(unit_raw, currency)
        multiplier = _multiplier_of(mult_raw)
        is_integer = "." not in body
        plain_integer = is_integer and unit == "plain" and not sign and multiplier == 1.0
        prefix = masked[max(0, start - 24) : start]
        suffix = masked[end : end + 40]
        magnitude = float(body.replace(",", "")) * multiplier
        reason = None
        if plain_integer and "," not in body and len(body) == 4 and 1900 <= int(body) <= 2100:
            reason = "year"
        elif is_integer and unit == "plain" and multiplier == 1.0 and (
            _HORIZON_AFTER.match(suffix) or _HORIZON_BEFORE.search(prefix)
        ):
            reason = "window"
        elif plain_integer and _CONSTANT_BEFORE.search(prefix):
            reason = "convention_constant"
        elif plain_integer and _PARAMETER_BEFORE.search(prefix):
            reason = "parameter"
        elif plain_integer and _ORDINAL_BEFORE.search(prefix):
            reason = "ordinal"
        elif plain_integer and after != "" and after in ".)" and _line_prefix_blank(masked, start):
            reason = "list_marker"
        elif _key(magnitude, unit) in quoted:
            reason = "quoted_from_question"
        if reason is not None:
            ignored.append(IgnoredNumber(literal, reason, start))
            continue
        sign_source: SignSource | None = None
        negative = sign is not None and sign != "+"
        if sign is not None:
            sign_source = "explicit"
        else:
            cue = _verbal_sign(masked, start, end)
            if cue:
                sign_source, negative = "verbal", cue < 0
        claims.append(
            NumericClaim(
                text=literal,
                value=-magnitude if negative else magnitude,
                decimals=len(body.split(".", 1)[1]) if "." in body else 0,
                unit=unit,
                signed=sign_source is not None,
                start=start,
                end=end,
                sentence=sentence,
                multiplier=multiplier,
                sign_source=sign_source,
            )
        )
    ignored.sort(key=lambda item: item.start)
    marks.sort(key=lambda item: item.start)
    return _Scan(claims, ignored, marks)


def extract_numbers(
    text: str,
    *,
    question: str | None = None,
) -> tuple[list[NumericClaim], list[IgnoredNumber]]:
    """Return ``(claims, ignored)`` for ``text`` using the documented heuristics."""

    scan = _scan(text, question)
    return scan.claims, scan.ignored


def _line_prefix_blank(text: str, position: int) -> bool:
    line_start = text.rfind("\n", 0, position) + 1
    return bool(_LIST_MARKER.match(text[line_start:position]))


def _key(magnitude: float, unit: Unit) -> tuple[Unit, Decimal]:
    """Value-and-unit key used to recognize restated question numbers (sign ignored)."""

    if not math.isfinite(magnitude):
        return unit, Decimal(0)
    return unit, Decimal(repr(abs(magnitude))).normalize()


def _question_keys(question: str) -> set[tuple[Unit, Decimal]]:
    claims, _ = extract_numbers(question)
    return {_key(claim.value, claim.unit) for claim in claims if claim.unit != "unparsed"}


# Support --------------------------------------------------------------------------


def _outputs_of(result: Any) -> Mapping[str, float]:
    provenance = getattr(result, "provenance", None)
    if provenance is not None:
        return provenance.outputs
    if isinstance(result, Mapping):
        return result
    raise TypeError("results must map result ids to ToolResult objects or to {name: value} outputs")


def _scales(unit: Unit) -> tuple[float, ...]:
    if unit in {"percent", "pp"}:
        return (100.0,)
    if unit == "bp":
        return (10_000.0,)
    if unit in {"usd", "multiple"}:
        return (1.0,)
    if unit == "unparsed":
        return ()
    return (1.0, 100.0)


def _base(name: str | None) -> str:
    return (name or "").split("[", 1)[0]


def claim_matches(claim: NumericClaim, value: float, name: str | None = None) -> bool:
    """Rounding-aware match of one claim against one tool output (see module docstring).

    ``name`` is the output's name. An unsigned claim may match the magnitude of
    a negative output only when the output is in :data:`MAGNITUDE_OUTPUTS`.
    """

    if not math.isfinite(claim.value) or not math.isfinite(float(value)):
        return False
    tolerance = 0.5 * 10.0 ** (-claim.decimals) * claim.multiplier
    noise = _FLOAT_NOISE * max(1.0, abs(claim.value))
    magnitude_ok = not claim.signed and _base(name) in MAGNITUDE_OUTPUTS
    for scale in _scales(claim.unit):
        scaled = float(value) * scale
        if not claim.signed and scaled < 0 and not magnitude_ok:
            continue
        candidate = abs(scaled) if magnitude_ok else scaled
        if abs(candidate - claim.value) <= tolerance + noise:
            return True
    return False


def _flipped(claim: NumericClaim, value: float, name: str) -> bool:
    """Would the claim match if the output had the opposite sign?"""

    if not math.isfinite(claim.value) or float(value) == 0.0:
        return False
    if claim.signed:
        return claim_matches(claim, -float(value), name)
    return float(value) < 0 and claim_matches(claim, -float(value), name)


@dataclass(frozen=True)
class _Scope:
    dates: frozenset[str]
    month_days: frozenset[str]
    fields: frozenset[str]
    positions: frozenset[str] = frozenset()

    def admits(self, name: str) -> str | None:
        """``None`` if the output may support a claim in this scope, else the reason it may not."""

        row = _ROW_OUTPUT.match(name)
        field = row.group("field") if row else FIELD_OF_OUTPUT.get(_base(name))
        if row is not None:
            day = row.group("date")
            if day not in self.dates and day[5:] not in self.month_days:
                return "date_mismatch"
        if field is not None and self.fields and field not in self.fields:
            return "field_mismatch"
        position = POSITION_OF_OUTPUT.get(_base(name))
        if position is not None and len(self.positions) == 1 and position not in self.positions:
            return "position_mismatch"
        return None


def _fields_in(text: str) -> frozenset[str]:
    return frozenset(_FIELD_OF_WORD[m.group(1).lower()] for m in _FIELD_WORD.finditer(text))


def _positions_in(text: str) -> frozenset[str]:
    found = set()
    if _LAST_WORD.search(text):
        found.add("last")
    if _FIRST_WORD.search(text):
        found.add("first")
    return frozenset(found)


def _match(
    claim: NumericClaim,
    results: Mapping[str, Any],
    ids: Iterable[str],
    scope: _Scope,
) -> tuple[tuple[str, str] | None, str | None]:
    note: str | None = None
    for rid in ids:
        for name, value in _outputs_of(results[rid]).items():
            if value is None:
                continue
            if claim_matches(claim, value, name):
                reason = scope.admits(name)
                if reason is None:
                    return (rid, name), None
                note = note or reason
            elif note is None and _flipped(claim, value, name):
                note = "sign_mismatch"
    return None, note


def _attribute(claim: NumericClaim, groups: list[CitationGroup]) -> CitationGroup | None:
    same = [group for group in groups if group.sentence == claim.sentence]
    following = [group for group in same if group.start >= claim.end]
    if following:
        return following[0]
    preceding = [group for group in same if group.end <= claim.start]
    return preceding[-1] if preceding else None


def verify_grounding(
    answer: str,
    results: Mapping[str, Any],
    *,
    question: str | None = None,
) -> GroundingReport:
    """Check every numeric claim of ``answer`` against ``results`` (result_id -> ToolResult)."""

    text = answer or ""
    groups = find_citations(text)
    scan = _scan(text, question)
    known = set(results)
    cited_ids = tuple(dict.fromkeys(rid for group in groups for rid in group.result_ids))
    unknown = tuple(rid for rid in cited_ids if rid not in known)

    boundaries = _sentence_starts(text)
    spans = _sentence_spans(_mask(text, [(m.start(), m.end()) for m in _CITATION.finditer(text)]))
    sentence_fields = [_fields_in(text[a:b]) for a, b in spans]
    sentence_positions = [_positions_in(text[a:b]) for a, b in spans]
    sentence_dates: dict[int, list[_DateMark]] = {}
    for mark in scan.dates:
        sentence_dates.setdefault(_sentence_of(boundaries, mark.start), []).append(mark)
    question_marks = _scan(question, None).dates if question else []
    question_fields = _fields_in(question or "")
    question_positions = _positions_in(question or "")

    checks: list[ClaimCheck] = []
    for claim in scan.claims:
        group = _attribute(claim, groups)
        cited = group.result_ids if group is not None else ()
        if claim.unit == "unparsed":
            checks.append(ClaimCheck(claim, cited, "unparsed"))
            continue
        marks = [*sentence_dates.get(claim.sentence, []), *question_marks]
        inside = claim.sentence < len(sentence_fields)
        fields = sentence_fields[claim.sentence] if inside else frozenset()
        positions = sentence_positions[claim.sentence] if inside else frozenset()
        scope = _Scope(
            frozenset(m.iso for m in marks if m.iso),
            frozenset(m.month_day for m in marks if m.month_day),
            fields or question_fields,
            positions or question_positions,
        )
        if group is None:
            match, note = _match(claim, results, sorted(known), scope)
            status: ClaimStatus = "supported" if match else "unsupported"
            checks.append(ClaimCheck(claim, (), status, *(match or (None, None)), note))
            continue
        valid = [rid for rid in group.result_ids if rid in known]
        if not valid:
            checks.append(ClaimCheck(claim, group.result_ids, "unknown_citation"))
            continue
        match, note = _match(claim, results, valid, scope)
        checks.append(
            ClaimCheck(claim, group.result_ids, "supported" if match else "unsupported", *(match or (None, None)), note)
        )
    return GroundingReport(tuple(checks), tuple(scan.ignored), cited_ids, unknown)
