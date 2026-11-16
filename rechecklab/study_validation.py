"""Read-only reliability checks for the already frozen live-study parser.

These checks neither reinterpret saved answers nor change the v1 grader. A
precision failure makes the record unsuitable for fitting or evaluation. A
precision-preserving parser/grader requires a future protocol version.
"""
from __future__ import annotations

from decimal import Context, Decimal, MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, localcontext
import re

from .study_protocol import PARSER_VERSION, PROTOCOL_VERSION, parse_study_answer

AUDIT_VERSION = "frozen-decimal-precision-audit-v1"
FROZEN_DECIMAL_PRECISION = 28
PRECISION_LIMITATION = (
    "Frozen explicit-final-v1 normalization uses Decimal precision 28. "
    "Only exact finite decimal values and fractions exactly representable at "
    "that precision are accepted as reliable numeric evidence. This validation "
    "does not relabel records; precision-preserving grading needs a new version."
)


def audit_numeric_precision(protocol, response, answer_type):
    """Return audit evidence; reject numeric values rounded by the frozen parser.

    Decimal literals are compared without arithmetic or context rounding.
    Fractions are checked by exact multiplication of the parsed value and the
    original denominator. The multiplication uses sufficient coefficient
    precision and a wide exponent range; no approximate fraction expansion or
    tolerance can turn a rounded answer into reliable evidence.
    """
    parser = protocol.get("parser") if isinstance(protocol, dict) else None
    if not isinstance(protocol, dict) or protocol.get("protocol_version") != PROTOCOL_VERSION \
            or not isinstance(parser, dict) or parser.get("version") != PARSER_VERSION:
        raise ValueError("Numeric precision audit requires the frozen explicit-final-v1 study protocol.")
    if answer_type not in ("number", "choice", "open"):
        raise ValueError("Numeric precision audit requires number, choice or open answer_type.")
    evidence = {"audit_version": AUDIT_VERSION, "parser_version": PARSER_VERSION,
                "decimal_context_precision": FROZEN_DECIMAL_PRECISION,
                "applicable": answer_type == "number", "limitation": PRECISION_LIMITATION}
    if answer_type != "number":
        return {**evidence, "status": "NOT_APPLICABLE"}
    message = response.get("message") if isinstance(response, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Numeric precision audit requires the actual saved response content.")
    content = message["content"]
    # Use the frozen context explicitly, without changing the caller's context.
    with localcontext(Context(prec=FROZEN_DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)):
        parsed = parse_study_answer(content, "number")
    if parsed is None:
        return {**evidence, "status": "UNPARSEABLE", "parse_valid": False,
                "exact_value_verified": False,
                "reason": "Frozen parser produced no answer; existing offline grading remains incorrect."}
    marked = re.findall(r"\bFINAL(?:\s+ANSWER)?\s*:\s*([^\n]+)", content, re.I)
    candidate = marked[-1].strip().strip("*` ")
    candidate = re.sub(r"^[$£₹€]\s*", "", candidate).rstrip(".").replace(",", "")
    parsed_decimal = Decimal(parsed)
    if "/" in candidate:
        numerator, denominator = (Decimal(part.strip()) for part in candidate.split("/"))
        # The product's coefficient has at most the sum of coefficient lengths.
        # This avoids context-rounded division and huge exponent-expanded ints.
        coefficient_digits = len(parsed_decimal.as_tuple().digits) + len(denominator.as_tuple().digits) + 1
        with localcontext() as exact_context:
            exact_context.prec = max(FROZEN_DECIMAL_PRECISION, coefficient_digits)
            exact_context.Emax, exact_context.Emin = MAX_EMAX, MIN_EMIN
            exact_product = parsed_decimal * denominator
        if exact_product != numerator:
            raise ValueError(
                "Frozen numeric parser precision loss: fraction is nonterminating or not exactly "
                "representable at Decimal precision 28; this record cannot support reliable "
                "training/evaluation labels. A new parser/grader protocol version is required."
            )
        numeric_form = "fraction"
    else:
        exact_decimal = Decimal(candidate)
        if parsed_decimal != exact_decimal:
            raise ValueError(
                "Frozen numeric parser precision loss: Decimal precision 28 rounded the exact "
                "marked answer; this record cannot support reliable training/evaluation labels. "
                "A new parser/grader protocol version is required."
            )
        numeric_form = "decimal"
    return {**evidence, "status": "EXACT", "parse_valid": True,
            "exact_value_verified": True, "numeric_form": numeric_form}
