from app.terms.glossary_index import GlossaryRow, MVP_CATEGORIES, build_rows, load_merged
from app.terms.mask import MaskEntry, RestoreReport, find_term_matches_ko, mask_text, restore_text
from app.terms.matcher import TermMatch, apply_matches, find_term_matches
from app.terms.policy import CONFIDENCE_GATED, FULL_MASK, filter_ko_terms_for_lang, is_gated, load_allowlist, load_policy

__all__ = [
    "GlossaryRow",
    "MVP_CATEGORIES",
    "build_rows",
    "load_merged",
    "TermMatch",
    "apply_matches",
    "find_term_matches",
    "MaskEntry",
    "RestoreReport",
    "find_term_matches_ko",
    "mask_text",
    "restore_text",
    "FULL_MASK",
    "CONFIDENCE_GATED",
    "load_policy",
    "load_allowlist",
    "is_gated",
    "filter_ko_terms_for_lang",
]
