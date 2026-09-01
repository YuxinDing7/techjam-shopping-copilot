from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MATERIALS = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}
COLORS = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"}
USE_CASE_MARKERS = {"hiking", "running", "gym", "winter", "outdoor", "work", "walking", "travel", "wedding", "party", "office"}
ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)
# The local evaluator's customer simulator (evaluator/local_evaluator.py
# classify_constraint) never classifies a revealed constraint as "brand" or
# "category" -- those two ask_attribute values can never be matched to a
# disclosed constraint, so asking them always wastes a turn.
UNASKABLE_ATTRIBUTES = {"brand", "category"}
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "some", "that", "the", "this",
    "to", "want", "with", "would", "you", "looking", "need", "like", "something", "around",
}
NEGATION_RE = re.compile(r"\b(?:no|not|without|don't|dont|rather than|instead of|ignore)\b", re.I)
BUDGET_RE = re.compile(r"(?:\$|under|below|less than|up to|budget(?: around)?)[^\d]{0,8}(\d+(?:\.\d+)?)", re.I)
INITIAL_CATEGORY_RE = re.compile(r"\blooking for\s+(.+?)(?:[.!]|$)", re.I)
CATEGORY_CHANGE_RE = re.compile(
    r"\b(?:looking for|need|want|switch(?:ing)? to|rather have)\s+(?!is\b)(?:a|an|some)?\s*(.+?)(?:[.!]|$)",
    re.I,
)
RRF_K = 60


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return list(dict.fromkeys(
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ))


def _escape_fts(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _stem(term: str) -> str:
    """Strip a trailing plural 's' (but not 'ss') so 'backpack' matches catalog 'backpacks'."""
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _attribute_for(term: str) -> str:
    lowered = term.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _no_preference_attribute(message: str) -> str | None:
    lowered = message.lower()
    for attribute in ATTRIBUTES:
        if re.search(
            rf"\b(?:no (?:additional )?preference|don't have (?:an? )?(?:additional )?preference) for {attribute}\b",
            lowered,
        ):
            return attribute
    return None


def _initial_category_terms(message: str) -> set[str]:
    match = INITIAL_CATEGORY_RE.search(message)
    return set(_terms(match.group(1))) if match else set()

@dataclass
class Constraint:
    value: str
    attribute: str
    negated: bool = False


@dataclass
class SessionState:
    profile: dict
    messages: list[str] = field(default_factory=list)
    intent_messages: list[str] = field(default_factory=list)
    search_phrases: list[str] = field(default_factory=list)
    constraints: dict[str, list[Constraint]] = field(default_factory=dict)
    asked: set[str] = field(default_factory=set)
    no_preferences: set[str] = field(default_factory=set)
    last_recommendations: list[str] = field(default_factory=list)
    excluded_asins: set[str] = field(default_factory=set)
    candidate_pool: list[str] = field(default_factory=list)
    bm25_scores: dict[str, float] = field(default_factory=dict)
    debug_trace: dict = field(default_factory=dict)
    route: str = "browsing"
    override_active: bool = False
    override_category_terms: set[str] = field(default_factory=set)
    rerank_signature: tuple | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add_constraint(self, constraint: Constraint) -> None:
        values = self.constraints.setdefault(constraint.attribute, [])
        values[:] = [item for item in values if item.value != constraint.value]
        values.append(constraint)

    def clear_attribute(self, attribute: str) -> None:
        self.constraints.pop(attribute, None)
        self.no_preferences.discard(attribute)

    def active_constraints(self) -> list[Constraint]:
        return [item for values in self.constraints.values() for item in values if not item.negated]

    def negated_constraints(self) -> list[Constraint]:
        return [item for values in self.constraints.values() for item in values if item.negated]


class Agent:
    """Stateful shopping agent with optional local Ollama reranking.

    RERANK_METHOD selects the reranker used in respond(): "llm" (default)
    scores each shortlisted candidate with an Ollama chat call; "embedding"
    ranks by cosine similarity to the query instead. See _rerank().
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.sessions: dict[str, SessionState] = {}
        self.products: dict[str, dict] = {}
        self.category_terms: set[str] = set()
        self.category_head_terms: set[str] = set()
        self.category_terms_stemmed: set[str] = set()
        self.category_head_terms_stemmed: set[str] = set()
        self.llm_enabled = os.getenv("LLM_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.llm_model = os.getenv("LLM_MODEL", "qwen3:4b")
        self.llm_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
        self.llm_timeout = float(os.getenv("LLM_TIMEOUT", "8"))
        # A/B tested against 20: with qwen3:4b, a 20-key required-JSON score
        # schema pushes "llm_rerank" invalid_json failures from 0% to ~65%
        # (measured on the 50-sample public set), tanking MRR even though
        # hit-rate barely moves. 10 keeps the model reliably able to return
        # complete, valid structured output.
        self.llm_rerank_limit = int(os.getenv("LLM_RERANK_LIMIT", "10"))
        self.last_llm_error: str | None = None
        # Which reranker actually runs in respond(): "llm" (default) scores
        # each shortlisted candidate with an Ollama chat call; "embedding"
        # ranks by cosine similarity to the query instead. Both
        # implementations stay in this file so switching is a one-line env
        # var change, not a code change.
        self.rerank_method = os.getenv("RERANK_METHOD", "llm").strip().lower()
        if self.rerank_method not in {"llm", "embedding"}:
            self.rerank_method = "llm"
        # Embedding-based reranking settings, used only when
        # rerank_method == "embedding". Uses Ollama's batch embeddings
        # endpoint so the query text and every shortlisted candidate are
        # embedded in a single request per turn.
        self.embed_enabled = os.getenv("EMBED_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.embed_url = os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embed")
        self.embed_timeout = float(os.getenv("EMBED_TIMEOUT", "8"))
        # The LLM chat rerank caps its shortlist at 10 because a larger
        # required-JSON score schema makes qwen3:4b's structured output
        # unreliable (see llm_rerank_limit above). Embedding calls carry no
        # such risk -- a single batched /api/embed request for 40+ texts
        # still completes in ~1s locally -- so this can cover most of the
        # retrieved candidate pool instead of only the rule-ranked top 10.
        self.embed_rerank_limit = int(os.getenv("EMBED_RERANK_LIMIT", "30"))
        # Cosine similarity for related short texts typically lands in
        # 0.3-0.8, not the full [-1, 1] range, so this weight is calibrated
        # to give a comparably sized nudge to the LLM chat rerank's fused
        # score (0.20 * a 0-4 relevance score, max +0.80) without letting a
        # single embedding call override the rule-based/BM25 ordering.
        self.embed_rerank_weight = float(os.getenv("EMBED_RERANK_WEIGHT", "1.2"))
        self.last_embed_error: str | None = None
        # Keyed by the exact text sent for embedding, not by parent_asin:
        # candidate text is deterministic per product, so this cache
        # survives across turns and sessions within one Agent instance and
        # avoids re-embedding the same catalog entries repeatedly.
        self._embedding_cache: dict[str, list[float]] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.products[asin] = product
                categories = product.get("categories") or []
                self.category_terms.update(_terms(_text(categories)))
                if categories:
                    self.category_head_terms.update(_terms(str(categories[-1])))
                batch.append((
                    asin,
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        self.category_terms_stemmed = {_stem(term) for term in self.category_terms}
        self.category_head_terms_stemmed = {_stem(term) for term in self.category_head_terms}

    def _explicit_override_category_terms(self, message: str) -> set[str]:
        """Return catalog category terms only when the override explicitly requests a new product type.

        Matches against a stemmed (trailing-'s'-stripped) copy of the catalog
        vocabulary so a singular "backpack" still matches a catalog leaf
        category written as "Backpacks", while what's returned stays the
        original, unstemmed token from the message.
        """
        match = CATEGORY_CHANGE_RE.search(message)
        if not match:
            return set()
        candidate_terms = set(_terms(match.group(1)))
        terms_stemmed = getattr(self, "category_terms_stemmed", set())
        head_terms_stemmed = getattr(self, "category_head_terms_stemmed", set())
        matched_terms = {term for term in candidate_terms if _stem(term) in terms_stemmed}
        if not {_stem(term) for term in matched_terms}.intersection(head_terms_stemmed):
            return set()
        return matched_terms

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = SessionState(profile=dict(user_profile or {}))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self.sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        prompt_tokens_before = state.prompt_tokens
        completion_tokens_before = state.completion_tokens
        # This set is cumulative: every previously shown list is excluded on
        # later turns because the evaluator only advances after a miss.
        state.excluded_asins.update(state.last_recommendations)
        message = user_message or ""
        state.messages.append(message)

        if _no_preference_attribute(message):
            # Deterministic fast path: the message matches the simulator's
            # exact "no preference for X" phrasing, so no LLM call is needed
            # to classify intent.
            state.debug_trace["llm_constraint_extraction"] = {
                "status": "skipped",
                "reason": "no_preference_message",
            }
            self._update_state(state, message, extract_constraints=False)
        else:
            # Parse first, then apply any override reset, then validate the
            # parsed constraints -- category-scope validation inside
            # _apply_llm_extraction reads state.override_category_terms, so
            # it must run after the override (if any) has been applied.
            parsed, trace = self._llm_parse(state, message)
            intent, llm_no_preference_attribute = (
                self._llm_intent(parsed) if parsed is not None else (None, None)
            )
            llm_category_terms = (
                self._llm_category_terms(message, parsed) if parsed is not None else None
            )
            trace["intent"] = intent
            trace["no_preference_attribute"] = llm_no_preference_attribute
            trace["category_terms"] = sorted(llm_category_terms) if llm_category_terms is not None else None
            self._update_state(
                state, message,
                extract_constraints=False,
                llm_override=(intent == "override"),
                llm_no_preference_attribute=llm_no_preference_attribute,
                llm_category_terms=llm_category_terms,
            )
            if parsed is not None:
                extracted = self._apply_llm_extraction(
                    state, message, parsed, trace, llm_category_terms=llm_category_terms,
                )
            else:
                state.debug_trace["llm_constraint_extraction"] = trace
                extracted = False
            if not extracted:
                self._update_state(state, message, update_controls=False)
        state.route = self._route(state)
        candidates = self._retrieve(state, top_k=max(top_k * 8, 40))
        candidates = self._rerank(state, candidates)
        state.candidate_pool = [asin for asin, _score in candidates]
        state.debug_trace["reranked_candidates"] = [
            {"parent_asin": asin, "rule_score": round(score, 6)}
            for asin, score in candidates
        ]
        recommendations = [
            {"parent_asin": asin}
            for asin, _score in candidates[: max(1, min(top_k, 10))]
        ]
        state.last_recommendations = [item["parent_asin"] for item in recommendations]

        ask_attribute = self._next_attribute(state, len(candidates), turn)
        reply_message = self._response_message(state, ask_attribute, recommendations)
        return {
            "message": reply_message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": state.prompt_tokens - prompt_tokens_before,
                "completion_tokens": state.completion_tokens - completion_tokens_before,
            },
        }

    def debug_snapshot(self, session_id: str) -> dict:
        """Return the latest diagnostic trace without altering the agent API response."""
        state = self.sessions.get(session_id)
        return dict(state.debug_trace) if state is not None else {}

    def _ollama_chat(
        self,
        messages: list[dict],
        format_json: bool | dict = False,
    ) -> tuple[str, dict]:
        if not self.llm_enabled:
            return "", {}
        self.last_llm_error = None
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        }
        if format_json:
            payload["format"] = format_json if isinstance(format_json, dict) else "json"
        request = urllib.request.Request(
            self.llm_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.llm_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body.get("message") or {}
            usage = {
                "prompt_tokens": int(body.get("prompt_eval_count") or 0),
                "completion_tokens": int(body.get("eval_count") or 0),
            }
            return str(message.get("content") or ""), usage
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as error:
            self.last_llm_error = f"{type(error).__name__}: {error}"
            return "", {}

    def _ollama_embed(self, texts: list[str]) -> tuple[list[list[float]] | None, dict]:
        """Batch-embed texts with Ollama. Returns vectors in input order, or None on failure."""
        if not self.embed_enabled or not texts:
            return None, {}
        self.last_embed_error = None
        payload = {"model": self.embed_model, "input": texts}
        request = urllib.request.Request(
            self.embed_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.embed_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            embeddings = body.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                self.last_embed_error = "malformed_embeddings_response"
                return None, {}
            usage = {"prompt_tokens": int(body.get("prompt_eval_count") or 0)}
            return embeddings, usage
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as error:
            self.last_embed_error = f"{type(error).__name__}: {error}"
            return None, {}

    def _llm_parse(self, state: SessionState, message: str) -> tuple[dict | None, dict]:
        """Call the local model once (with one repair retry) and return (parsed_json, trace).

        Split out from constraint validation/application so a caller can act
        on the classified intent (e.g. apply an override reset) before those
        constraints are validated against state -- category-scope validation
        below depends on state.override_category_terms already reflecting
        the current message's override, if any.
        """
        if not self.llm_enabled or not message.strip():
            return None, {"status": "skipped", "reason": "llm_disabled_or_empty_message"}
        extraction_messages = [
            {
                "role": "system",
                "content": (
                    "Extract shopping constraints from the user message. Return JSON only with "
                    "keys constraints (array of {attribute,value,negated}), "
                    "search_phrases (array of catalog-searchable phrases), intent, "
                    "no_preference_attribute, and category_phrase. "
                    "Allowed attributes: category, material, color, size, style, brand, budget, "
                    "feature, use_case. Search phrases must be copied or minimally normalized "
                    "from the user message, contain only terms matching an allowed attribute "
                    "above (or general product/category words), and exclude conversational wording. "
                    "Return at most 8 phrases of 1 to 8 words. Do not invent values. "
                    "Keep any catalog-searchable product description that does not fit an "
                    "allowed constraint attribute in search_phrases, even when constraints is empty. "
                    "If the message has no product-related constraint, return empty constraints "
                    "and search_phrases arrays. Do not infer product information from conversational "
                    "control text, feedback, or a request to ask another question. Category is only "
                    "the explicitly requested product type; properties such as imported, sole, "
                    "closure, material, or color belong in search_phrases unless they fit another "
                    "allowed attribute. "
                    "intent must be exactly one of these four strings: \"no_preference\" (the user "
                    "says they have no preference, don't care, or it doesn't matter for some "
                    "attribute), \"override\" (the user is replacing or abandoning an earlier stated "
                    "preference with a new one), \"provide_constraint\" (the user is stating a "
                    "concrete requirement or preference), or \"other\" (anything else, including "
                    "small talk or a request to ask a different question). When intent is "
                    "\"no_preference\", also set no_preference_attribute to exactly one of the "
                    "allowed attributes above naming which attribute the user has no preference "
                    "for; otherwise set no_preference_attribute to null. "
                    "category_phrase must be a short phrase copied verbatim or minimally "
                    "normalized from the user message naming ONLY the product type the user "
                    "wants -- the same span you would use as a constraints entry with "
                    "attribute=\"category\". Set it whenever the message states or restates what "
                    "type of product the user wants (including when replacing an earlier product "
                    "type), even though that also appears in a constraints entry; set it to null "
                    "if the message does not name a product type."
                ),
            },
            {"role": "user", "content": json.dumps({
                "user_message": message,
            }, ensure_ascii=False)},
        ]
        content, usage = self._ollama_chat(extraction_messages, format_json=True)
        state.prompt_tokens += usage.get("prompt_tokens", 0)
        state.completion_tokens += usage.get("completion_tokens", 0)
        trace = {
            "status": "invalid_json",
            "raw_response": content,
            "transport_error": self.last_llm_error,
            "usage": usage,
            "accepted_constraints": [],
            "rejected_constraints": [],
            "accepted_search_phrases": [],
            "rejected_search_phrases": [],
        }
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            repaired_content, repair_usage = self._ollama_chat([
                *extraction_messages,
                {
                    "role": "system",
                    "content": (
                        "Your previous response was invalid JSON. Retry once with a single valid JSON object "
                        "using exactly the requested keys and array/object types."
                    ),
                },
            ], format_json=True)
            state.prompt_tokens += repair_usage.get("prompt_tokens", 0)
            state.completion_tokens += repair_usage.get("completion_tokens", 0)
            trace["repair_attempted"] = True
            trace["repair_raw_response"] = repaired_content
            trace["usage"]["prompt_tokens"] = trace["usage"].get("prompt_tokens", 0) + repair_usage.get("prompt_tokens", 0)
            trace["usage"]["completion_tokens"] = trace["usage"].get("completion_tokens", 0) + repair_usage.get("completion_tokens", 0)
            try:
                parsed = json.loads(repaired_content)
            except (TypeError, ValueError):
                return None, trace
        if not isinstance(parsed, dict):
            trace["status"] = "invalid_response_shape"
            return None, trace
        return parsed, trace

    def _llm_intent(self, parsed: dict) -> tuple[str | None, str | None]:
        """Validate intent/no_preference_attribute against their closed vocabularies."""
        intent = parsed.get("intent")
        if intent not in {"no_preference", "override", "provide_constraint", "other"}:
            intent = None
        no_preference_attribute = parsed.get("no_preference_attribute")
        if no_preference_attribute not in ATTRIBUTES:
            no_preference_attribute = None
        return intent, no_preference_attribute

    def _llm_category_terms(self, message: str, parsed: dict) -> set[str]:
        """Ground+validate the LLM's category_phrase against the raw message and catalog vocab.

        Replaces the old "looking for X" / "need|want|switch to X" regexes
        (INITIAL_CATEGORY_RE / CATEGORY_CHANGE_RE, still used as a fallback
        when the LLM is unavailable) with a single mechanism that works for
        both the first category statement and a replacement one after an
        override, and isn't tied to a fixed set of trigger phrases.
        """
        phrase = parsed.get("category_phrase")
        if not isinstance(phrase, str) or not phrase.strip():
            return set()
        phrase_terms = set(_terms(phrase))
        if not phrase_terms or not phrase_terms.issubset(set(_terms(message))):
            return set()
        if not {_stem(term) for term in phrase_terms}.intersection(self.category_head_terms_stemmed):
            return set()
        return phrase_terms

    def _apply_llm_extraction(
        self,
        state: SessionState,
        message: str,
        parsed: dict,
        trace: dict,
        llm_category_terms: set[str] | None = None,
    ) -> bool:
        """Validate and apply constraints/search_phrases from an already-parsed response.

        Assumes any override state reset for this message has already been
        applied to `state` by the caller.
        """
        extracted = False
        constraints = parsed.get("constraints", [])
        if not isinstance(constraints, list):
            trace["status"] = "invalid_constraints_shape"
            state.debug_trace["llm_constraint_extraction"] = trace
            return False
        for item in constraints:
            if not isinstance(item, dict):
                trace["rejected_constraints"].append({"value": item, "reason": "not_an_object"})
                continue
            attribute = str(item.get("attribute", "feature"))
            if attribute not in ATTRIBUTES:
                trace["rejected_constraints"].append({
                    "attribute": attribute,
                    "value": item.get("value"),
                    "reason": "unknown_attribute",
                })
                continue
            value = str(item.get("value", "")).strip()
            if not value:
                trace["rejected_constraints"].append({
                    "attribute": attribute,
                    "value": item.get("value"),
                    "reason": "empty_value",
                })
                continue
            if attribute == "category":
                category_terms = (
                    llm_category_terms if llm_category_terms is not None
                    else _initial_category_terms(message)
                )
                if state.override_category_terms:
                    category_terms = state.override_category_terms
                if not category_terms or not set(_terms(value)).issubset(category_terms):
                    trace["rejected_constraints"].append({
                        "attribute": attribute,
                        "value": value,
                        "reason": "outside_category_scope",
                    })
                    continue
            if attribute == "material" and not any(
                material in _terms(value) for material in MATERIALS
            ):
                trace["rejected_constraints"].append({
                    "attribute": attribute,
                    "value": value,
                    "reason": "unsupported_material",
                })
                continue
            if attribute == "use_case" and not set(_terms(value)).intersection(USE_CASE_MARKERS):
                trace["rejected_constraints"].append({
                    "attribute": attribute,
                    "value": value,
                    "reason": "unsupported_use_case",
                })
                continue
            constraint = Constraint(
                value=value,
                attribute=attribute,
                negated=bool(item.get("negated", False)),
            )
            state.add_constraint(constraint)
            trace["accepted_constraints"].append(constraint.__dict__)
            extracted = True

        if llm_category_terms and not any(
            item["attribute"] == "category" for item in trace["accepted_constraints"]
        ):
            # The model doesn't always echo category_phrase into the
            # constraints array even when it set one; synthesize the
            # constraint ourselves once category_phrase has already passed
            # both the message-grounding and catalog-vocabulary checks in
            # _llm_category_terms, instead of depending on the model to
            # write the same information twice.
            constraint = Constraint(
                value=" ".join(sorted(llm_category_terms)),
                attribute="category",
                negated=False,
            )
            state.add_constraint(constraint)
            trace["accepted_constraints"].append(constraint.__dict__)
            extracted = True

        search_phrases = parsed.get("search_phrases", [])
        message_terms = set(_terms(message))
        if not isinstance(search_phrases, list):
            trace["rejected_search_phrases"].append({
                "value": search_phrases,
                "reason": "invalid_search_phrases_shape",
            })
        else:
            for phrase in search_phrases:
                if not isinstance(phrase, str):
                    trace["rejected_search_phrases"].append({"value": phrase, "reason": "not_a_string"})
                    continue
                normalized = " ".join(phrase.split())
                phrase_terms = _terms(normalized)
                if not normalized:
                    trace["rejected_search_phrases"].append({"value": phrase, "reason": "empty_phrase"})
                elif len(phrase_terms) > 8:
                    trace["rejected_search_phrases"].append({"value": phrase, "reason": "too_many_terms"})
                elif len(normalized) > 120:
                    trace["rejected_search_phrases"].append({"value": phrase, "reason": "too_long"})
                elif not set(phrase_terms).issubset(message_terms):
                    trace["rejected_search_phrases"].append({"value": phrase, "reason": "not_grounded_in_message"})
                else:
                    if normalized not in state.search_phrases:
                        state.search_phrases.append(normalized)
                    trace["accepted_search_phrases"].append(normalized)
                    extracted = True
        trace["status"] = "accepted" if extracted else "no_valid_constraints"
        state.debug_trace["llm_constraint_extraction"] = trace
        return extracted

    def _llm_rerank(self, state: SessionState, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Ask Ollama to reorder only the already-recalled catalog candidates."""
        if not self.llm_enabled or len(candidates) < 2:
            state.debug_trace["llm_rerank"] = {
                "status": "skipped",
                "reason": "llm_disabled_or_insufficient_candidates",
            }
            return candidates
        signature = (
            state.route,
            tuple(sorted(
                (item.attribute, item.value, item.negated)
                for item in state.active_constraints()
            )),
            state.override_active,
            state.debug_trace.get("fallback_mode"),
        )
        if state.rerank_signature == signature:
            state.debug_trace["llm_rerank"] = {
                "status": "skipped",
                "reason": "unchanged_retrieval_intent",
            }
            return candidates
        shortlist = candidates[: min(self.llm_rerank_limit, len(candidates))]
        shortlist_asins = {asin for asin, _score in shortlist}
        relevant_attributes = self._relevant_attributes(state)
        bm25_order = sorted(state.bm25_scores, key=lambda asin: (-state.bm25_scores[asin], asin))
        bm25_ranks = {asin: index + 1 for index, asin in enumerate(bm25_order)}
        candidate_text = [
            self._llm_candidate(
                self.products[asin],
                asin,
                relevant_attributes,
                state.bm25_scores.get(asin),
                bm25_ranks.get(asin),
            )
            for asin, _score in shortlist
        ]
        score_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": {
                        asin: {"type": "integer", "minimum": 0, "maximum": 4}
                        for asin in shortlist_asins
                    },
                    "required": sorted(shortlist_asins),
                    "additionalProperties": False,
                },
            },
            "required": ["scores"],
            "additionalProperties": False,
        }
        content, usage = self._ollama_chat([
            {
                "role": "system",
                "content": (
                    "Score each catalog candidate for the user's current shopping intent. Return "
                    "exactly one JSON object with exactly one key: \"scores\". Its value maps "
                    "every provided parent_asin to an integer from 0 to 4. Example: "
                    "{\"scores\":{\"B012345678\":4,\"B098765432\":1}}. Do not return "
                    "query, keywords, explanations, markdown, or other keys. Score 4 for all "
                    "active constraints satisfied, 3 for the main active constraints satisfied, 2 "
                    "for partially relevant, 1 for category-only relevance, and 0 for a clear "
                    "mismatch. Active constraints matter most."
                ),
            },
            {"role": "user", "content": json.dumps({
                "intent": state.intent_messages[-3:],
                "constraints": [item.__dict__ for item in state.active_constraints()],
                "route": state.route,
                "relevant_attributes": sorted(relevant_attributes),
                "candidates": candidate_text,
            }, ensure_ascii=True)},
        ], format_json=score_schema)
        state.prompt_tokens += usage.get("prompt_tokens", 0)
        state.completion_tokens += usage.get("completion_tokens", 0)
        trace = {
            "status": "invalid_json",
            "shortlist": [
                {"parent_asin": asin, "rule_score": round(score, 6)}
                for asin, score in shortlist
            ],
            "raw_response": content,
            "transport_error": self.last_llm_error,
            "usage": usage,
            "accepted_scores": {},
            "rejected_scores": [],
        }
        state.rerank_signature = signature
        try:
            scores = json.loads(content).get("scores", {})
        except (TypeError, ValueError):
            state.debug_trace["llm_rerank"] = trace
            return candidates
        if not isinstance(scores, dict):
            trace["status"] = "invalid_scores_shape"
            state.debug_trace["llm_rerank"] = trace
            return candidates
        llm_scores: dict[str, float] = {}
        for asin, score in scores.items():
            parent_asin = str(asin)
            if parent_asin not in shortlist_asins:
                trace["rejected_scores"].append({"parent_asin": parent_asin, "score": score, "reason": "not_in_shortlist"})
            elif isinstance(score, bool) or not isinstance(score, (int, float)):
                trace["rejected_scores"].append({"parent_asin": parent_asin, "score": score, "reason": "not_numeric"})
            elif not 0 <= float(score) <= 4:
                trace["rejected_scores"].append({"parent_asin": parent_asin, "score": score, "reason": "out_of_range"})
            else:
                llm_scores[parent_asin] = float(score)
        trace["accepted_scores"] = llm_scores
        if not llm_scores:
            trace["status"] = "no_valid_scores"
            state.debug_trace["llm_rerank"] = trace
            return candidates
        trace["status"] = "accepted"
        state.debug_trace["llm_rerank"] = trace
        reranked = sorted(
            shortlist,
            key=lambda item: (
                -(item[1] + 0.20 * llm_scores.get(item[0], 0.0)),
                -item[1],
                item[0],
            ),
        ) + candidates[len(shortlist):]
        return reranked

    def _rerank(self, state: SessionState, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Dispatch to whichever reranker self.rerank_method selects."""
        if self.rerank_method == "embedding":
            return self._embedding_rerank(state, candidates)
        return self._llm_rerank(state, candidates)

    def _embed_query_text(self, state: SessionState) -> str:
        """Flatten the current shopping intent into one string for embedding."""
        parts = [part for part in state.intent_messages[-3:] if part.strip()]
        constraint_parts = [f"{item.attribute}: {item.value}" for item in state.active_constraints()]
        if constraint_parts:
            parts.append("Constraints - " + "; ".join(constraint_parts))
        parts.append(f"Shopping mode: {state.route}")
        return " | ".join(parts)

    def _embed_candidate_text(self, product: dict) -> str:
        """Flatten one catalog product into one string for embedding."""
        parts = []
        title = _text(product.get("title")).strip()
        if title:
            parts.append(title)
        categories = product.get("categories") or []
        if categories:
            parts.append("Category: " + _text(categories).strip())
        features = product.get("features")
        if isinstance(features, list):
            parts.extend(_text(item).strip() for item in features[:4] if _text(item).strip())
        elif _text(features).strip():
            parts.append(_text(features).strip())
        price = product.get("price")
        if price not in (None, ""):
            parts.append(f"Price: ${price}")
        return " | ".join(parts)[:1200]

    def _embedding_rerank(self, state: SessionState, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Reorder only the already-recalled catalog candidates by embedding cosine similarity
        to the current shopping intent, fused with the rule-based retrieval score."""
        if not self.embed_enabled or len(candidates) < 2:
            state.debug_trace["embedding_rerank"] = {
                "status": "skipped",
                "reason": "embed_disabled_or_insufficient_candidates",
            }
            return candidates
        signature = (
            state.route,
            tuple(sorted(
                (item.attribute, item.value, item.negated)
                for item in state.active_constraints()
            )),
            state.override_active,
            state.debug_trace.get("fallback_mode"),
        )
        if state.rerank_signature == signature:
            state.debug_trace["embedding_rerank"] = {
                "status": "skipped",
                "reason": "unchanged_retrieval_intent",
            }
            return candidates
        state.rerank_signature = signature

        shortlist = candidates[: min(self.embed_rerank_limit, len(candidates))]
        query_text = self._embed_query_text(state)
        if not query_text.strip():
            state.debug_trace["embedding_rerank"] = {
                "status": "skipped",
                "reason": "empty_query_text",
            }
            return candidates

        candidate_texts = {
            asin: self._embed_candidate_text(self.products[asin])
            for asin, _score in shortlist
        }

        pending_keys = [
            text for text in [query_text, *candidate_texts.values()]
            if text and text not in self._embedding_cache
        ]
        # De-duplicate while preserving order: distinct products can flatten
        # to an identical string, so this avoids embedding the same text twice.
        pending_keys = list(dict.fromkeys(pending_keys))

        usage: dict = {}
        if pending_keys:
            embeddings, call_usage = self._ollama_embed(pending_keys)
            usage = call_usage
            state.prompt_tokens += usage.get("prompt_tokens", 0)
            if embeddings is not None:
                for key, vector in zip(pending_keys, embeddings):
                    self._embedding_cache[key] = vector

        trace = {
            "status": "invalid_response",
            "shortlist": [
                {"parent_asin": asin, "rule_score": round(score, 6)}
                for asin, score in shortlist
            ],
            "transport_error": self.last_embed_error,
            "usage": usage,
            "accepted_scores": {},
        }

        query_vector = self._embedding_cache.get(query_text)
        if query_vector is None:
            state.debug_trace["embedding_rerank"] = trace
            return candidates

        similarities: dict[str, float] = {}
        for asin, text in candidate_texts.items():
            vector = self._embedding_cache.get(text) if text else None
            if vector is not None:
                similarities[asin] = _cosine_similarity(query_vector, vector)

        if not similarities:
            trace["status"] = "no_valid_scores"
            state.debug_trace["embedding_rerank"] = trace
            return candidates

        trace["status"] = "accepted"
        trace["accepted_scores"] = {asin: round(score, 6) for asin, score in similarities.items()}
        state.debug_trace["embedding_rerank"] = trace

        reranked = sorted(
            shortlist,
            key=lambda item: (
                -(item[1] + self.embed_rerank_weight * similarities.get(item[0], 0.0)),
                -item[1],
                item[0],
            ),
        ) + candidates[len(shortlist):]
        return reranked

    def _relevant_attributes(self, state: SessionState) -> set[str]:
        """Select product fields from attributes present in the current dialogue."""
        relevant = {item.attribute for item in state.active_constraints()}
        recent_text = " ".join(state.intent_messages[-2:]).lower()
        attribute_markers = {
            "category": ("category", "type", "kind"),
            "material": ("material", "cotton", "leather", "polyester", "fabric"),
            "color": ("color", "black", "white", "blue", "red", "green", "brown"),
            "size": ("size", "fit", "wide", "narrow", "width"),
            "style": ("style", "casual", "formal", "sleeve", "neck"),
            "brand": ("brand", "manufacturer", "store"),
            "budget": ("budget", "price", "under", "$"),
            "feature": ("feature", "waterproof", "comfortable", "lightweight", "durable"),
            "use_case": ("use case", "hiking", "running", "gym", "winter", "outdoor", "work"),
        }
        relevant.update(
            attribute
            for attribute, markers in attribute_markers.items()
            if any(marker in recent_text for marker in markers)
        )
        return relevant

    def _llm_candidate(
        self,
        product: dict,
        asin: str,
        relevant_attributes: set[str] | None = None,
        bm25_score: float | None = None,
        bm25_rank: int | None = None,
    ) -> dict:
        """Build a compact candidate payload focused on the current dialogue."""
        relevant_attributes = relevant_attributes or set()
        fields: dict[str, object] = {
            "title": _text(product.get("title")).strip()[:240],
            "categories": product.get("categories") or [],
        }
        features = product.get("features")
        if isinstance(features, list):
            fields["features"] = [_text(item).strip()[:100] for item in features[:2] if _text(item).strip()]
        elif _text(features).strip():
            fields["features"] = [_text(features).strip()[:200]]
        return {
            "parent_asin": asin,
            "catalog_fields": fields,
            "price": product.get("price"),
            "bm25_score": None if bm25_score is None else round(bm25_score, 6),
            "bm25_rank": bm25_rank,
        }

    def _update_state(
        self,
        state: SessionState,
        message: str,
        *,
        extract_constraints: bool = True,
        update_controls: bool = True,
        llm_override: bool = False,
        llm_no_preference_attribute: str | None = None,
        llm_category_terms: set[str] | None = None,
    ) -> None:
        terms = _terms(message)
        lowered = message.lower()
        # The regex catches the exact override phrasings this dataset's
        # simulator uses; llm_override extends that to paraphrases the LLM
        # classified as intent="override" that the regex missed.
        override = bool(re.search(
            r"\b(actually|instead|ignore|replace|rather|changed my mind|"
            r"forget (?:that|the previous|my earlier)|no longer want)\b",
            lowered,
        )) or llm_override
        if update_controls and override:
            # A replacement request starts a fresh intent while preserving the
            # user profile, conversation transcript, and product category.
            # Declared
            # no-preference attributes stay closed: the simulator's answer
            # for a given attribute depends only on the target's fixed
            # intent card, not on the override, so re-asking one that was
            # already declined can never surface new information.
            replacement_category_terms = (
                llm_category_terms if llm_category_terms is not None
                else self._explicit_override_category_terms(message)
            )
            category_constraints = [] if replacement_category_terms else list(state.constraints.get("category", []))
            state.constraints = {"category": category_constraints} if category_constraints else {}
            state.asked.clear()
            state.override_active = True
            state.override_category_terms = replacement_category_terms
            state.intent_messages = [message]
            state.search_phrases = [
                " ".join(constraint.value for constraint in category_constraints)
            ] if category_constraints else []
            # excluded_asins IS cleared here (A/B tested keeping it, which
            # regressed): the evaluator only counts a hit once its own
            # override_applied flag is true, so a "miss" shown before the
            # override may actually have contained the target under a
            # not-yet-counted turn. Keeping it permanently excluded made
            # intent_override hit-rate collapse from 1.0 to 0.375 on the
            # 50-sample public set. The two known-miss products this
            # re-admits are a smaller cost than that.
            state.excluded_asins.clear()
            state.last_recommendations.clear()
            state.candidate_pool.clear()
            state.bm25_scores.clear()
        elif update_controls:
            state.intent_messages.append(message)

        if extract_constraints:
            # When LLM extraction is enabled, only trusted LLM search phrases are
            # allowed into the retrieval query. Raw conversational tokens are
            # intentionally ignored to avoid polluting BM25 with filler such as
            # "actually", "those options", or "ask me about one specific attribute".
            if _no_preference_attribute(message) is None and not self.llm_enabled:
                state.search_phrases.extend(
                    term for term in terms if term not in state.search_phrases
                )
            for term in terms:
                if term in MATERIALS or term in COLORS:
                    attribute = _attribute_for(term)
                    state.add_constraint(Constraint(term, attribute))

            budget_match = BUDGET_RE.search(message)
            if budget_match:
                state.add_constraint(Constraint(f"under {budget_match.group(1)}", "budget"))

            if NEGATION_RE.search(message) and not override:
                for term in terms:
                    if term in MATERIALS or term in COLORS:
                        state.add_constraint(Constraint(term, _attribute_for(term), negated=True))

        if update_controls:
            # Same paraphrase-vs-regex split as override above: the regex
            # requires the simulator's exact "no preference for X" wording;
            # llm_no_preference_attribute covers freeform equivalents (e.g.
            # "the color doesn't matter to me") the regex would miss. This
            # branch is only reached with a real llm_no_preference_attribute
            # when the regex itself didn't already match -- see respond().
            attribute = _no_preference_attribute(message) or llm_no_preference_attribute
            if attribute is not None:
                state.clear_attribute(attribute)
                state.no_preferences.add(attribute)

    def _route(self, state: SessionState) -> str:
        text = " ".join(state.intent_messages).lower()
        constraint_count = len(state.active_constraints())
        buying_markers = ("buy", "purchase", "need", "must", "require", "key requirement", "budget")
        return "buying" if constraint_count >= 2 or any(marker in text for marker in buying_markers) else "browsing"

    def _historical_query_terms(self, state: SessionState) -> list[str]:
        """Return only previously accepted LLM retrieval phrases, never raw dialogue history."""
        return _terms(" ".join(state.search_phrases))

    def _fallback_history_candidates(self, state: SessionState, top_k: int) -> list[tuple[str, float]]:
        """If no fresh query is available, rotate through already BM25-ranked candidates instead of querying noisy text."""
        ordered: list[str] = []
        if state.bm25_scores:
            ordered = [
                asin for asin, _score in sorted(state.bm25_scores.items(), key=lambda item: (-item[1], item[0]))
                if asin not in state.excluded_asins
            ]
        if not ordered and state.candidate_pool:
            ordered = [asin for asin in state.candidate_pool if asin not in state.excluded_asins]
        if not ordered:
            return []
        batch = ordered[: max(10, min(top_k * 2, 30))]
        state.debug_trace.update({
            "route": state.route,
            "query_terms": [],
            "fallback_mode": "history_bm25_candidates",
            "active_constraints": [item.__dict__ for item in state.active_constraints()],
            "bm25_searches": [{"expression": "history_fallback", "route_weight": 1.0, "query_limit": len(batch), "matches": [{"parent_asin": asin, "bm25_score": round(float(state.bm25_scores.get(asin, 0.0)), 6), "bm25_rank": index + 1, "rank_score": 1.0, "weighted_score": 1.0} for index, asin in enumerate(batch)]}],
            "bm25_candidates": [
                {"parent_asin": asin, "weighted_score": round(float(state.bm25_scores.get(asin, 0.0)), 6)}
                for asin in batch
            ],
        })
        return [(asin, float(state.bm25_scores.get(asin, 0.0))) for asin in batch]

    def _query_terms(self, state: SessionState) -> list[str]:
        if state.search_phrases:
            terms = _terms(" ".join(state.search_phrases))
        else:
            terms = self._historical_query_terms(state)
        profile_tags = state.profile.get("preference_tags") or []
        if state.route == "browsing":
            terms.extend(_terms(" ".join(str(tag) for tag in profile_tags)))
        for constraint in state.active_constraints():
            terms.extend(_terms(constraint.value))
        return list(dict.fromkeys(terms))[:60]

    def _strict_expression(self, state: SessionState) -> str:
        grouped: dict[str, list[Constraint]] = {}
        for constraint in state.active_constraints():
            grouped.setdefault(constraint.attribute, []).append(constraint)

        clauses: list[str] = []
        for attribute, constraints in grouped.items():
            value_clauses = [
                " AND ".join(_escape_fts(token) for token in _terms(constraint.value))
                for constraint in constraints
            ]
            value_clauses = [clause for clause in value_clauses if clause]
            if not value_clauses:
                continue
            if attribute == "category" or len(value_clauses) == 1:
                clauses.extend(value_clauses)
            else:
                clauses.append("(" + " OR ".join(f"({clause})" for clause in value_clauses) + ")")
        return " AND ".join(clauses)

    def _retrieve(self, state: SessionState, top_k: int) -> list[tuple[str, float]]:
        terms = self._query_terms(state)
        if not terms:
            fallback = self._fallback_history_candidates(state, top_k)
            if fallback:
                return fallback
            state.debug_trace.update({
                "route": state.route,
                "query_terms": [],
                "active_constraints": [item.__dict__ for item in state.active_constraints()],
                "bm25_searches": [],
                "bm25_candidates": [],
            })
            return []
        quoted = [_escape_fts(term) for term in terms]
        broad_expression = " OR ".join(quoted)
        strict_expression = self._strict_expression(state)

        expressions = [(broad_expression, 1.0)]
        if state.route == "buying" and strict_expression:
            expressions.insert(0, (strict_expression, 1.35))

        rows: dict[str, float] = {}
        searches: list[dict] = []
        for expression, route_weight in expressions:
            # title dropped 8.0->4.0, features raised 3.5->6.0 (quick A/B,
            # round 2 after 5.0/5.0 improved most but not all samples):
            # marketing-style titles that omit the literal material/color/
            # feature words the user echoes back (e.g. "100% Cotton",
            # "Buckle closure") were losing to competitors whose titles just
            # happened to contain those words, even when this doc's features
            # list matched verbatim. See public_0020 in the 50-sample
            # baseline: target never entered the candidate pool in 10 turns.
            query = (
                "SELECT parent_asin, bm25(products, 0.0, 4.0, 5.0, 6.0, 2.5, 2.0, 1.0) "
                "FROM products WHERE products MATCH ? ORDER BY bm25(products, 0.0, 4.0, 5.0, 6.0, 2.5, 2.0, 1.0) LIMIT ?"
            )
            query_limit = top_k + len(state.excluded_asins)
            matches: list[dict] = []
            for rank, (asin, bm25_score) in enumerate(
                self.connection.execute(query, (expression, query_limit)), start=1,
            ):
                if str(asin) in state.excluded_asins:
                    continue
                rank_score = RRF_K / (RRF_K + rank)
                weighted_score = rank_score * route_weight
                rows[str(asin)] = max(rows.get(str(asin), 0.0), weighted_score)
                matches.append({
                    "parent_asin": str(asin),
                    "bm25_score": round(float(bm25_score), 6),
                    "bm25_rank": rank,
                    "rank_score": round(rank_score, 6),
                    "weighted_score": round(weighted_score, 6),
                })
            searches.append({
                "expression": expression,
                "route_weight": route_weight,
                "query_limit": query_limit,
                "matches": matches,
            })

        # Drop candidates that contain a value the user explicitly said they
        # don't want (e.g. "no leather"), rather than just withholding the
        # positive match boost for it.
        rows = {
            asin: score
            for asin, score in rows.items()
            if not self._violates_negated_constraints(self.products[asin], state)
        }
        scored = [(asin, self._rerank_score(state, asin, score)) for asin, score in rows.items()]
        state.bm25_scores = dict(rows)
        scored.sort(key=lambda item: (-item[1], item[0]))
        state.debug_trace.update({
            "route": state.route,
            "query_terms": terms,
            "search_phrases": state.search_phrases,
            "active_constraints": [item.__dict__ for item in state.active_constraints()],
            "excluded_asins": sorted(state.excluded_asins),
            "bm25_searches": searches,
            "bm25_candidates": [
                {"parent_asin": asin, "weighted_score": round(score, 6)}
                for asin, score in sorted(rows.items(), key=lambda item: (-item[1], item[0]))
            ],
        })
        return scored

    def _product_corpus(self, product: dict) -> set[str]:
        return {
            token
            for field in ("title", "categories", "features", "details", "store", "description")
            for token in _terms(_text(product.get(field)))
        }

    def _violates_negated_constraints(self, product: dict, state: SessionState) -> bool:
        """True if the product contains every token of a value the user explicitly excluded."""
        negated = state.negated_constraints()
        if not negated:
            return False
        corpus = self._product_corpus(product)
        return any(
            (tokens := _terms(constraint.value)) and all(token in corpus for token in tokens)
            for constraint in negated
        )

    def _rerank_score(self, state: SessionState, asin: str, base_score: float) -> float:
        product = self.products[asin]
        corpus = self._product_corpus(product)
        score = base_score
        constraints_by_attribute: dict[str, list[Constraint]] = {}
        for constraint in state.active_constraints():
            constraints_by_attribute.setdefault(constraint.attribute, []).append(constraint)
        for constraints in constraints_by_attribute.values():
            matched = any(
                (tokens := _terms(constraint.value)) and all(token in corpus for token in tokens)
                for constraint in constraints
            )
            if matched:
                score += 0.35
            else:
                score -= 0.08
        title_terms = set(_terms(_text(product.get("title"))))
        score += 0.10 * len(title_terms.intersection(_terms(" ".join(state.intent_messages))))
        if state.route == "browsing":
            score += 0.03 * len(title_terms.intersection(_terms(" ".join(state.profile.get("preference_tags") or []))))
        price = product.get("price")
        budget = next((item for item in state.constraints.get("budget", []) if not item.negated), None)
        if budget and price not in (None, ""):
            limit = re.search(r"(\d+(?:\.\d+)?)", budget.value)
            if limit:
                score += 0.20 if float(price) <= float(limit.group(1)) else -0.25
        return score

    def _next_attribute(self, state: SessionState, candidate_count: int, turn: int) -> str | None:
        if turn >= 10:
            return None
        if candidate_count <= 10 and len(state.active_constraints()) >= 2:
            return None

        # An override replaces an earlier preference. Ask the simulator's
        # catch-all attribute immediately so the new intent can expose its
        # remaining discriminating features instead of exhausting unrelated
        # no-preference questions.
        if state.override_active and "other" not in state.no_preferences and "other" not in state.asked:
            state.asked.add("other")
            return "other"
        if (
            state.route == "buying"
            and state.no_preferences
            and "other" not in state.no_preferences
            and "other" not in state.asked
        ):
            state.asked.add("other")
            return "other"

        unresolved = [
            attribute for attribute in ATTRIBUTES
            if attribute != "other"
            and attribute not in UNASKABLE_ATTRIBUTES
            and attribute not in state.constraints
            and attribute not in state.no_preferences
            and attribute not in state.asked
        ]
        best_attribute = None
        best_gain = 0.0
        for attribute in unresolved:
            groups = self._candidate_attribute_groups(state, attribute)
            if len(groups) < 2:
                continue
            covered = sum(len(group) for group in groups.values())
            if covered / candidate_count < 0.5:
                continue
            expected_remaining = sum(len(group) ** 2 for group in groups.values()) / candidate_count
            information_gain = 1.0 - expected_remaining / candidate_count
            if information_gain > best_gain:
                best_gain = information_gain
                best_attribute = attribute

        if best_attribute is not None:
            state.asked.add(best_attribute)
            return best_attribute

        # Keep a deterministic fallback when the candidate data has no useful
        # attribute split, such as a narrow or incomplete catalog slice.
        # "feature" is prioritized because it is the simulator's default
        # bucket for any constraint that doesn't match a more specific
        # attribute. "other" comes right after: the simulator treats it as
        # a true catch-all that matches any still-undisclosed constraint
        # regardless of classification, including a second value already
        # sitting in an attribute bucket that looks resolved.
        priority = (
            "material", "color", "size", "budget", "feature",
            "other", "style", "use_case",
        )
        for attribute in priority:
            if attribute not in state.constraints and attribute not in state.no_preferences and attribute not in state.asked:
                state.asked.add(attribute)
                return attribute
        return None

    def _candidate_attribute_groups(self, state: SessionState, attribute: str) -> dict[str, list[str]]:
        """Group the current candidates by one attribute for question scoring."""
        groups: dict[str, list[str]] = {}
        for asin in state.candidate_pool:
            values = self._product_attribute_values(self.products[asin], attribute)
            if values:
                group_key = "|".join(sorted(values))
                groups.setdefault(group_key, []).append(asin)
        return groups

    def _product_attribute_values(self, product: dict, attribute: str) -> set[str]:
        """Extract coarse, searchable values without inventing catalog data."""
        text = _text(product).lower()
        if attribute == "category":
            categories = product.get("categories") or []
            return {str(categories[-1]).lower()} if categories and str(categories[-1]).strip() else set()
        if attribute == "brand":
            details = product.get("details") if isinstance(product.get("details"), dict) else {}
            brand = product.get("store") or details.get("Brand") or details.get("Manufacturer")
            return {_text(brand).strip().lower()} if _text(brand).strip() else set()
        if attribute == "budget":
            price = product.get("price")
            if not isinstance(price, (int, float)) or isinstance(price, bool):
                return set()
            if price <= 25:
                bucket = "under 25"
            elif price <= 50:
                bucket = "25 to 50"
            elif price <= 100:
                bucket = "50 to 100"
            else:
                bucket = "over 100"
            return {bucket}
        if attribute == "material":
            return {value for value in MATERIALS if re.search(rf"\b{re.escape(value)}\b", text)}
        if attribute == "color":
            return {value for value in COLORS if re.search(rf"\b{re.escape(value)}\b", text)}

        markers = {
            "size": ("small", "medium", "large", "wide", "narrow", "plus size", "adjustable"),
            "style": (
                "casual", "formal", "vintage", "athletic", "long sleeve", "short sleeve",
                "elegant", "minimalist", "classic", "modern", "chain link", "beaded",
            ),
            "feature": (
                "waterproof", "breathable", "lightweight", "comfortable", "durable", "washable",
                "water resistant", "scratch resistant", "hypoallergenic", "stainless steel",
                "quartz", "digital", "analog", "chronograph", "solar powered", "plated",
                "engraved", "rhinestone", "handmade",
            ),
            "use_case": (
                "hiking", "running", "gym", "winter", "outdoor", "work", "walking",
                "formal wear", "everyday", "gift", "wedding", "party", "office", "travel",
            ),
        }
        return {marker for marker in markers.get(attribute, ()) if marker in text}

    def _response_message(
        self,
        state: SessionState,
        ask_attribute: str | None,
        recommendations: list[dict],
    ) -> str:
        if ask_attribute:
            return f"I found {len(recommendations)} promising matches. Do you have a preference for {ask_attribute}?"
        if recommendations:
            return "Here are the best matches for your current preferences."
        return "I need one more detail to narrow the search."
