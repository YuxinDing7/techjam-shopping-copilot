from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

from evaluator.local_evaluator import catalog_index, evaluate, metric_summary, normalize_recommendations
from starter.agent import Agent, Constraint, SessionState


class EchoTargetAgent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        asin = "A"
        if "B" in user_message:
            asin = "B"
        return {"message": "ok", "ask_attribute": None, "recommendations": [{"parent_asin": asin}]}


class EvaluatorTest(unittest.TestCase):
    def test_normalization_preserves_first_valid_unique_order(self) -> None:
        payload = [
            {"parent_asin": "A"}, {"parent_asin": "bad"}, {"parent_asin": "A"},
            "B", {"parent_asin": "C"},
        ]
        self.assertEqual(normalize_recommendations(payload, {"A", "B", "C"}), ["A", "B", "C"])

    def test_metric_summary_assigns_turn_11_to_miss(self) -> None:
        sessions = [
            {"hit": True, "reciprocal_rank": .5, "first_hit_turn": 2},
            {"hit": False, "reciprocal_rank": 0.0, "first_hit_turn": None},
        ]
        self.assertEqual(metric_summary(sessions), {
            "sample_count": 2,
            "hit_rate_at_10": .5,
            "mrr": .25,
            "mttc": 6.5,
        })

    def test_llm_enabled_keeps_only_filtered_search_phrases(self) -> None:
        agent = Agent.__new__(Agent)
        agent.llm_enabled = True
        state = SessionState(profile={})

        agent._update_state(
            state,
            "Actually, ignore my earlier preference. What I need is: leather.",
            extract_constraints=True,
            update_controls=True,
        )

        self.assertEqual(state.search_phrases, [])
        self.assertEqual(state.active_constraints()[0].value, "leather")

    def test_override_preserves_category_and_resets_retrieval_state(self) -> None:
        agent = Agent.__new__(Agent)
        agent.llm_enabled = True
        state = SessionState(profile={})
        state.add_constraint(Constraint("Accessories", "category"))
        state.add_constraint(Constraint("Belts", "category"))
        state.add_constraint(Constraint("Buckle closure", "feature"))
        state.search_phrases = ["Accessories Belts", "Buckle closure"]
        state.excluded_asins = {"old-result"}
        state.last_recommendations = ["old-result"]
        state.candidate_pool = ["old-result"]
        state.bm25_scores = {"old-result": 1.0}

        agent._update_state(
            state,
            "Actually, ignore my earlier preference. What I need is: leather.",
            extract_constraints=False,
            update_controls=True,
        )

        self.assertEqual([item.value for item in state.active_constraints()], ["Accessories", "Belts"])
        self.assertEqual(state.search_phrases, ["Accessories Belts"])
        self.assertFalse(state.excluded_asins)
        self.assertFalse(state.last_recommendations)
        self.assertFalse(state.candidate_pool)
        self.assertFalse(state.bm25_scores)

    def test_override_prioritizes_catch_all_clarification(self) -> None:
        agent = Agent.__new__(Agent)
        state = SessionState(profile={}, override_active=True)

        self.assertEqual(agent._next_attribute(state, candidate_count=20, turn=3), "other")
        self.assertIn("other", state.asked)

    def test_buying_prioritizes_catch_all_after_no_preference(self) -> None:
        agent = Agent.__new__(Agent)
        state = SessionState(profile={}, route="buying", no_preferences={"style"})

        self.assertEqual(agent._next_attribute(state, candidate_count=20, turn=3), "other")
        self.assertIn("other", state.asked)

    def test_llm_rerank_skips_when_constraints_are_unchanged(self) -> None:
        agent = Agent.__new__(Agent)
        agent.llm_enabled = True
        state = SessionState(profile={}, route="buying")
        state.add_constraint(Constraint("leather", "material"))
        state.rerank_signature = (
            "buying", (("material", "leather", False),), False, None,
        )

        self.assertEqual(agent._llm_rerank(state, [("A", 1.0), ("B", 0.9)]), [("A", 1.0), ("B", 0.9)])
        self.assertEqual(state.debug_trace["llm_rerank"]["reason"], "unchanged_retrieval_intent")

    def test_attribute_only_override_keeps_category_scope(self) -> None:
        agent = Agent.__new__(Agent)
        agent.llm_enabled = True
        agent.category_terms = {"accessories", "belts", "running", "shoes", "leather"}
        agent.category_head_terms = {"belts", "shoes"}
        state = SessionState(profile={})
        state.add_constraint(Constraint("Accessories", "category"))
        state.add_constraint(Constraint("Belts", "category"))

        agent._update_state(
            state,
            "Actually, ignore my earlier preference. What I need is: leather.",
            extract_constraints=False,
            update_controls=True,
        )

        self.assertEqual([item.value for item in state.active_constraints()], ["Accessories", "Belts"])
        self.assertFalse(state.override_category_terms)

    def test_strict_expression_uses_or_for_alternative_materials(self) -> None:
        agent = Agent.__new__(Agent)
        state = SessionState(profile={})
        state.add_constraint(Constraint("cotton", "material"))
        state.add_constraint(Constraint("90% cotton 10% polyester", "material"))
        state.add_constraint(Constraint("grey", "color"))

        self.assertEqual(
            agent._strict_expression(state),
            '(("cotton") OR ("90" AND "cotton" AND "10" AND "polyester")) AND "grey"',
        )

    def test_empty_llm_query_does_not_reuse_raw_history(self) -> None:
        agent = Agent.__new__(Agent)
        state = SessionState(profile={}, intent_messages=["Those options are not quite right yet."])

        self.assertEqual(agent._historical_query_terms(state), [])

    def test_evaluate_derives_hidden_fields_when_public_set_omits_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_rows = [
                {
                    "parent_asin": "A",
                    "title": "Blue running shoe",
                    "features": ["cotton"],
                    "details": {"department": "womens"},
                    "description": ["walking shoe"],
                    "categories": ["Clothing", "Shoes"],
                    "store": "Example",
                    "average_rating": 4.2,
                    "rating_number": 10,
                    "price": 49.0,
                },
                {
                    "parent_asin": "B",
                    "title": "Black winter boot",
                    "features": ["leather"],
                    "details": {"department": "womens"},
                    "description": ["winter boot"],
                    "categories": ["Clothing", "Boots"],
                    "store": "Example",
                    "average_rating": 4.4,
                    "rating_number": 12,
                    "price": 89.0,
                },
            ]
            catalog_path.write_text("".join(json.dumps(row) + "\n" for row in catalog_rows), encoding="utf-8")
            catalog_ids, categories, products = catalog_index(catalog_path)
            samples = [{
                "sample_id": "public_v2_0001",
                "scenario_type": "buying",
                "user_profile": {"summary": "x"},
                "ground_truth": {"parent_asin": "A"},
            }]
            result = evaluate(EchoTargetAgent(), samples, catalog_ids, categories, products)
            self.assertEqual(result["hit_rate_at_10"], 1.0)


if __name__ == "__main__":
    unittest.main()
