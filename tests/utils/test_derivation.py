"""Unit tests for derivation records and assertion basis (conclusion ledger)."""

import asyncio

import pytest

from src.utils.derivation import (
    DerivationRecord,
    classify_assertion_basis,
    classify_with_spans,
    current_derivation,
    derivation_scope,
    find_declared_spans,
    model_used_scope,
    normalize_for_match,
    prompt_fingerprint,
)


class TestAssertionBasis:
    def test_explicit_verbatim_in_cited_message(self):
        basis = classify_assertion_basis(
            level="explicit",
            content="I moved to the infra team",
            cited_message_contents=["I moved to the infra team"],
        )
        assert basis == "declared_verbatim"

    def test_changed_case_or_omitted_context_is_reading(self):
        basis = classify_assertion_basis(
            level="explicit",
            content="  I Moved   to the infra team. ",
            cited_message_contents=["so i moved to the INFRA team, finally"],
        )
        assert basis == "llm_reading"

    def test_explicit_paraphrase_is_llm_reading(self):
        basis = classify_assertion_basis(
            level="explicit",
            content="User works on the infrastructure team",
            cited_message_contents=["Big news: I moved to the infra team last week!"],
        )
        assert basis == "llm_reading"

    def test_explicit_without_citations_is_llm_reading(self):
        assert (
            classify_assertion_basis(level="explicit", content="User likes tea")
            == "llm_reading"
        )

    def test_short_content_never_counts_as_verbatim(self):
        # "yes" appears in the message but carries no assertion on its own.
        basis = classify_assertion_basis(
            level="explicit",
            content="yes",
            cited_message_contents=["yes, I am vegetarian"],
            verbatim_min_chars=12,
        )
        assert basis == "llm_reading"

    @pytest.mark.parametrize("level", ["deductive", "inductive", "contradiction"])
    def test_derived_levels_are_inference(self, level: str):
        basis = classify_assertion_basis(
            level=level,
            content="I moved to the infra team",
            cited_message_contents=["I moved to the infra team"],
        )
        assert basis == "llm_inference"

    def test_api_producer_wins(self):
        basis = classify_assertion_basis(
            level="explicit",
            content="I moved to the infra team",
            cited_message_contents=["I moved to the infra team"],
            producer="api",
        )
        assert basis == "api_supplied"


class TestPromptFingerprint:
    def test_stable_and_short(self):
        a = prompt_fingerprint("template {messages}", None)
        b = prompt_fingerprint("template {messages}", None)
        assert a == b
        assert len(a) == 16

    def test_changes_with_any_part(self):
        base = prompt_fingerprint("template", "instructions")
        assert prompt_fingerprint("template2", "instructions") != base
        assert prompt_fingerprint("template", "other") != base

    def test_part_boundaries_matter(self):
        assert prompt_fingerprint("ab", "c") != prompt_fingerprint("a", "bc")

    def test_none_equals_empty(self):
        assert prompt_fingerprint("t", None) == prompt_fingerprint("t", "")


class TestDerivationScope:
    def test_binds_and_resets(self):
        assert current_derivation() is None
        record = DerivationRecord(producer="deriver", model="m", run_id="r")
        with derivation_scope(record):
            assert current_derivation() is record
        assert current_derivation() is None

    def test_nested_scopes_restore_outer(self):
        outer = DerivationRecord(producer="deriver")
        inner = DerivationRecord(producer="api")
        with derivation_scope(outer):
            with derivation_scope(inner):
                assert current_derivation() is inner
            assert current_derivation() is outer

    def test_propagates_into_tasks(self):
        record = DerivationRecord(producer="dreamer.deduction")

        async def read() -> DerivationRecord | None:
            return current_derivation()

        async def main() -> DerivationRecord | None:
            with derivation_scope(record):
                return await asyncio.create_task(read())

        assert asyncio.run(main()) is record

    def test_metadata_drops_unset_fields(self):
        record = DerivationRecord(producer="api")
        assert record.as_metadata() == {"producer": "api"}


def test_normalize_for_match():
    assert normalize_for_match('  "Hello,   World!" ') == "hello,   world".replace(
        "   ", " "
    )


class TestDeclaredSpans:
    def test_verbatim_span_has_original_offsets(self):
        message = "  I moved to the infra team  "
        kind, spans = find_declared_spans(
            "I moved to the infra team", [("m1", message)]
        )
        assert kind == "verbatim"
        assert len(spans) == 1
        span = spans[0]
        assert span.message_id == "m1"
        assert message[span.start : span.end] == span.text
        assert span.text == "I moved to the infra team"

    def test_verbatim_in_several_messages_gives_one_span_each(self):
        kind, spans = find_declared_spans(
            "I moved to the infra team",
            [
                ("m1", "I moved to the infra team"),
                ("m2", "I moved to the infra team"),
            ],
        )
        assert kind == "verbatim"
        assert [s.message_id for s in spans] == ["m1", "m2"]

    def test_near_verbatim_is_only_evidence(self):
        message = "Hello there. I have moved to the infra team. See you soon."
        kind, spans = find_declared_spans(
            "I moved to the infra team",
            [("m1", message)],
            near_verbatim_min_ratio=0.9,
        )
        assert kind == "none"
        assert spans[0].match == "closest"
        assert spans[0].text == message
        assert message[spans[0].start : spans[0].end] == spans[0].text

    def test_paraphrase_keeps_closest_sentence_as_evidence(self):
        kind, spans = find_declared_spans(
            "User works on infrastructure",
            [("m1", "Weather is nice. I moved to the infra team.")],
        )
        assert kind == "none"
        assert len(spans) == 1
        assert spans[0].match == "closest"
        assert spans[0].text == "Weather is nice. I moved to the infra team."

    def test_no_messages_no_spans(self):
        assert find_declared_spans("anything at all here", []) == ("none", [])

    def test_span_dict_is_json_ready(self):
        _, spans = find_declared_spans(
            "I moved to the infra team", [("m1", "I moved to the infra team")]
        )
        data = spans[0].as_dict()
        assert set(data) == {"message_id", "start", "end", "sha256", "match", "score"}
        assert data["score"] == 1.0


class TestClassifyWithSpans:
    def test_near_verbatim_basis(self):
        basis, spans = classify_with_spans(
            level="explicit",
            content="I moved to the infra team",
            cited_messages=[("m1", "Yes. I have moved to the infra team.")],
        )
        assert basis == "llm_reading"
        assert spans and spans[0].match == "closest"

    def test_api_explicit_keeps_spans(self):
        basis, spans = classify_with_spans(
            level="explicit",
            content="I moved to the infra team",
            cited_messages=[("m1", "I moved to the infra team")],
            producer="api",
        )
        assert basis == "api_supplied"
        assert spans[0].match == "verbatim"

    def test_api_derived_is_api_supplied(self):
        basis, spans = classify_with_spans(
            level="deductive", content="x is y because z", producer="api"
        )
        assert basis == "api_supplied"
        assert spans == []


class TestModelUsedScope:
    def test_sets_and_restores_model_used(self):
        record = DerivationRecord(producer="dreamer.deduction", model="primary")
        with derivation_scope(record):
            with model_used_scope("fallback"):
                current = current_derivation()
                assert current is not None
                assert current.model_used == "fallback"
                assert current.model == "primary"
            assert current_derivation() is record

    def test_noop_without_record_or_model(self):
        with model_used_scope("m"):
            assert current_derivation() is None
        record = DerivationRecord(producer="deriver")
        with derivation_scope(record), model_used_scope(None):
            assert current_derivation() is record


@pytest.mark.parametrize(
    "message,claim",
    [
        ("I never said I live in Pelotas.", "I live in Pelotas."),
        ("I earn 10000 dollars per year.", "I earn 100000 dollars per year."),
        ("I used to live in Pelotas, not anymore.", "I live in Pelotas"),
        ("Ela disse que moro em Pelotas. Isso é falso.", "moro em Pelotas"),
        ("I do not like tea.", "I like tea."),
        ("Do I live in Pelotas?", "I live in Pelotas"),
    ],
)
def test_fuzzy_or_partial_match_never_becomes_declaration(message, claim):
    basis, spans = classify_with_spans(
        level="explicit", content=claim, cited_messages=[("m", message)]
    )
    assert basis == "llm_reading"
    assert spans and spans[0].match == "closest"
    assert "text" not in spans[0].as_dict()


def test_provider_scope_isolation_and_tool_ids():
    record = DerivationRecord(producer="dreamer.deduction", model="primary")
    with derivation_scope(record):
        with model_used_scope(
            "actual", selected_model="fallback", reported_model="actual", fallback=True
        ):
            with model_used_scope(None, tool_call_id="tool-1"):
                current = current_derivation()
                assert current is not None
                assert current.model_used == "actual"
                assert current.selected_model == "fallback"
                assert current.tool_call_id == "tool-1"
            assert current_derivation().tool_call_id is None
        assert current_derivation() == record
