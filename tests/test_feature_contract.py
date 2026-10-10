"""Synthetic contract fixtures only, NOT extracted research features.

No models, gradients, optimizers, training, downloads or GPU jobs are used.
Layer names, numbers and history policies below are arbitrary test choices,
not approved definitions or evidence of forecasting accuracy.
"""
import copy
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from src.third_eye.feature_contract import (
    ContractError,
    HistoryContext,
    HistoryEntry,
    ObservationAvailability,
    UnresolvedContractError,
    load_feature_contract,
)


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "feature_schema.yaml"
SYNTHETIC_LAYER_ORDER = ("synthetic.block.0.q_proj", "synthetic.block.1.q_proj")
SYNTHETIC_DIMENSIONS = {"num_lora_layers": 2, "history_len": 3}
HISTORY_FIELDS = (
    "prev_accepted_value", "prev_target_change", "prev_ood_change", "prev_retention_change",
)


def synthetic_inputs(contract, *, ranking_state=0, rounds=(), padding_side="right", order="oldest_first"):
    values = {
        group.name: {
            field.name: np.zeros(contract.resolved_shape(field), dtype=field.dtype)
            for field in group.fields
        }
        for group in contract.input_groups
    }
    history = values["recursive_history"]
    history["current_gen_index"][0] = ranking_state
    entries = [None] * contract.history_len
    start = contract.history_len - len(rounds) if padding_side == "left" else 0
    for slot, update_round in enumerate(rounds, start=start):
        history["history_mask"][slot] = 1
        entries[slot] = HistoryEntry(update_round, {
            name: ObservationAvailability(
                "forecast" if name == "prev_accepted_value" else "applied_h1",
                update_round if name == "prev_accepted_value" else update_round + 1,
            )
            for name in HISTORY_FIELDS
        })
    return values, HistoryContext(ranking_state, padding_side, order, tuple(entries))


class FeatureContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_feature_contract(
            SCHEMA_PATH, dimensions=SYNTHETIC_DIMENSIONS, layer_order=SYNTHETIC_LAYER_ORDER,
        )
        self.values, self.context = synthetic_inputs(self.contract)

    def validate(self, values=None, context=None, contract=None, layer_order=SYNTHETIC_LAYER_ORDER):
        return (self.contract if contract is None else contract).validate_inputs(
            self.values if values is None else values,
            layer_order=layer_order,
            history_context=self.context if context is None else context,
        )

    def schema_variant(self, mutate):
        document = yaml.safe_load(SCHEMA_PATH.read_text())
        mutate(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_schema.yaml"
            path.write_text(yaml.safe_dump(document, sort_keys=False))
            return load_feature_contract(
                path, dimensions=SYNTHETIC_DIMENSIONS, layer_order=SYNTHETIC_LAYER_ORDER,
            )

    def raw_schema(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_schema.yaml"
            path.write_text(text)
            return load_feature_contract(path)

    def test_source_copy_and_declared_order(self):
        self.assertEqual(self.contract.source_sha256,
                         "5092f748af6f854bf52a19b57d943259dbc844555c97e0236128373520602c68")
        self.assertEqual(self.contract.source_sha256, hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest())
        original = yaml.safe_load(SCHEMA_PATH.read_text())
        expected = [
            (name, [field["name"] for field in group["fields"]])
            for name, group in original.items() if isinstance(group, dict) and "fields" in group
        ]
        self.assertEqual([(group.name, [field.name for field in group.fields])
                          for group in self.contract.groups], expected)
        self.assertEqual(sum(len(group.fields) for group in self.contract.groups), 33)
        self.assertIsNone(self.contract.frozen_on)

    def test_explicit_dimension_summary(self):
        self.assertEqual(self.contract.dimension_summary(), {
            "static_state_feature_dim": 5,
            "static_candidate_feature_dim": 18,
            "history_feature_dim": 15,
        })
        self.assertEqual(self.contract.probe_steps, 15)

    def test_unresolved_dimensions_are_reported_not_guessed(self):
        contract = load_feature_contract(SCHEMA_PATH)
        self.assertIsNone(contract.dimension_summary()["static_candidate_feature_dim"])
        self.assertIsNone(contract.dimension_summary()["history_feature_dim"])
        issues = " ".join(contract.unresolved_definitions())
        for term in ("num_lora_layers", "history_len", "layer_order", "confidence", "duplicate", "lhv2", "direction"):
            self.assertIn(term, issues)
        with self.assertRaisesRegex(UnresolvedContractError, "layer_order"):
            self.validate(contract=contract)

    def test_layer_order_does_not_infer_layer_count(self):
        contract = load_feature_contract(SCHEMA_PATH, dimensions={"history_len": 3},
                                         layer_order=SYNTHETIC_LAYER_ORDER)
        with self.assertRaisesRegex(UnresolvedContractError, "num_lora_layers"):
            self.validate(contract=contract)

    def test_history_length_must_be_explicit_even_when_declared(self):
        contract = load_feature_contract(SCHEMA_PATH, dimensions={"num_lora_layers": 2},
                                         layer_order=SYNTHETIC_LAYER_ORDER)
        with self.assertRaisesRegex(UnresolvedContractError, "history_len"):
            self.validate(contract=contract)

    def test_layer_count_does_not_infer_layer_order(self):
        contract = load_feature_contract(SCHEMA_PATH, dimensions=SYNTHETIC_DIMENSIONS)
        with self.assertRaisesRegex(UnresolvedContractError, "layer_order"):
            self.validate(contract=contract)

    def test_invalid_dimension_bindings(self):
        for dimensions in ({"num_lora_layers": True}, {"num_lora_layers": 0},
                           {"num_lora_layers": -1}, {"num_lora_layers": 2.0},
                           {"num_lora_layers": "2"}, {"history_len": 2}, {"inferred": 2}):
            with self.subTest(dimensions=dimensions), self.assertRaises(ContractError):
                load_feature_contract(SCHEMA_PATH, dimensions=dimensions)

    def test_invalid_layer_manifest(self):
        for order in ((), ("a",), ("a", "a"), ("a", ""), ("a", 2), "ab"):
            with self.subTest(order=order), self.assertRaises(ContractError):
                load_feature_contract(SCHEMA_PATH, dimensions=SYNTHETIC_DIMENSIONS, layer_order=order)

    def test_dimensions_snapshot_does_not_follow_caller_mutation(self):
        dimensions = dict(SYNTHETIC_DIMENSIONS)
        contract = load_feature_contract(SCHEMA_PATH, dimensions=dimensions, layer_order=SYNTHETIC_LAYER_ORDER)
        dimensions["num_lora_layers"] = 100
        self.assertEqual(contract.dimension_summary()["static_candidate_feature_dim"], 18)
        with self.assertRaises(TypeError):
            contract.dimensions["num_lora_layers"] = 100

    def test_declared_field_order_changes_fingerprint(self):
        changed = self.schema_variant(lambda data: data["current_state_performance"]["fields"].reverse())
        self.assertEqual(changed.groups[0].fields[0].name, "gen_confidence_std")
        self.assertNotEqual(changed.schema_fingerprint, self.contract.schema_fingerprint)
        self.assertNotEqual(changed.fingerprint, self.contract.fingerprint)

    def test_layout_fingerprint_includes_explicit_layer_order(self):
        changed = load_feature_contract(SCHEMA_PATH, dimensions=SYNTHETIC_DIMENSIONS,
                                        layer_order=tuple(reversed(SYNTHETIC_LAYER_ORDER)))
        self.assertEqual(changed.schema_fingerprint, self.contract.schema_fingerprint)
        self.assertNotEqual(changed.fingerprint, self.contract.fingerprint)
        with self.assertRaisesRegex(ContractError, "manifest"):
            self.validate(layer_order=tuple(reversed(SYNTHETIC_LAYER_ORDER)))

    def test_missing_unknown_or_unsupported_schema_entries(self):
        mutations = (
            lambda data: data.pop("labels"),
            lambda data: data.update(schema_version="future"),
            lambda data: data.update(unrecognized=True),
            lambda data: data["candidate_data_stats"].update(scope="per_state"),
            lambda data: data["candidate_data_stats"]["fields"].pop(),
            lambda data: data["candidate_data_stats"]["fields"].append(
                copy.deepcopy(data["candidate_data_stats"]["fields"][0])),
            lambda data: data["candidate_data_stats"]["fields"].append(
                copy.deepcopy(data["labels"]["fields"][0])),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ContractError):
                self.schema_variant(mutation)

    def test_schema_dtypes_and_shapes_are_strict(self):
        for replacement in ({"dtype": "float64"}, {"shape": [2]}, {"shape": [True]},
                            {"shape": [1.0]}, {"shape": ["guessed_dimension"]},
                            {"shape": []}, {"shape": "[1]"}):
            with self.subTest(replacement=replacement), self.assertRaises(ContractError):
                self.schema_variant(lambda data: data["candidate_data_stats"]["fields"][0].update(replacement))

    def test_unsafe_yaml_duplicate_keys_and_merge_keys_are_rejected(self):
        texts = (
            "!!python/object/apply:os.system ['echo UNSAFE']",
            SCHEMA_PATH.read_text() + "\nschema_version: 0.1-pilot\n",
            "<<: {schema_version: 0.1-pilot}\n" + SCHEMA_PATH.read_text(),
            "[not, a, schema]",
            "1: non-string-key",
        )
        with patch("os.system") as execute:
            for text in texts:
                with self.subTest(text=text[:40]), self.assertRaises(ContractError):
                    self.raw_schema(text)
            execute.assert_not_called()

    def test_derived_strings_are_not_executed(self):
        with patch("os.system") as execute:
            changed = self.schema_variant(lambda data: data["derived"].update(
                static_candidate_feature_dim="__import__('os').system('echo UNSAFE')"))
            self.assertEqual(changed.dimension_summary()["static_candidate_feature_dim"], 18)
            execute.assert_not_called()

    def test_frozen_date_is_metadata_not_certification(self):
        changed = self.schema_variant(lambda data: data.update(frozen_on="2026-10-10"))
        self.assertEqual(changed.frozen_on, "2026-10-10")
        self.assertIn("approval", " ".join(changed.unresolved_definitions()))
        with self.assertRaises(ContractError):
            self.schema_variant(lambda data: data.update(frozen_on=True))

    def test_valid_synthetic_inputs_remain_structural_only_and_unmodified(self):
        before = copy.deepcopy(self.values)
        result = self.validate()
        self.assertTrue(result.structural_only)
        self.assertIn("lhv2", " ".join(result.unresolved_definitions))
        for group, fields in before.items():
            for name, value in fields.items():
                np.testing.assert_array_equal(value, self.values[group][name])

    def test_missing_and_unknown_input_fields(self):
        mutations = (
            lambda values: values.pop("candidate_data_stats"),
            lambda values: values["candidate_data_stats"].pop("pre_update_nll"),
            lambda values: values.update(metadata={}),
            lambda values: values["candidate_data_stats"].update(unrecognized=np.zeros(1, np.float32)),
        )
        for index, mutation in enumerate(mutations):
            values = copy.deepcopy(self.values)
            mutation(values)
            with self.subTest(index=index), self.assertRaises(ContractError):
                self.validate(values=values)

    def test_input_dtypes_and_containers_are_not_coerced(self):
        for value in (np.zeros(1, np.float64), np.zeros(1, np.int32), np.zeros(1, bool),
                      np.array(["0"]), np.array([0], dtype=object), [0.0], 0.0,
                      np.ma.array([0.0], dtype=np.float32)):
            self.values["candidate_data_stats"]["pre_update_nll"] = value
            with self.subTest(type=type(value)), self.assertRaises(ContractError):
                self.validate()
        self.values, self.context = synthetic_inputs(self.contract)
        self.values["recursive_history"]["current_gen_index"] = np.zeros(1, np.int64)
        with self.assertRaisesRegex(ContractError, "int32"):
            self.validate()

    def test_malformed_scalar_layer_and_history_shapes(self):
        for group, name, shape in (
            ("candidate_data_stats", "pre_update_nll", ()),
            ("candidate_data_stats", "pre_update_nll", (1, 1)),
            ("gradient_interference_stats", "layerwise_grad_norms", (3,)),
            ("recursive_history", "prev_target_change", (3, 1)),
            ("recursive_history", "history_mask", (4,)),
        ):
            values = copy.deepcopy(self.values)
            values[group][name] = np.zeros(shape, dtype=np.float32)
            with self.subTest(name=name, shape=shape), self.assertRaisesRegex(ContractError, "shape"):
                self.validate(values=values)

    def test_nonfinite_inputs_including_padding_are_rejected(self):
        for group, name in (("candidate_data_stats", "pre_update_nll"),
                            ("recursive_history", "prev_accepted_value"),
                            ("recursive_history", "history_mask")):
            for bad in (np.nan, np.inf, -np.inf):
                values = copy.deepcopy(self.values)
                values[group][name][0] = bad
                with self.subTest(name=name, bad=bad), self.assertRaisesRegex(ContractError, "finite"):
                    self.validate(values=values)

    def test_label_inclusion_is_rejected_at_top_level_and_inside_each_group(self):
        for field in self.contract.label_group.fields:
            values = copy.deepcopy(self.values)
            values[field.name] = np.zeros(1, np.float32)
            with self.subTest(name=field.name), self.assertRaisesRegex(ContractError, "Label"):
                self.validate(values=values)
            for group in self.contract.input_groups:
                values = copy.deepcopy(self.values)
                values[group.name][field.name] = np.zeros(1, np.float32)
                with self.subTest(name=field.name, group=group.name), self.assertRaisesRegex(ContractError, "Label"):
                    self.validate(values=values)
        self.values["labels"] = {}
        with self.assertRaisesRegex(ContractError, "Label"):
            self.validate()

    def test_labels_are_validated_only_through_separate_api(self):
        labels = {"labels": {field.name: np.zeros(1, np.float32) for field in self.contract.label_group.fields}}
        self.assertTrue(self.contract.validate_labels(labels).structural_only)
        for bad in (np.array([np.nan], np.float32), np.zeros(2, np.float32), np.zeros(1, np.float64)):
            changed = copy.deepcopy(labels)
            changed["labels"]["lhv2"] = bad
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                self.contract.validate_labels(changed)
        labels["candidate_data_stats"] = {}
        with self.assertRaises(ContractError):
            self.contract.validate_labels(labels)

    def test_label_mutation_cannot_change_input_validation(self):
        labels = {"labels": {field.name: np.ones(1, np.float32) for field in self.contract.label_group.fields}}
        before = self.validate()
        labels["labels"]["lhv2"][0] = 999
        self.assertEqual(self.validate(), before)

    def test_empty_and_real_zero_histories_are_distinct_and_valid(self):
        self.assertTrue(self.validate().structural_only)
        values, context = synthetic_inputs(self.contract, ranking_state=1, rounds=(0,))
        self.assertEqual(values["recursive_history"]["history_mask"].tolist(), [1, 0, 0])
        for name in HISTORY_FIELDS:
            self.assertFalse(values["recursive_history"][name].any())
        self.assertTrue(self.validate(values, context).structural_only)

    def test_explicit_left_right_padding_and_oldest_newest_order(self):
        for count in range(4):
            for padding_side in ("left", "right"):
                for order in ("oldest_first", "newest_first"):
                    rounds = tuple(range(count))
                    if order == "newest_first":
                        rounds = tuple(reversed(rounds))
                    values, context = synthetic_inputs(self.contract, ranking_state=count, rounds=rounds,
                                                       padding_side=padding_side, order=order)
                    with self.subTest(count=count, side=padding_side, order=order):
                        self.assertTrue(self.validate(values, context).structural_only)

    def test_mask_must_be_binary_contiguous_and_aligned(self):
        for mask in ([0.5, 0, 0], [-1, 0, 0], [2, 0, 0], [0, 1, 0]):
            self.values["recursive_history"]["history_mask"] = np.array(mask, np.float32)
            with self.subTest(mask=mask), self.assertRaisesRegex(ContractError, "history_mask"):
                self.validate()
        values, context = synthetic_inputs(self.contract, ranking_state=1, rounds=(0,))
        with self.assertRaisesRegex(ContractError, "padding_side"):
            self.validate(values, replace(context, padding_side="left"))

    def test_padding_values_and_metadata_must_be_empty(self):
        for name in HISTORY_FIELDS:
            values = copy.deepcopy(self.values)
            values["recursive_history"][name][0] = 0.1
            with self.subTest(name=name), self.assertRaisesRegex(ContractError, "padded values"):
                self.validate(values)
        _, real_context = synthetic_inputs(self.contract, ranking_state=1, rounds=(0,))
        with self.assertRaisesRegex(ContractError, "Padded"):
            self.validate(context=replace(self.context, entries=real_context.entries))

    def test_real_entries_require_complete_availability_metadata(self):
        values, context = synthetic_inputs(self.contract, ranking_state=1, rounds=(0,))
        with self.assertRaisesRegex(ContractError, "provenance"):
            self.validate(values, replace(context, entries=(None, None, None)))
        observations = dict(context.entries[0].observations)
        observations.pop("prev_accepted_value")
        with self.assertRaisesRegex(ContractError, "missing fields"):
            self.validate(values, replace(context, entries=(HistoryEntry(0, observations), None, None)))

    def test_history_context_cutoff_and_policies_are_required(self):
        with self.assertRaisesRegex(ContractError, "history_context"):
            self.contract.validate_inputs(self.values, layer_order=SYNTHETIC_LAYER_ORDER, history_context=None)
        for context in (replace(self.context, ranking_state=True), replace(self.context, ranking_state=-1),
                        replace(self.context, ranking_state=1), replace(self.context, padding_side="unknown"),
                        replace(self.context, order="unknown"), replace(self.context, entries=())):
            with self.subTest(context=context), self.assertRaises(ContractError):
                self.validate(context=context)

    def test_history_rounds_cannot_be_future_duplicate_or_wrong_order(self):
        for ranking_state, rounds in ((1, (1,)), (2, (0, 0)), (2, (1, 0))):
            values, context = synthetic_inputs(self.contract, ranking_state=ranking_state, rounds=rounds)
            with self.subTest(rounds=rounds), self.assertRaises(ContractError):
                self.validate(values, context)

    def test_history_policies_reject_nonstring_metadata(self):
        for bad in (None, 1, ["right"], np.array(["right", "left"])):
            for name in ("padding_side", "order"):
                with self.subTest(name=name, bad=bad), self.assertRaises(ContractError):
                    self.validate(context=replace(self.context, **{name: bad}))

    def change_observation(self, values, context, field, source, available_at):
        entries = list(context.entries)
        slot = next(index for index, entry in enumerate(entries) if entry is not None)
        entry = entries[slot]
        observations = dict(entry.observations)
        observations[field] = ObservationAvailability(source, available_at)
        entries[slot] = replace(entry, observations=observations)
        return self.validate(values, replace(context, entries=tuple(entries)))

    def test_realized_h2_rejected_before_endpoint_even_if_file_claims_available(self):
        values, context = synthetic_inputs(self.contract, ranking_state=1, rounds=(0,))
        for available_at in (1, 2):
            with self.subTest(available_at=available_at), self.assertRaisesRegex(ContractError, "unavailable"):
                self.change_observation(values, context, "prev_accepted_value", "applied_h2", available_at)

    def test_observed_h2_is_allowed_only_after_endpoint_and_observation(self):
        values, context = synthetic_inputs(self.contract, ranking_state=2, rounds=(0,))
        self.assertTrue(self.change_observation(values, context, "prev_accepted_value", "applied_h2", 2).structural_only)
        with self.assertRaisesRegex(ContractError, "unavailable"):
            self.change_observation(values, context, "prev_accepted_value", "applied_h2", 3)
        with self.assertRaisesRegex(ContractError, "unavailable"):
            self.change_observation(values, context, "prev_target_change", "applied_h1", 3)

    def test_simulated_h2_and_future_forecast_are_not_accepted_history(self):
        values, context = synthetic_inputs(self.contract, ranking_state=2, rounds=(0,))
        with self.assertRaisesRegex(ContractError, "unsupported history source"):
            self.change_observation(values, context, "prev_accepted_value", "simulated_h2", 2)
        with self.assertRaisesRegex(ContractError, "unavailable"):
            self.change_observation(values, context, "prev_accepted_value", "forecast", 3)
        with self.assertRaisesRegex(ContractError, "unsupported history source"):
            self.change_observation(values, context, "prev_target_change", "forecast", 0)

    def test_observation_metadata_is_strict(self):
        values, context = synthetic_inputs(self.contract, ranking_state=2, rounds=(0,))
        for source in (None, 1, ["forecast"], np.array(["forecast", "applied_h2"])):
            with self.subTest(source=source), self.assertRaisesRegex(ContractError, "source must be a string"):
                self.change_observation(values, context, "prev_accepted_value", source, 0)
        for available_at in (True, -1, 0.5, "0"):
            with self.subTest(available_at=available_at), self.assertRaises(ContractError):
                self.change_observation(values, context, "prev_accepted_value", "forecast", available_at)


if __name__ == "__main__":
    unittest.main()
