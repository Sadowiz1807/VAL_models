"""Dynamic TEXT_SLU configuration and append-only label manifests (stdlib only)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

PARAMETER_TYPES = {"ENUM", "NUMBER", "ENTITY", "FREE_TEXT", "BOOLEAN", "STATE_REFERENCE", "CONTEXT_REFERENCE"}
SOURCES = ["INPUT_SPAN", "STATE_REFERENCE", "CONTEXT_REFERENCE"]


def _slot(kind, required=False, **constraints):
    return {"type": kind, "required": required, **constraints}


def _ontology():
    app_state = ["state.current_target_application"]
    browser_state = ["state.active_browser"]
    tab_refs = ["FOCUSED_ACTION", "ACTIVE_ACTION", "LAST_ACTION"]
    app = {}
    for name in ("OPEN", "CLOSE", "FOCUS", "PLAY", "PAUSE", "RESUME", "STOP", "NEXT", "PREVIOUS"):
        parameters = {"application": _slot("ENTITY", name in ("OPEN", "CLOSE", "FOCUS", "PLAY"),
                                           state_reference_paths=list(app_state))}
        if name == "PLAY":
            parameters["query"] = _slot("FREE_TEXT", True)
        app[name] = {"parameters": parameters}
    browser = lambda: _slot("ENTITY", state_reference_paths=list(browser_state))
    web = {
        "OPEN": {"parameters": {"target": _slot("FREE_TEXT", True, state_reference_paths=["state.active_url"]), "browser": browser()}},
        "SEARCH": {"parameters": {"query": _slot("FREE_TEXT", True), "browser": browser(), "engine": _slot("ENTITY")}},
    }
    for name in ("BACK", "FORWARD", "REFRESH"):
        web[name] = {"parameters": {"browser": browser()}}
    for name in ("SCROLL_UP", "SCROLL_DOWN"):
        web[name] = {"parameters": {"amount": _slot("NUMBER", minimum=1, maximum=10), "browser": browser()}}
    web["TAB.NEW"] = {"parameters": {"browser": browser()}}
    for name in ("TAB.CLOSE", "TAB.SWITCH"):
        web[name] = {
            "parameters": {
                "tab_index": _slot("NUMBER", minimum=1, integer=True),
                "tab_query": _slot("FREE_TEXT"),
                "tab_reference": _slot("CONTEXT_REFERENCE", context_reference_types=list(tab_refs)),
            },
            "mutually_exclusive": [["tab_index", "tab_query", "tab_reference"]],
            "needs_context_when_absent": ["tab_index", "tab_query", "tab_reference"],
        }
    web["TAB.REOPEN"] = {"parameters": {}}
    web["PLAY"] = {"parameters": {"query": _slot("FREE_TEXT", True), "site": _slot("ENTITY"), "browser": browser()}}
    for name in ("PAUSE", "RESUME", "STOP", "NEXT", "PREVIOUS"):
        web[name] = {"parameters": {"browser": browser()}}
    system = {
        "MEDIA.SET_VOLUME": {"parameters": {"volume": _slot("NUMBER", True, minimum=0, maximum=100)}},
        "MEDIA.VOLUME_UP": {"parameters": {"amount": _slot("NUMBER", minimum=1, maximum=100)}},
        "MEDIA.VOLUME_DOWN": {"parameters": {"amount": _slot("NUMBER", minimum=1, maximum=100)}},
    }
    for name in ("MEDIA.MUTE", "MEDIA.UNMUTE", "POWER.LOCK", "POWER.SHUTDOWN", "POWER.RESTART", "POWER.SLEEP", "SCREENSHOT.CAPTURE"):
        system[name] = {"parameters": {}}
    domains = ["APPLICATION_CONTROL", "WEB_CONTROL", "SYSTEM_CONTROL", "RUN_COMMAND"]
    return {
        "acts": ["EXECUTE", "ASK_CLARIFICATION", "CONFIRM", "CANCEL", "RESPOND", "UNSUPPORTED"],
        "root_action_domains": domains, "schema_order": list(domains),
        "capabilities": {
            "APPLICATION_CONTROL": {"actions": app}, "WEB_CONTROL": {"actions": web},
            "SYSTEM_CONTROL": {"actions": system},
            "RUN_COMMAND": {"actions": {"EXECUTE": {"parameters": {
                "command_id": _slot("ENTITY", True), "arguments": _slot("FREE_TEXT"),
            }}}},
        },
    }


def create_config():
    return {
        "architecture": "text_semantic_transformer_v1", "config_version": 1,
        "ontology_version": "VALLS-TEXT-ONTOLOGY-1",
        "schema_version": "VALLS-TEXT-SLU-DRAFT-1",
        "normalizer_version": "unicode_nfc_v1", "ontology": _ontology(),
        "tokenizer": {"type": "unicode_character_v1", "max_length": 512},
        "semantic_core": {"d_model": 512, "layers": 4, "attention_heads": 8,
                          "d_ff": 2048, "dropout": 0.1, "gradient_checkpointing": False},
        "operation_decoder": {"max_operations": 4, "attention_heads": 8, "dropout": 0.1},
        "relations": {
            "turn": ["NEW", "APPEND_AFTER", "MODIFY", "SUPERSEDE", "CONTINUE", "REPEAT", "REFERENCE"],
            "operation": ["NONE", "AFTER_SUCCESS", "AFTER_FAILURE", "AFTER_TERMINAL", "AFTER_START"],
            "context_reference_types": ["FOCUSED_ACTION", "ACTIVE_ACTION", "LAST_ACTION", "FOCUSED_TASK_GROUP", "LAST_TASK_GROUP", "RECENT_ACTION"],
        },
        "thresholds": {"act_min_confidence": 0.70, "goal_min_confidence": 0.70,
                       "action_min_confidence": 0.60, "parameter_min_confidence": 0.60,
                       "operation_presence_threshold": 0.50, "slot_presence_threshold": 0.50,
                       "max_ood_score": 0.50},
        "decoding": {"max_span_length": 256, "require_calibration": True},
        "loss_weights": {"act": 1.0, "operation_presence": 1.0, "operation_count": 1.0,
                         "operation_domain_retrieval": 1.0, "action": 1.0, "parameters": 1.0,
                         "turn_relation": 0.5, "context_required": 0.5,
                         "context_reference": 0.5, "operation_relations": 0.5, "ood": 0.5},
        "parameter_loss_weights": {"presence": 1.0, "source": 1.0, "start": 1.0, "end": 1.0,
                                   "state": 1.0, "context": 1.0, "enum": 1.0, "boolean": 1.0},
        "training": {
            "batch_size": 16, "learning_rate": 1e-4, "weight_decay": 0.01,
            "max_epochs": 100, "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0, "num_workers": 0, "seed": 42, "amp": "off",
            "stage": "NLU_JOINT", "parameter_presence_negative_weight": 0.25,
            "parameter_hard_negative_schemas": 4,
        },
        "stages": {
            "SEMANTIC": ["act", "operation_presence", "operation_count", "operation_domain_retrieval", "action",
                         "turn_relation", "context_required", "context_reference", "operation_relations"],
            "PARAMETER": ["parameters"], "SAFETY": ["ood"],
            "NLU_JOINT": ["act", "operation_presence", "operation_count", "operation_domain_retrieval", "action", "parameters",
                          "turn_relation", "context_required", "context_reference", "operation_relations", "ood"],
        },
    }


def _labels(values, name):
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"Invalid {name}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {name}")


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_nlu_config(config):
    if config.get("architecture") != "text_semantic_transformer_v1" or config.get("config_version") != 1:
        raise ValueError("Unsupported NLU architecture/version")
    if config.get("normalizer_version") != "unicode_nfc_v1" or config["tokenizer"]["type"] != "unicode_character_v1":
        raise ValueError("Unsupported normalization/tokenizer profile")
    for name in ("ontology_version", "schema_version"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"Missing {name}")
    _positive_int(config["tokenizer"]["max_length"], "max_length")
    if config["tokenizer"]["max_length"] < 3:
        raise ValueError("max_length must accommodate BOS, a character, EOS")
    o = config["ontology"]
    for name in ("acts", "root_action_domains", "schema_order"):
        _labels(o[name], name)
    if set(o["schema_order"]) != set(o["root_action_domains"]) or set(o["capabilities"]) != set(o["schema_order"]):
        raise ValueError("Domain/order/capability mismatch")
    for name, values in config["relations"].items():
        _labels(values, name)
    if "NONE" not in config["relations"]["operation"]:
        raise ValueError("Operation relations require NONE")
    context_types = set(config["relations"]["context_reference_types"])
    for domain in o["schema_order"]:
        if "." in domain:
            raise ValueError("Root domain must not contain dots")
        actions = o["capabilities"][domain]["actions"]
        if not isinstance(actions, dict) or not actions:
            raise ValueError("Each domain needs actions")
        for action, spec in actions.items():
            if not isinstance(action, str) or any(not part.strip() for part in action.split(".")):
                raise ValueError("Invalid action path")
            if not isinstance(spec.get("parameters"), dict):
                raise ValueError("Missing parameters mapping")
            for name, p in spec["parameters"].items():
                if not isinstance(name, str) or not name.strip() or "." in name:
                    raise ValueError("Invalid parameter name")
                if p.get("type") not in PARAMETER_TYPES or type(p.get("required")) is not bool:
                    raise ValueError("Invalid parameter type/required")
                if p["type"] == "ENUM":
                    _labels(p.get("values"), "enum values")
                for bound in ("minimum", "maximum"):
                    if bound in p and (type(p[bound]) not in (int, float) or not math.isfinite(p[bound])):
                        raise ValueError("Invalid numeric bound")
                if "minimum" in p and "maximum" in p and p["minimum"] > p["maximum"]:
                    raise ValueError("Reversed numeric bounds")
                if "integer" in p and type(p["integer"]) is not bool:
                    raise ValueError("integer constraint must be boolean")
                for key in ("state_reference_paths", "context_reference_types"):
                    if key in p:
                        _labels(p[key], key)
                if not set(p.get("context_reference_types", [])) <= context_types:
                    raise ValueError("Unknown allowed context reference")
                if any(not x.startswith("state.") for x in p.get("state_reference_paths", [])):
                    raise ValueError("Invalid state selector")
                if p["type"] == "STATE_REFERENCE" and not p.get("state_reference_paths"):
                    raise ValueError("STATE_REFERENCE needs an allowlist")
                if p["type"] == "CONTEXT_REFERENCE" and not p.get("context_reference_types"):
                    raise ValueError("CONTEXT_REFERENCE needs an allowlist")
            for group in spec.get("mutually_exclusive", []):
                _labels(group, "mutually exclusive group")
                if len(group) < 2 or not set(group) <= set(spec["parameters"]):
                    raise ValueError("Invalid mutual-exclusion group")
            if not set(spec.get("needs_context_when_absent", [])) <= set(spec["parameters"]):
                raise ValueError("Unknown context fallback slot")
    c, d = config["semantic_core"], config["operation_decoder"]
    for key in ("d_model", "layers", "attention_heads", "d_ff"):
        _positive_int(c[key], key)
    for key in ("max_operations", "attention_heads"):
        _positive_int(d[key], key)
    if c["d_model"] % c["attention_heads"] or c["d_model"] % d["attention_heads"]:
        raise ValueError("Hidden size must be divisible by attention heads")
    if type(c["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    if any(not 0 <= x["dropout"] < 1 for x in (c, d)):
        raise ValueError("Invalid dropout")
    expected_thresholds = {"act_min_confidence", "goal_min_confidence", "action_min_confidence",
                           "parameter_min_confidence", "operation_presence_threshold",
                           "slot_presence_threshold", "max_ood_score"}
    if set(config["thresholds"]) != expected_thresholds:
        raise ValueError("Missing/unknown threshold keys")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
           for x in config["thresholds"].values()):
        raise ValueError("Thresholds must be finite and in [0,1]")
    _positive_int(config["decoding"]["max_span_length"], "max_span_length")
    if type(config["decoding"]["require_calibration"]) is not bool:
        raise ValueError("require_calibration must be boolean")
    t = config["training"]
    for key in ("batch_size", "max_epochs", "gradient_accumulation_steps"):
        _positive_int(t[key], key)
    for key in ("num_workers", "parameter_hard_negative_schemas"):
        if type(t[key]) is not int or t[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("learning_rate", "gradient_clip_norm"):
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("weight_decay", "parameter_presence_negative_weight"):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"Invalid {key}")
    if type(t["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if t["amp"] not in ("off", "fp16", "bf16") or t["stage"] not in config["stages"]:
        raise ValueError("Unsupported AMP/stage")
    implemented = {"act", "operation_presence", "operation_count", "operation_domain_retrieval", "action", "parameters",
                   "turn_relation", "context_required", "context_reference", "operation_relations", "ood"}
    if set(config["loss_weights"]) != implemented:
        raise ValueError("Loss weights must match implemented objectives")
    if set(config["parameter_loss_weights"]) != {"presence", "source", "start", "end", "state", "context", "enum", "boolean"}:
        raise ValueError("Invalid parameter loss components")
    for weights in (config["loss_weights"], config["parameter_loss_weights"]):
        if any(not math.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
    for stage, losses in config["stages"].items():
        _labels(losses, f"stage {stage}")
        if not set(losses) <= implemented or not any(config["loss_weights"][x] > 0 for x in losses):
            raise ValueError("Stage has unknown losses or no positive loss weights")


def preserve_ids(old_names, candidate_names, label):
    if len(old_names) != len(set(old_names)) or len(candidate_names) != len(set(candidate_names)):
        raise ValueError(f"Duplicate {label}")
    if not set(old_names) <= set(candidate_names):
        raise ValueError(f"Removal/rename requires explicit migration: {label}")
    old_set = set(old_names)
    return list(old_names) + [x for x in candidate_names if x not in old_set]


def _semantic_schema(schema):
    return {k: v for k, v in schema.items() if k != "description"}


def build_registry(config, previous=None):
    validate_nlu_config(config)
    previous = previous or {}
    o = config["ontology"]
    acts = preserve_ids(previous.get("acts", []), o["acts"], "acts")
    domains = preserve_ids(previous.get("domains", []), o["schema_order"], "domains")
    candidates, slot_specs, states, action_specs = [], {}, [], {}
    for domain in o["schema_order"]:
        for path, spec in o["capabilities"][domain]["actions"].items():
            canonical = f"{domain}.{path}"
            candidates.append(canonical)
            action_specs[canonical] = {k: copy.deepcopy(v) for k, v in spec.items() if k not in ("parameters", "description")}
            for name, parameter in spec["parameters"].items():
                key = f"{canonical}.{name}"
                slot_specs[key] = {"canonical": canonical, "name": name, "schema": copy.deepcopy(parameter)}
                for state in parameter.get("state_reference_paths", []):
                    if state not in states:
                        states.append(state)
    actions = preserve_ids(previous.get("actions", []), candidates, "actions")
    old_slots = previous.get("slots", [])
    keys = preserve_ids([s["key"] for s in old_slots], list(slot_specs), "slots")
    for old in old_slots:
        if _semantic_schema(old["schema"]) != _semantic_schema(slot_specs[old["key"]]["schema"]):
            raise ValueError(f"Schema migration required: {old['key']}")
    for action, old in previous.get("action_constraints", {}).items():
        if old != action_specs[action]:
            raise ValueError(f"Action constraint migration required: {action}")
    slots = [{"key": key, "name": slot_specs[key]["name"],
              "action_id": actions.index(slot_specs[key]["canonical"]),
              "schema": slot_specs[key]["schema"]} for key in keys]
    result = {"acts": acts, "domains": domains, "actions": actions,
              "action_to_domain": [domains.index(name.split(".", 1)[0]) for name in actions],
              "slots": slots, "state_paths": preserve_ids(previous.get("state_paths", []), states, "state_paths"),
              "context_types": preserve_ids(previous.get("context_types", []), config["relations"]["context_reference_types"], "context_types"),
              "turn_relations": list(config["relations"]["turn"]),
              "operation_relations": list(config["relations"]["operation"]),
              "action_constraints": action_specs}
    for name in ("turn_relations", "operation_relations", "context_types"):
        if name in previous and previous[name] != result[name]:
            raise ValueError(f"Reference/relation migration is not part of label expansion: {name}")
    return result


def validate_registry(config, registry):
    if build_registry(config, previous=registry) != registry:
        raise ValueError("Manifest/config mismatch or corrupt IDs; explicit migration required")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def save_config(config, path):
    validate_nlu_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    validate_nlu_config(config)
    return config
