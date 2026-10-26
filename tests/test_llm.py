import json

import httpx
import pytest

from app.services import llm


def test_split_ref():
    assert llm.split_ref("zai:glm-4.7-flash") == (llm.PROVIDERS["zai"], "glm-4.7-flash")
    assert llm.split_ref("typhoon:typhoon-v2.5-30b-a3b-instruct")[1] == "typhoon-v2.5-30b-a3b-instruct"
    # Ollama names contain ":" too; without a known provider prefix they stay Ollama models.
    assert llm.split_ref("gemma3:4b") == (llm.PROVIDERS["ollama"], "gemma3:4b")
    assert llm.split_ref("hf.co/scb10x/typhoon2.5-qwen3-4b-gguf:latest")[0].is_ollama
    assert llm.ref(llm.PROVIDERS["zai"], "glm-4.7-flash") == "zai:glm-4.7-flash"


def test_extract_json_handles_fences_and_prose():
    assert json.loads(llm._extract_json('```json\n{"items": []}\n```')) == {"items": []}
    assert json.loads(llm._extract_json('นี่คือผลลัพธ์ {"items": [{"id": 1, "text": "ก"}]} ครับ'))["items"][0]["id"] == 1


def test_default_model_prefers_cloud_key(monkeypatch):
    monkeypatch.setattr(llm.config, "PROOFREAD_MODEL", "")
    monkeypatch.setattr(llm.PROVIDERS["zai"], "api_key", "")
    assert llm.default_model() == llm.config.OLLAMA_MODEL
    monkeypatch.setattr(llm.PROVIDERS["zai"], "api_key", "k")
    assert llm.default_model() == "zai:glm-5.3"
    monkeypatch.setattr(llm.config, "PROOFREAD_MODEL", "typhoon:typhoon-v2.5-30b-a3b-instruct")
    assert llm.default_model() == "typhoon:typhoon-v2.5-30b-a3b-instruct"


def test_cloud_chat_sends_key_json_mode_and_parses(monkeypatch):
    monkeypatch.setattr(llm.PROVIDERS["zai"], "api_key", "secret")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        content = '```json\n{"items": [{"id": 1, "text": "กุ๊กไก่"}]}\n```'
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}], "usage": {"completion_tokens": 20},
        }, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    answer, stats = llm.chat_json("zai:glm-4.7-flash", "sys", "user", {}, 30)
    assert answer == {"items": [{"id": 1, "text": "กุ๊กไก่"}]}
    assert seen["url"].endswith("/chat/completions")
    assert seen["headers"] == {"Authorization": "Bearer secret"}
    assert seen["body"]["model"] == "glm-4.7-flash"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["thinking"] == {"type": "disabled"}
    assert stats["eval_tokens"] == 20


def test_cloud_without_key_is_refused(monkeypatch):
    monkeypatch.setattr(llm.PROVIDERS["typhoon"], "api_key", "")
    with pytest.raises(RuntimeError, match="API key"):
        llm.chat_json("typhoon:typhoon-v2.5-30b-a3b-instruct", "s", "u", {}, 5)


def test_typhoon_bare_list_models(monkeypatch):
    p = llm.PROVIDERS["typhoon"]
    monkeypatch.setattr(p, "api_key", "k")
    monkeypatch.setattr(p, "curated", False)
    data = [{"id": "typhoon-v2.5-30b-a3b-instruct"}, {"id": "typhoon-ocr"}, {"id": "typhoon-asr-realtime"}]
    monkeypatch.setattr(llm.httpx, "get", lambda url, headers=None, timeout=None: httpx.Response(
        200, json=data, request=httpx.Request("GET", url)))
    info = llm._list_models(p)
    assert info["ok"] and info["models"] == ["typhoon-v2.5-30b-a3b-instruct"]  # OCR/ASR models filtered out


def test_zai_model_that_always_thinks_is_retried_with_effort(monkeypatch):
    p = llm.PROVIDERS["zai"]
    monkeypatch.setattr(p, "api_key", "secret")
    monkeypatch.setattr(p, "always_think", set())
    bodies = []

    def fake_post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        req = httpx.Request("POST", url)
        if json["thinking"] == {"type": "disabled"}:
            return httpx.Response(400, json={"error": {"code": "1210", "message": "always thinks"}}, request=req)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"items": []}', "reasoning_content": "..."}}]},
                              request=req)

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    answer, stats = llm.chat_json("zai:glm-6-flash", "s", "u", {}, 30)
    assert answer == {"items": []} and stats["thinking"]
    assert bodies[1]["thinking"] == {"type": "enabled"} and bodies[1]["reasoning_effort"] == "high"
    llm.chat_json("zai:glm-6-flash", "s", "u", {}, 30)
    assert len(bodies) == 3  # remembered: no second refusal


def test_glm53_gets_reasoning_effort_up_front(monkeypatch):
    # The Coding Plan endpoint silently ignores thinking=disabled, so no 1210 refusal arrives.
    p = llm.PROVIDERS["zai"]
    monkeypatch.setattr(p, "always_think", set())
    assert p.body_for("glm-5.3-flash")["reasoning_effort"] == "high"
    assert p.body_for("glm-5.3")["thinking"] == {"type": "enabled"}
    assert p.body_for("glm-4.7-flash") == {"thinking": {"type": "disabled"}}
