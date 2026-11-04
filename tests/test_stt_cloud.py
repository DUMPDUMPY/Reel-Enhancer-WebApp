import httpx
import json

import pytest

from app import config
from app.services import stt_cloud
from app.services.segmenter import build_cues
from app.services.stt_cloud import elevenlabs, gemini, google


class _FakeHttpx:
    """Dispatches httpx.post by URL: the OAuth token call vs the STT call."""

    def __init__(self, token_calls, stt_post):
        self.token_calls = token_calls
        self.stt_post = stt_post

    def post(self, url, **kw):
        if "oauth2.googleapis.com" in url:
            self.token_calls.append((url, kw.get("data")))
            return httpx.Response(200, json={"access_token": "T", "expires_in": 3600},
                                  request=httpx.Request("POST", url))
        return self.stt_post(url, **kw)

# Shape of ElevenLabs Scribe words[] (word / spacing / audio_event).
SCRIBE = {"language_code": "tha", "text": "...", "words": [
    {"text": "(เสียงดนตรี)", "type": "audio_event", "start": 0.0, "end": 1.0, "logprob": 0},
    {"text": "วันนี้", "type": "word", "start": 1.0, "end": 1.4, "logprob": 0},
    {"text": "กุ๊กไก่", "type": "word", "start": 1.4, "end": 1.9, "logprob": 0},
    {"text": " ", "type": "spacing", "start": 1.9, "end": 2.0, "logprob": 0},
    {"text": "แอบหนีเที่ยว", "type": "word", "start": 2.0, "end": 2.8, "logprob": 0},
    {"text": "สวัสดีครับ", "type": "word", "start": 5.0, "end": 5.8, "logprob": 0},
]}


def test_scribe_to_whisper_shape_and_cues():
    out = elevenlabs.to_whisper_json(SCRIBE)
    assert len(out["segments"]) == 1  # music event dropped; the pause is split later, per word
    assert out["segments"][0]["start"] == 1.0 and out["segments"][0]["text"] == "วันนี้กุ๊กไก่ แอบหนีเที่ยวสวัสดีครับ"
    cues = build_cues(out, ("กุ๊กไก่",))
    assert cues[0]["start"] == 1.0 and "กุ๊กไก่" in cues[0]["text"]
    assert cues[-1]["text"] == "สวัสดีครับ" and cues[-1]["start"] == 5.0


def test_scribe_keeps_real_subscribe_line():
    # Whisper's outro-hallucination filter must not drop a line Scribe actually heard.
    data = {"words": [{"text": "อย่าลืมกดติดตามกันด้วยนะ", "type": "word", "start": 0.5, "end": 2.0}]}
    cues = build_cues(elevenlabs.to_whisper_json(data))
    assert [c["text"].replace("\n", "") for c in cues] == ["อย่าลืมกดติดตามกันด้วยนะ"]


def test_transcribe_request(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "k")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    seen = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        seen.update(url=url, headers=headers, data=data)
        return httpx.Response(200, json=SCRIBE, request=httpx.Request("POST", url))

    monkeypatch.setattr(elevenlabs.httpx, "post", fake_post)
    out = elevenlabs.transcribe(wav, hint="กุ๊กไก่, เจ้าเหมียว", model="elevenlabs:scribe_v2")
    assert seen["url"].endswith("/v1/speech-to-text") and seen["headers"] == {"xi-api-key": "k"}
    assert seen["data"]["model_id"] == "scribe_v2" and seen["data"]["timestamps_granularity"] == "word"
    assert seen["data"]["keyterms"] == ["กุ๊กไก่", "เจ้าเหมียว"]
    assert out["segments"]
    assert elevenlabs.models() == ["elevenlabs:scribe_v2"]


def test_default_model_prefers_cloud(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIBE_MODEL", "")
    for key in ("ELEVENLABS_API_KEY", "GOOGLE_CLOUD_API_KEY", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.setattr(config, key, "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert stt_cloud.default_model() is None  # Whisper's own model
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "k")
    assert stt_cloud.default_model() == "elevenlabs:" + config.ELEVENLABS_STT_MODEL
    assert stt_cloud.models() == ["elevenlabs:scribe_v2"]
    monkeypatch.setattr(config, "TRANSCRIBE_MODEL", "ggml-large-v3-q5_0.bin")
    assert stt_cloud.default_model() == "ggml-large-v3-q5_0.bin"
    monkeypatch.setattr(config, "TRANSCRIBE_MODEL", "elevenlabs:scribe_v2")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    assert stt_cloud.default_model() is None  # cloud default without a key falls back


def test_thai_characters_never_split_mid_word():
    # Real Scribe output for a song: Thai "words" are characters, and a held note
    # leaves a 0.5 s gap inside ไหน (ไห | น).
    chars = [("ถ้", 45.12, 45.28), ("า", 45.28, 45.44), ("วั", 45.44, 45.56), ("น", 45.56, 45.76),
             ("ไ", 45.76, 45.8), ("ห", 45.8, 45.86), ("น", 46.38, 46.4), ("ไ", 46.4, 46.52),
             ("ม่", 46.52, 46.84), ("มี", 46.84, 47.24), ("เ", 47.24, 47.28), ("ธ", 47.28, 47.44), ("อ", 47.44, 47.56),
             ("ฉั", 48.6, 48.7), ("น", 48.7, 48.8), ("ร้", 48.8, 49.0), ("อ", 49.0, 49.1), ("ง", 49.1, 49.3)]
    data = {"words": [{"text": t, "type": "word", "start": a, "end": b} for t, a, b in chars]}
    cues = build_cues(elevenlabs.to_whisper_json(data))
    texts = [c["text"].replace("\n", "") for c in cues]
    assert texts == ["ถ้าวันไหนไม่มีเธอ", "ฉันร้อง"]  # split at the 1 s pause between words only
    assert cues[0]["start"] == 45.12


def test_segment_edges_keep_word_times():
    data = {"words": [
        {"text": "สวัสดี", "type": "word", "start": 1.0, "end": 1.5},
        {"text": " ", "type": "spacing", "start": 1.5, "end": 1.7},
        {"text": "ครับ", "type": "word", "start": 1.7, "end": 2.0},
        {"text": " ", "type": "spacing", "start": 2.0, "end": 3.0},  # long trailing spacing
    ]}
    seg = elevenlabs.to_whisper_json(data)["segments"][0]
    assert (seg["start"], seg["end"]) == (1.0, 2.0) and seg["words"][-1]["word"] == "ครับ"


# ---------- Gemini 3.5 Transcribe (Interactions API) ----------

# REST shape from ai.google.dev/api/interactions-api: word offsets are text like "0.100s"
# attached as word_info annotations on the model-output steps.
GEMINI_INTERACTION = {
    "status": "completed",
    "steps": [{
        "type": "model_output",
        "content": [{
            "type": "text",
            "text": "วันนี้กุ๊กไก่แอบหนีเที่ยวสวัสดีครับ",
            "annotations": [
                {"type": "word_info", "text": "วันนี้", "start_offset": "1.0s", "end_offset": "1.4s"},
                {"type": "word_info", "text": "กุ๊กไก่", "start_offset": "1.4s", "end_offset": "1.9s"},
                {"type": "word_info", "text": "แอบหนีเที่ยว", "start_offset": "2.0s", "end_offset": "2.8s"},
                {"type": "word_info", "text": "สวัสดีครับ", "start_offset": "5.0s", "end_offset": "5.8s"},
            ],
        }],
    }],
}


def test_gemini_request_and_parse(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(config, "GEMINI_STT_LANGUAGES", "th-TH")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json)
        return httpx.Response(200, json=GEMINI_INTERACTION, request=httpx.Request("POST", url))

    monkeypatch.setattr(gemini.httpx, "post", fake_post)
    out = gemini.transcribe(wav, hint="กุ๊กไก่", model="gemini:gemini-3.5-transcribe")
    assert seen["url"].endswith("/v1beta/interactions")
    assert seen["headers"] == {"x-goog-api-key": "k"}
    assert seen["json"]["model"] == "models/gemini-3.5-transcribe"
    tconf = seen["json"]["generation_config"]["transcription_config"]
    assert tconf["language_codes"] == ["th-TH"]
    assert tconf["mode"] == {"type": "verbatim", "timestamp_granularities": ["word"]}  # required for cue timing
    assert out["segments"][0]["start"] == 1.0
    assert out["segments"][-1]["start"] == 5.0 and out["segments"][-1]["text"] == "สวัสดีครับ"
    cues = build_cues(out, ("กุ๊กไก่",))
    assert "กุ๊กไก่" in cues[0]["text"]
    # Word timestamps block custom vocabulary, so hint words are never sent.
    assert "custom_vocabulary" not in seen["json"]["generation_config"]["transcription_config"]
    assert gemini.models() == ["gemini:gemini-3.5-transcribe"]


def test_gemini_splits_on_long_gap_not_word_gap(monkeypatch, tmp_path):
    # Like Scribe, Gemini can return single Thai characters; a 0.5 s gap inside
    # ไหน must not split the word, the 1 s pause between words must.
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    chars = [("ถ้", 45.12, 45.28), ("าวั", 45.28, 45.56), ("นไ", 45.56, 45.8),
             ("ห", 45.8, 45.86), ("น", 46.38, 46.4), ("ไม่", 46.4, 46.84), ("มีเ", 46.84, 47.28),
             ("ธอ", 47.28, 47.56), ("ฉั", 48.6, 48.8), ("นร้อง", 48.8, 49.3)]
    data = {
        "status": "completed",
        "steps": [{"type": "model_output", "content": [{
            "type": "text", "text": "ถ้าวันไหนไม่มีเธอฉันร้อง",
            "annotations": [
                {"type": "word_info", "text": t, "start_offset": f"{a}s", "end_offset": f"{b}s"}
                for t, a, b in chars
            ],
        }]}],
    }
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(gemini.httpx, "post",
                        lambda *a, **k: httpx.Response(200, json=data, request=httpx.Request("POST", "u")))
    cues = build_cues(gemini.transcribe(wav))
    assert [c["text"].replace("\n", "") for c in cues] == ["ถ้าวันไหนไม่มีเธอ", "ฉันร้อง"]


def test_gemini_quota_error_message(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(
        gemini.httpx, "post",
        lambda *a, **k: httpx.Response(429, json={}, request=httpx.Request("POST", "u")))
    try:
        gemini.transcribe(wav)
        raise AssertionError("must raise")
    except RuntimeError as e:
        assert "โควตา" in str(e)  # Thai message telling the user what to do


# ---------- Google Cloud Speech-to-Text v2 (Chirp, OAuth2 service account) ----------

SA_CREDS = {
    "client_email": "sa@proj.iam.gserviceaccount.com",
    "private_key": "-----BEGIN PRIVATE KEY-----\nGGGG\n-----END PRIVATE KEY-----\n",
    "private_key_id": "kid",
}

# REST shape of Speech.Recognize (v2): word offsets are {seconds, nanos} maps.
GOOGLE_RECOGNIZE = {
    "results": [
        {"alternatives": [{
            "transcript": "วันนี้กุ๊กไก่ แอบหนีเที่ยว",
            "words": [
                {"word": "วันนี้", "startOffset": {"seconds": 1, "nanos": 0}, "endOffset": {"seconds": 1, "nanos": 400000000}},
                {"word": "กุ๊กไก่", "startOffset": {"seconds": 1, "nanos": 400000000}, "endOffset": {"seconds": 1, "nanos": 900000000}},
                {"word": "แอบหนีเที่ยว", "startOffset": {"seconds": 2, "nanos": 0}, "endOffset": {"seconds": 2, "nanos": 800000000}},
            ],
        }]},
        {"alternatives": [{
            "transcript": "สวัสดีครับ",
            "words": [
                {"word": "สวัสดีครับ", "startOffset": {"seconds": 5, "nanos": 0}, "endOffset": {"seconds": 5, "nanos": 800000000}},
            ],
        }]},
    ],
}


@pytest.fixture
def google_auth(monkeypatch, tmp_path):
    """A service-account file + a working token endpoint."""
    import jwt as real_jwt

    creds_path = tmp_path / "sa.json"
    creds_path.write_text(json.dumps(SA_CREDS))
    monkeypatch.setattr(config, "GOOGLE_CLOUD_CREDENTIALS", str(creds_path))
    monkeypatch.setattr(config, "GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setattr(config, "GOOGLE_CLOUD_LOCATION", "us")
    monkeypatch.setattr(google, "_token", None)
    monkeypatch.setattr(real_jwt, "encode", lambda *a, **k: "FAKE.JWT")
    return creds_path


def _token_post(on_calls):
    def fake(url, data=None, timeout=None):
        on_calls.append((url, data))
        return httpx.Response(200, json={"access_token": "T", "expires_in": 3600}, request=httpx.Request("POST", url))
    return fake


def test_google_request_and_parse(monkeypatch, tmp_path, google_auth):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json)
        return httpx.Response(200, json=GOOGLE_RECOGNIZE, request=httpx.Request("POST", url))

    token_calls = []
    monkeypatch.setattr("app.services.stt_cloud.google.httpx", _FakeHttpx(token_calls, fake_post))
    out = google.transcribe(wav, hint="กุ๊กไก่", model="google:chirp_3")
    # The implicit recognizer is part of the URL path (body fields: config + content only).
    assert seen["url"].endswith("/v2/projects/proj/locations/us/recognizers/_:recognize")
    assert seen["headers"] == {"Authorization": "Bearer T"}  # v2 rejects plain API keys
    assert "recognizer" not in seen["json"]
    rconf = seen["json"]["config"]
    assert rconf["model"] == "chirp_3" and rconf["language_codes"] == ["th-TH"]
    assert "recognizer" not in rconf
    assert rconf["features"] == {"enable_word_time_offsets": True}
    texts = [s["text"] for s in out["segments"]]
    assert texts == ["วันนี้กุ๊กไก่แอบหนีเที่ยว", "สวัสดีครับ"]  # long gap made the second segment
    assert out["segments"][-1]["start"] == 5.0
    assert google.models() == ["google:chirp_3"]
    assert token_calls[0][0] == "https://oauth2.googleapis.com/token"
    assert token_calls[0][1]["grant_type"].endswith("jwt-bearer")


def test_google_v1_api_key(monkeypatch, tmp_path):
    """Cloud AIzaSy key → v1 speech:recognize with latest_long (v2 rejects keys)."""
    monkeypatch.setattr(config, "GOOGLE_CLOUD_API_KEY", "AIzaSyTest")
    monkeypatch.setattr(config, "GOOGLE_CLOUD_CREDENTIALS", "")
    monkeypatch.setattr(config, "GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setattr(google, "_token", None)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json)
        return httpx.Response(200, json={
            "results": [{"alternatives": [{
                "transcript": "สวัสดี",
                "words": [{"word": "สวัสดี", "startTime": "1s", "endTime": "1.5s"}],
            }]}],
        }, request=httpx.Request("POST", url))

    monkeypatch.setattr(google.httpx, "post", fake_post)
    out = google.transcribe(wav, model="google:latest_long")
    assert seen["url"] == "https://speech.googleapis.com/v1/speech:recognize?key=AIzaSyTest"
    conf = seen["json"]["config"]
    assert conf["model"] == "latest_long" and conf["languageCode"] == "th-TH"
    assert conf["enableWordTimeOffsets"] is True
    assert "encoding" not in conf  # WAV headers carry it
    assert [s["text"] for s in out["segments"]] == ["สวัสดี"]  # v1 "1s" strings parse too
    assert google.models() == ["google:latest_long"]


def test_google_401_shows_raw_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOOGLE_CLOUD_API_KEY", "")
    monkeypatch.setattr(config, "GOOGLE_CLOUD_CREDENTIALS", "/tmp/sa.json")
    monkeypatch.setattr(config, "GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setattr(google, "_access_token", lambda: "T")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")

    def fake_post(url, headers=None, json=None, timeout=None):
        return httpx.Response(401, json={"error": {"message": "API keys are not supported by this API"}},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(google.httpx, "post", fake_post)
    with pytest.raises(RuntimeError, match="API keys are not supported by this API"):
        google.transcribe(wav, model="google:chirp_3")


def test_google_disabled_without_any_credentials():
    monkey_d = {"GOOGLE_CLOUD_API_KEY": "", "GOOGLE_CLOUD_CREDENTIALS": "", "GOOGLE_CLOUD_PROJECT": "proj"}
    for k, v in monkey_d.items():
        setattr(config, k, v)
    assert google.enabled() is False and google.models() == []
    setattr(config, "GOOGLE_CLOUD_PROJECT", "")
