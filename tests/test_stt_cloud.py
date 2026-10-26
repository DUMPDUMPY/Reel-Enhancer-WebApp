import httpx

from app import config
from app.services import stt_cloud
from app.services.segmenter import build_cues

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
    out = stt_cloud.to_whisper_json(SCRIBE)
    assert len(out["segments"]) == 2  # music event dropped, long pause splits
    assert out["segments"][0]["start"] == 1.0 and out["segments"][0]["text"] == "วันนี้กุ๊กไก่ แอบหนีเที่ยว"
    cues = build_cues(out, ("กุ๊กไก่",))
    assert cues[0]["start"] == 1.0 and "กุ๊กไก่" in cues[0]["text"]
    assert cues[-1]["text"] == "สวัสดีครับ" and cues[-1]["start"] == 5.0


def test_transcribe_request(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "k")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    seen = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        seen.update(url=url, headers=headers, data=data)
        return httpx.Response(200, json=SCRIBE, request=httpx.Request("POST", url))

    monkeypatch.setattr(stt_cloud.httpx, "post", fake_post)
    out = stt_cloud.transcribe(wav, hint="กุ๊กไก่, เจ้าเหมียว", model="elevenlabs:scribe_v2")
    assert seen["url"].endswith("/v1/speech-to-text") and seen["headers"] == {"xi-api-key": "k"}
    assert seen["data"]["model_id"] == "scribe_v2" and seen["data"]["timestamps_granularity"] == "word"
    assert seen["data"]["keyterms"] == ["กุ๊กไก่", "เจ้าเหมียว"]
    assert out["segments"]
    assert stt_cloud.models() == ["elevenlabs:scribe_v2"]


def test_default_model_prefers_cloud(monkeypatch):
    monkeypatch.setattr(config, "TRANSCRIBE_MODEL", "")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    assert stt_cloud.default_model() is None  # Whisper's own model
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "k")
    assert stt_cloud.default_model() == "elevenlabs:" + config.ELEVENLABS_STT_MODEL
    monkeypatch.setattr(config, "TRANSCRIBE_MODEL", "ggml-large-v3-q5_0.bin")
    assert stt_cloud.default_model() == "ggml-large-v3-q5_0.bin"
    monkeypatch.setattr(config, "TRANSCRIBE_MODEL", "elevenlabs:scribe_v2")
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    assert stt_cloud.default_model() is None  # cloud default without a key falls back
