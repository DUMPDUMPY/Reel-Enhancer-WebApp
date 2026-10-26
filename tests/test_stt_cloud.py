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
    assert len(out["segments"]) == 1  # music event dropped; the pause is split later, per word
    assert out["segments"][0]["start"] == 1.0 and out["segments"][0]["text"] == "วันนี้กุ๊กไก่ แอบหนีเที่ยวสวัสดีครับ"
    cues = build_cues(out, ("กุ๊กไก่",))
    assert cues[0]["start"] == 1.0 and "กุ๊กไก่" in cues[0]["text"]
    assert cues[-1]["text"] == "สวัสดีครับ" and cues[-1]["start"] == 5.0


def test_scribe_keeps_real_subscribe_line():
    # Whisper's outro-hallucination filter must not drop a line Scribe actually heard.
    data = {"words": [{"text": "อย่าลืมกดติดตามกันด้วยนะ", "type": "word", "start": 0.5, "end": 2.0}]}
    cues = build_cues(stt_cloud.to_whisper_json(data))
    assert [c["text"].replace("\n", "") for c in cues] == ["อย่าลืมกดติดตามกันด้วยนะ"]


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


def test_thai_characters_never_split_mid_word():
    # Real Scribe output for a song: Thai "words" are characters, and a held note
    # leaves a 0.5 s gap inside ไหน (ไห | น).
    chars = [("ถ้", 45.12, 45.28), ("า", 45.28, 45.44), ("วั", 45.44, 45.56), ("น", 45.56, 45.76),
             ("ไ", 45.76, 45.8), ("ห", 45.8, 45.86), ("น", 46.38, 46.4), ("ไ", 46.4, 46.52),
             ("ม่", 46.52, 46.84), ("มี", 46.84, 47.24), ("เ", 47.24, 47.28), ("ธ", 47.28, 47.44), ("อ", 47.44, 47.56),
             ("ฉั", 48.6, 48.7), ("น", 48.7, 48.8), ("ร้", 48.8, 49.0), ("อ", 49.0, 49.1), ("ง", 49.1, 49.3)]
    data = {"words": [{"text": t, "type": "word", "start": a, "end": b} for t, a, b in chars]}
    cues = build_cues(stt_cloud.to_whisper_json(data))
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
    seg = stt_cloud.to_whisper_json(data)["segments"][0]
    assert (seg["start"], seg["end"]) == (1.0, 2.0) and seg["words"][-1]["word"] == "ครับ"
