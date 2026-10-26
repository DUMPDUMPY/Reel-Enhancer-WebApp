# Reel Enhancer & Thai Auto-Sub

เว็บแอปในวง Wi-Fi บ้าน สำหรับคลิป Reels/TikTok ที่ตัดต่อเสร็จแล้ว: อัปโหลดจากมือถือ → AI ถอดเสียงพูดภาษาไทยเป็นซับ 1–2 บรรทัดตรงจังหวะ → แก้คำบนมือถือ → เรนเดอร์ภาพคมขึ้นพร้อมฝังซับ 1080×1920 → ดาวน์โหลดไปโพสต์

งานหนักทั้งหมดรันบน **Intel iGPU** ของ NAS (Core Ultra / Arrow Lake ผ่าน `/dev/dri`) ไม่ดึง CPU

| ส่วน | เครื่องมือ | รันบน |
|---|---|---|
| ถอดเสียง + จับเวลา | whisper.cpp server (Vulkan) + Silero VAD | iGPU |
| ตัดคำ/ตัดบรรทัดไทย | PyThaiNLP (ไม่ตัดกลางคำ, ไม่ตัดกลางชื่อตัวละคร) | CPU (เบามาก) |
| ตรวจคำผิด (เลือกได้) | API ที่ใส่ key (Z.AI GLM, Typhoon, OpenRouter, Gemini) หรือ Ollama ในเครื่อง | Cloud / iGPU ของ Ollama |
| ซับ .ass | pysubs2 + ฟอนต์ Kanit / Prompt / Sarabun | – |
| ขยาย/ลดนอยส์/เพิ่มความคม/เข้ารหัส | FFmpeg VAAPI (`scale_vaapi`, `denoise_vaapi`, `sharpness_vaapi`, `h264_vaapi`) | iGPU |
| วาดตัวหนังสือซับลงภาพ | libass | CPU (เบามาก) |
| เว็บ | FastAPI + HTML/JS (Python 3.14) | – |

ถ้า iGPU ใช้ไม่ได้ แอปจะถอยไปใช้ CPU (libx264) ให้เองอัตโนมัติ

## ติดตั้งบน Synology DSM (Container Manager)

1. คัดลอกโฟลเดอร์โปรเจกต์นี้ไปไว้ที่ NAS เช่น `/volume2/docker/reel-enhancer` (หรือ `git clone` ผ่าน SSH หลังลงแพ็กเกจ Git Server)
   - ต้องมีโฟลเดอร์ `data/` และ `models/` อยู่ในนั้น (มีมาให้ใน repo แล้ว ถ้าหายให้สร้างเป็นโฟลเดอร์ว่าง) เพราะ Container Manager ไม่สร้างให้เอง
   - Container Manager ตั้งเพดาน CPU (`cpus:`) ไม่ได้ จึงไม่มีใน `docker-compose.yml` การใช้ CPU ตอนเรนเดอร์สำรองคุมด้วย `CPU_THREADS`
2. เปิด `docker-compose.yml` แล้วตรวจค่า:
   - `OLLAMA_HOST` ค่าเริ่มต้นคือ `http://host.docker.internal:11434` ถ้าเชื่อมไม่ได้ ให้เปลี่ยนเป็น IP ของ NAS เช่น `http://192.168.1.4:11434`
   - `OLLAMA_MODEL` ต้องเป็นโมเดลที่**สร้างข้อความได้** (โมเดล embedding ใช้ไม่ได้) เช่น `gemma3:4b`, `qwen2.5:3b`
3. Container Manager → **Project** → **Create** → เลือกโฟลเดอร์นี้ → ใช้ `docker-compose.yml` ที่มีอยู่ → Build
   - ครั้งแรกจะ build whisper.cpp แบบ Vulkan (ประมาณ 10–20 นาที)
   - ตอนเปิดครั้งแรก container `reel-whisper` จะดาวน์โหลดโมเดลราว 1.1 GB ลงโฟลเดอร์ `models/`
4. ดึงโมเดลสำหรับตรวจคำผิดใน Ollama (ทำครั้งเดียว):
   ```sh
   docker exec -it ollama ollama pull gemma3:4b
   ```
5. เปิดจากมือถือ: `http://<IP-NAS>:8000` (เช่น `http://192.168.1.4:8000`)

ป้ายสถานะด้านบนของหน้าเว็บบอกว่าแต่ละส่วนพร้อมไหม: **Whisper ✓**, **Ollama ✓**, **Render iGPU**

### ตรวจว่าใช้ iGPU จริง

```sh
# whisper: ดูบรรทัด "Vulkan devices" ตอนเริ่ม ต้องเห็นชื่อ Intel GPU
docker logs reel-whisper | head -20
# FFmpeg: ต้องเห็น VAEntrypointEncSlice หรือ EncSliceLP ของ H264
docker exec reel-enhancer vainfo
# ผลตรวจที่แอปใช้ตัดสินใจ
curl http://<IP-NAS>:8000/api/system
```

ถ้าเปลี่ยนการตั้งค่า GPU แล้ว ให้ตรวจใหม่ได้ด้วย `curl -X POST http://<IP-NAS>:8000/api/hwcheck`

## วิธีใช้

1. **อัปโหลดคลิป** และใส่ชื่อตัวละคร/คำเฉพาะคั่นด้วยจุลภาค (เช่น `กุ๊กไก่, เจ้าเหมียว`)
   คำพวกนี้ใช้เป็น prompt ให้ Whisper, เป็นพจนานุกรมตอนตัดบรรทัด (ไม่ตัดกลางชื่อ) และเป็นบริบทให้ Ollama
2. **ถอดเสียงจับเวลา** — เปิด "ตัดช่วงที่มีแต่เพลง/เงียบ (VAD)" ไว้ เพื่อลดการเดาคำในช่วงที่มีแต่เพลง
3. **ตรวจและแก้ซับ** — พรีวิวสดบนวิดีโอ แตะแก้ข้อความ, ปรับเวลา (⏱เริ่ม / ⏱จบ = เวลาวิดีโอตอนนี้), เพิ่มหรือลบท่อน
   ถ้าเปิด Ollama ไว้ คำที่ AI แนะนำให้แก้จะขึ้นเป็นรายการให้กด "ใช้" หรือ "ไม่ใช้" ทีละจุด (AI ไม่แก้ซับเองโดยตรง และไม่ยุ่งกับเวลา)
4. **Enhance & ฝังซับ** → ดูผล → **Download Master MP4**
   บน Android ไฟล์จะไปอยู่ในโฟลเดอร์ Downloads และแอปแกลเลอรี/Google Photos จะเห็นทันที

### 🧪 ทดสอบ/เปรียบเทียบโมเดล Ollama

แท็บ **ทดสอบโมเดล** ที่หัวหน้าเว็บ: เลือกโมเดลได้หลายตัว (สูงสุด 8) แล้วกด "เริ่มทดสอบ" ระบบจะส่งข้อความชุดเดียวกันให้ทีละตัว และแสดงความคืบหน้าเป็นโหนด หนึ่งโหนดต่อหนึ่งโมเดล

- **ชุดทดสอบมาตรฐาน** (14 ท่อน มีเฉลย): มีทั้งชื่อตัวละครสะกดผิด, วรรณยุกต์, ร/ล, คำพ้องเสียง และภาษาพูดที่ห้ามแก้ (เช่น "น้า", "มากกกก")
  - คะแนน = (จำนวนจุดที่แก้ถูก − จำนวนประโยคที่ถูกอยู่แล้วแต่ไปแก้) ÷ จำนวนจุดที่ต้องแก้ × 100
- **ซับจากงานจริง**: ไม่มีเฉลย แสดงคำแนะนำของแต่ละโมเดลเทียบกันรายท่อน
- ผลที่แสดง: เวลาที่ใช้, tok/s, เวลาโหลดโมเดลครั้งแรก, โมเดลคิดก่อนตอบหรือไม่, จำนวนครั้งที่โมเดลเขียนประโยคใหม่จนระบบตัดทิ้ง

> ครั้งแรกที่เรียกแต่ละโมเดล Ollama ต้องโหลดโมเดลก่อน ดูเวลาโหลดได้จากหมายเหตุใต้ชื่อโมเดล ถ้าจะเทียบความเร็ว ให้ทดสอบซ้ำอีกรอบ

สเปกไฟล์ที่ได้: 1080×1920, H.264 High, 30 fps, VBR 12 Mbps (สูงสุด 15), AAC 192k, faststart

## ถอดเสียงด้วย ElevenLabs Scribe (ไม่ต้องรัน Whisper บน NAS)

ใส่ `ELEVENLABS_API_KEY` ใน `docker-compose.yml` แล้ว dropdown "โมเดลถอดเสียง" จะมี `elevenlabs:scribe_v2` อยู่ในกลุ่ม Cloud API และ**ถูกเลือกเป็นค่าเริ่มต้น**
- อยากให้ Whisper เป็นค่าเริ่มต้นแทน: ตั้ง `TRANSCRIBE_MODEL=ggml-large-v3-q5_0.bin`
- ถ้าใช้แต่ ElevenLabs จะ Stop container `reel-whisper` ไว้ก็ได้ (ประหยัดแรมราว 2 GB) หน้าเว็บจะเลือก Cloud ให้เอง
- ส่งเฉพาะไฟล์เสียง (16 kHz mono) ไปถอดเสียงพร้อมเวลาระดับคำ จากนั้นจัดท่อนซับบน NAS ด้วยวิธีเดียวกับ Whisper
- ชื่อในช่อง hint จะถูกส่งเป็น `keyterms` เพื่อให้สะกดชื่อถูก (ElevenLabs คิดเพิ่ม 20%) ปิดได้ด้วย `ELEVENLABS_KEYTERMS=0`
- ถ้าไม่ได้ใส่ key ทุกอย่างจะทำงานบน iGPU เหมือนเดิม

## ตรวจคำผิดด้วย API key (ไม่ต้องรันโมเดลบน NAS)

ใส่ key ใน `docker-compose.yml` แล้ว Recreate project โมเดลของเจ้านั้นจะขึ้นใน dropdown ทั้งหน้าทำคลิปและหน้าทดสอบโมเดล

| ตัวแปร | ผู้ให้บริการ | ตัวอย่างโมเดล |
|---|---|---|
| `ZAI_API_KEY` | Z.AI (GLM) | `zai:glm-5.3` (แนะนำ, ตรวจถูก 100%), `zai:glm-5.3-flash` (ต้องใช้ key แบบ Coding Plan), `zai:glm-4.7-flash`, `zai:glm-4.5-flash` (ฟรี) · `ZAI_MODELS=` กำหนดรายการเอง · ค่าเริ่มต้นใช้ endpoint ของ Coding Plan ถ้าเป็น key แบบจ่ายตามใช้ ให้ตั้ง `ZAI_BASE_URL=https://api.z.ai/api/paas/v4` · `ZAI_REASONING_EFFORT` (low/high/max, ค่าเริ่มต้น max) สำหรับ GLM 5.3 ที่ปิด thinking ไม่ได้ |
| `TYPHOON_API_KEY` | opentyphoon.ai (โมเดลไทย) | `typhoon:typhoon-v2.5-30b-a3b-instruct` |
| `OPENROUTER_API_KEY` + `OPENROUTER_MODELS` | OpenRouter | ตามที่ใส่ใน `OPENROUTER_MODELS` |
| `GEMINI_API_KEY` | Google Gemini | `gemini:gemini-2.5-flash` |
| `CUSTOM_LLM_BASE_URL` / `_API_KEY` / `_MODELS` | API แบบ OpenAI-compatible อื่นๆ | `custom:<ชื่อโมเดล>` |

- ชื่อโมเดลขึ้นต้นด้วยชื่อผู้ให้บริการ เช่น `zai:glm-5.3` ถ้าไม่มี prefix จะถือเป็นโมเดล Ollama (`gemma3:4b`)
- `PROOFREAD_MODEL` ตั้งค่าเริ่มต้นได้ ถ้าว่างไว้ ระบบจะใช้ผู้ให้บริการตัวแรกที่มี key แล้วค่อยถอยไปใช้ Ollama
- ข้อความซับ (ไม่ใช่วิดีโอหรือเสียง) จะถูกส่งไปยังผู้ให้บริการนั้น
- ใช้หน้า **ทดสอบโมเดล** เทียบ API กับ Ollama ด้วยชุดทดสอบเดียวกันได้ก่อนเลือกใช้จริง

## การตั้งค่า (environment ใน docker-compose.yml)

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `WHISPER_MODEL` | `ggml-large-v3-q5_0.bin` | โมเดลถอดเสียง (ต้องตั้งให้ตรงกันทั้ง 2 service) |
| `ZAI_API_KEY`, `TYPHOON_API_KEY`, ... | – | API key สำหรับตรวจคำผิดผ่าน cloud (ดูหัวข้อด้านบน) |
| `PROOFREAD_MODEL` | (ว่าง) | โมเดลตรวจคำเริ่มต้น เช่น `zai:glm-5.3` |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | host.docker.internal / `gemma3:4b` | ปล่อย `OLLAMA_HOST` ว่างเพื่อซ่อน Ollama |
| `PROOFREAD_BATCH` / `OLLAMA_TIMEOUT` | `10` / `300` | ส่งซับให้ AI ตรวจทีละกี่ท่อน และรอแต่ละชุดได้นานสุดกี่วินาที (ชุดที่ล้มเหลวจะข้ามไป ชุดอื่นยังใช้ได้) |
| `RENDER_MODE` | `auto` | `auto` / `vaapi` / `cpu` |
| `VIDEO_BITRATE` / `VIDEO_MAXRATE` | `12M` / `15M` | bitrate ของไฟล์ที่ได้ |
| `MAX_LINE_WIDTH` | `20` | จำนวนตัวอักษรต่อบรรทัดตอนจัดท่อนอัตโนมัติ (ไม่นับสระบน/ล่าง วรรณยุกต์) |
| `MAX_CUE_SECONDS` | `4.0` | ความยาวสูงสุดของ 1 ท่อน |
| `JOB_RETENTION_HOURS` | `72` | ลบงานเก่าอัตโนมัติ |

### เปลี่ยนโมเดลถอดเสียง

ใส่ไฟล์ `ggml-*.bin` ไว้ในโฟลเดอร์ `models/` แล้วเลือกจาก dropdown ในหน้าเว็บได้เลย
- `ggml-large-v3-turbo-q5_0.bin` เร็วกว่ามาก แต่แม่นภาษาไทยน้อยกว่า: ตั้ง `WHISPER_MODEL` เป็นชื่อนี้ แล้ว container จะดาวน์โหลดให้เอง
- โมเดลที่ฝึกมาสำหรับภาษาไทย (เช่น Thonburian Whisper) ต้องแปลงเป็น ggml ก่อนด้วยสคริปต์ `models/convert-h5-to-ggml.py` ของ whisper.cpp

### เพิ่มฟอนต์

วางไฟล์ `.ttf` / `.otf` ลงใน `data/fonts/` แล้วรีเฟรชหน้าเว็บ ฟอนต์จะขึ้นใน dropdown

## พัฒนา / ทดสอบ

```sh
pip install -r requirements-dev.txt   # ต้องมี ffmpeg ในเครื่อง
pytest
DATA_DIR=./data RENDER_MODE=cpu uvicorn app.main:app --reload
```

## โครงสร้าง

```
app/
  main.py              FastAPI routes
  jobs.py              เก็บงาน (data/jobs/<id>/job.json) + คิวทำงานทีละงาน
  pipeline.py          ขั้นตอนถอดเสียง / ตรวจคำ / เรนเดอร์
  bench.py             ทดสอบ/เปรียบเทียบโมเดล Ollama (ชุดทดสอบ + ให้คะแนน)
  services/
    transcriber.py     เรียก whisper.cpp server
    segmenter.py       แปลงผล Whisper เป็นท่อนซับ 1–2 บรรทัด (ตัดคำด้วย PyThaiNLP)
    thai_text.py       ความกว้างตัวอักษรไทย, ตัดบรรทัด
    llm.py             ผู้ให้บริการ LLM: Ollama + API แบบ OpenAI-compatible (Z.AI, Typhoon, OpenRouter, Gemini)
    proofreader.py     ตรวจคำผิด (ตอบเป็น JSON ที่ตรวจรูปแบบแล้ว, ลองใหม่ 1 ครั้ง, ล้มเหลวก็ไม่กระทบงาน)
    subtitles.py       สร้าง .ass / .vtt
    renderer.py        คำสั่ง FFmpeg (VAAPI + fallback CPU), ตรวจความสามารถ iGPU
  static/              หน้าเว็บสำหรับมือถือ
docker/whisper/        Dockerfile ของ whisper.cpp (Vulkan)
fonts/                 Kanit, Prompt, Sarabun (SIL Open Font License)
```
