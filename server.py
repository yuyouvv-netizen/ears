"""ears · 让你的AI听懂你的语气

按住说话 → 转写(说了什么) + 声学特征(怎么说的) + 个人化基线(和她平时比怎么样)
→ LLM综合判断此刻的状态 → 展示/推给你自己的AI。

设计原则：
- 轻量：pip一分钟装完，不碰torch，普通电脑随便跑
- 相对：情绪是相对的。同一个音高对A是平静对B是低落，所以存"她自己的平时"做基准
- 克制：声学特征只是线索，不许AI过度解读编故事
- 诚实：音频会发给你配置的云端转写接口（默认Groq），本地默认不留音频

配置见 .env（Python自己读，不依赖shell）。Windows/Mac/Linux通用。
"""
import json
import os
import statistics
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent

# ── 配置：Python自己读.env（已存在的环境变量优先），彻底绕开shell解析的坑 ──
_envf = BASE_DIR / ".env"
if _envf.exists():
    for _line in _envf.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_KEY = os.environ.get("LLM_API_KEY", "") or GROQ_KEY
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
ASR_BASE = os.environ.get("ASR_BASE_URL", "https://api.groq.com/openai/v1")
ASR_MODEL = os.environ.get("ASR_MODEL", "whisper-large-v3")
ASR_LANG = os.environ.get("ASR_LANG", "zh")
# 环境变量用直白名字；SFE_前缀的旧名继续兼容
def _env(name, default=""):
    return os.environ.get(name, "") or os.environ.get("SFE_" + name, "") or default

PROXY = _env("PROXY")  # 例: http://127.0.0.1:7890，云端接口国内直连不通时填
WEBHOOK = _env("WEBHOOK")  # 分析结果POST到这里，接你自己的AI
KEEP_AUDIO = os.environ.get("KEEP_AUDIO", "0") == "1"
# 情绪标签：为"家里养了个AI"的场景设计，逗号分隔可自定义
EMOTIONS = [e.strip() for e in _env(
    "EMOTIONS", "开心,兴奋,撒娇,平静,累,低落,委屈,生气,嘴硬,紧张").split(",") if e.strip()]

DATA_DIR = Path(_env("DATA", str(BASE_DIR / "data")))
LOG_FILE = DATA_DIR / "moments.jsonl"
PROFILE_FILE = DATA_DIR / "profile.json"
CLIPS_DIR = DATA_DIR / "clips"

BASELINE_MIN = 8      # 攒够多少条才启用个人化基线
BASELINE_KEEP = 200   # 滚动窗口大小

_proxies = {"http": PROXY, "https": PROXY} if PROXY else None
_session = requests.Session()
_session.headers["User-Agent"] = "ears/0.1"  # 默认UA会被Cloudflare拦

app = FastAPI()


# ── 声学特征：她"怎么说的" ──

def acoustic_features(wav_path: str) -> dict:
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    dur = len(y) / sr
    if dur < 0.3:
        return {"duration_s": round(dur, 1)}
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
    voiced = f0[(f0 > 60) & (f0 < 500)]
    rms = librosa.feature.rms(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats = {
        "duration_s": round(float(dur), 1),
        "pitch_hz": round(float(np.mean(voiced)), 1) if len(voiced) else 0.0,
        "pitch_range": round(float(np.std(voiced)), 1) if len(voiced) else 0.0,
        "energy": round(float(np.mean(rms)), 4),
        "energy_sway": round(float(np.std(rms)), 4),          # shimmer的近似：音量起伏
        "pause_ratio": round(float(np.mean(rms < np.percentile(rms, 20) * 1.5)), 2),
        "tempo": round(float(np.mean(onset)), 2),
        "tempo_sway": round(float(np.std(onset)), 2),           # 语速起伏：忽快忽慢
        "brightness": round(float(np.mean(cent)), 0),          # 频谱质心：音色亮暗
    }
    if len(voiced) > 2:
        # jitter的近似：相邻帧音高抖动
        feats["pitch_jitter"] = round(float(np.mean(np.abs(np.diff(voiced))) / np.mean(voiced)), 4)
    return feats


# ── 个人化基线：她"平时什么样" ──

BASELINE_KEYS = ["pitch_hz", "pitch_range", "energy", "pause_ratio", "tempo", "tempo_sway", "brightness"]


def load_profile() -> dict:
    try:
        return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_profile(feats: dict) -> None:
    prof = load_profile()
    for k in BASELINE_KEYS:
        if k in feats and feats[k]:
            prof.setdefault(k, []).append(feats[k])
            prof[k] = prof[k][-BASELINE_KEEP:]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(prof, ensure_ascii=False), encoding="utf-8")


KEY_ZH = {"pitch_hz": "音高", "pitch_range": "音高起伏", "energy": "音量",
          "pause_ratio": "停顿", "tempo": "语速", "tempo_sway": "语速起伏",
          "brightness": "音色亮度"}


def relative_view(feats: dict) -> dict:
    """和她自己的平时比。样本不够时返回空——冷启动期间只用绝对特征。

    用中位数±MAD而不是均值±标准差：偶尔几条在地铁/户外录的异常样本
    拉不动中位数，基线天生抗"场景污染"。
    """
    prof = load_profile()
    rel = {}
    for k in BASELINE_KEYS:
        hist = prof.get(k, [])
        if len(hist) < BASELINE_MIN or k not in feats or not feats[k]:
            continue
        med = statistics.median(hist)
        mad = statistics.median([abs(x - med) for x in hist]) * 1.4826  # 折算到σ尺度
        # 离散度设下限（中位数的5%），防止基线样本太相似时z值爆炸
        spread = max(mad, abs(med) * 0.05, 1e-6)
        z = (feats[k] - med) / spread
        if abs(z) < 0.8:
            continue
        direction = "偏高" if z > 0 else "偏低"
        degree = "明显" if abs(z) >= 3 else ("比较" if abs(z) >= 1.5 else "略")
        rel[KEY_ZH.get(k, k)] = degree + direction
    return rel


# ── 转写与判断 ──

def transcribe(wav_path: str) -> str:
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY 未配置")
    with open(wav_path, "rb") as f:
        r = _session.post(
            f"{ASR_BASE}/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": ("a.wav", f, "audio/wav")},
            data={"model": ASR_MODEL, "language": ASR_LANG},
            proxies=_proxies, timeout=60)
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def judge(text: str, feats: dict, rel: dict) -> dict:
    rel_line = f"\n和她平时相比: {json.dumps(rel, ensure_ascii=False)}" if rel else "\n(个人基线还在学习中)"
    prompt = (
        "你在帮一个AI伴侣听懂主人说话的语气。\n"
        f"她说的话:「{text}」\n"
        f"声学特征: {json.dumps(feats, ensure_ascii=False)}{rel_line}\n"
        f"从这些标签里选1个最贴切的: {'/'.join(EMOTIONS)}\n"
        "规则: 特征只是线索, 以说话内容为主; 不要过度解读; hint用一句话描述她此刻的状态, "
        "只描述状态本身, 不许编造原因或事件。\n"
        '只输出JSON: {"emotion":"...","confidence":0.0到1.0,"hint":"..."}'
    )
    r = _session.post(
        f"{LLM_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"},
        json={"model": LLM_MODEL, "max_tokens": 200,
              "messages": [{"role": "user", "content": prompt}]},
        proxies=_proxies, timeout=30)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    s, e = raw.find("{"), raw.rfind("}")
    out = json.loads(raw[s:e + 1])
    if out.get("emotion") not in EMOTIONS:
        out["emotion"] = "平静"
    return out


def fire_webhook(entry: dict) -> None:
    if not WEBHOOK:
        return

    def run():
        try:
            _session.post(WEBHOOK, json=entry, timeout=15)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


# ── 接口 ──

@app.post("/api/listen")
async def listen(file: UploadFile = File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = await file.read()
    clip_name = ""
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("in" + (Path(file.filename or "a.webm").suffix or ".webm"))
        src.write_bytes(raw)
        wav = Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(wav)],
                       capture_output=True, timeout=60)
        if not wav.exists():
            return JSONResponse({"error": "音频转码失败，检查ffmpeg是否安装"}, status_code=400)
        # 先算特征再转写：太短的录音（误触）直接拦下，
        # 不然Whisper面对空白音频会幻觉出"请点赞订阅"之类的训练语料
        feats = acoustic_features(str(wav))
        if feats.get("duration_s", 0) < 0.5:
            return JSONResponse({"error": "太短啦，再说一次"}, status_code=400)
        try:
            text = transcribe(str(wav))
        except Exception as exc:
            return JSONResponse({"error": f"转写失败: {exc}"}, status_code=502)
        if KEEP_AUDIO:
            try:
                CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                clip_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".mp3"
                subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-b:a", "64k",
                                str(CLIPS_DIR / clip_name)], capture_output=True, timeout=60)
                if not (CLIPS_DIR / clip_name).exists():
                    clip_name = ""
            except Exception:
                clip_name = ""
    rel = relative_view(feats)
    try:
        emo = judge(text, feats, rel)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        emo = {"emotion": "平静", "confidence": 0.0, "hint": f"情绪判断失败: {exc}"}
    update_profile(feats)  # 判断完再入库，避免这一条污染自己的基线
    entry = {
        "ts": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "text": text, "emotion": emo.get("emotion", "平静"),
        "confidence": emo.get("confidence", 0), "hint": emo.get("hint", ""),
        "features": feats, "relative": rel, "audio": clip_name,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    fire_webhook(entry)
    baseline_n = min(len(load_profile().get("pitch_hz", [])), BASELINE_MIN)
    return {"ts": entry["ts"], "text": text, "emotion": entry["emotion"],
            "confidence": entry["confidence"], "hint": entry["hint"], "relative": rel,
            "baseline_progress": f"{baseline_n}/{BASELINE_MIN}"}


@app.get("/api/recent")
async def recent(n: int = 20):
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-n:]
    return [json.loads(l) for l in lines]


def rebuild_profile(entries: list) -> None:
    """按剩余记录重建基线——删掉的那条，连它教给基线的东西一起忘掉"""
    prof = {}
    for e in entries[-BASELINE_KEEP:]:
        f = e.get("features", {})
        for k in BASELINE_KEYS:
            if f.get(k):
                prof.setdefault(k, []).append(f[k])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(prof, ensure_ascii=False), encoding="utf-8")


@app.post("/api/forget")
def forget(body: dict):
    """删一条记录：{"ts":"..."} 指定删，{"last":true} 删最近一条。基线同步重建。"""
    if not LOG_FILE.exists():
        return {"ok": False, "error": "还没有任何记录"}
    entries = [json.loads(l) for l in LOG_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    before = len(entries)
    if body.get("last"):
        if entries:
            entries.pop()
    elif body.get("ts"):
        entries = [e for e in entries if e.get("ts") != body["ts"]]
    else:
        return {"ok": False, "error": "要传 ts 或 last:true"}
    if len(entries) == before:
        return {"ok": False, "error": "没找到这条记录"}
    LOG_FILE.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + ("\n" if entries else ""),
        encoding="utf-8")
    rebuild_profile(entries)
    return {"ok": True, "remaining": len(entries)}


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
