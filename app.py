"""ClipForge: transcribe (Groq Whisper), elige momentos (Gemini + picos de volumen y chat) y exporta
clips 9:16 con encuadre automático, subtítulos karaoke y gancho (FFmpeg)."""
import json, os, re, subprocess, tempfile, glob, shutil, time, zipfile, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import threading
from urllib.parse import urlparse, parse_qs, urljoin
import numpy as np
import requests
import gradio as gr

try:
    import cv2
except ImportError:  # sin OpenCV: el seguimiento de cara cae a recorte centrado
    cv2 = None

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
CHUNK_SEC = 1200   # 20 min por trozo de audio: a 32 kbps queda < 25 MB (límite de Groq)
WINDOW_SEC = 6 * 3600  # Gemini lee hasta 6 h de transcripción de una vez: 1 petición por vídeo (la cuota gratis diaria es muy pequeña)
W, H = 1080, 1920
CAM_H = 640        # alto de la webcam en el modo streamer; el juego ocupa el resto
DRIVE = "/content/drive/MyDrive"
MAX_SHOWN = 60     # la web solo muestra los primeros; el resto está en Drive / en el zip

LAYOUTS = ["Seguir cara", "Streamer", "Centro"]
CAM_POS = ["Auto (detecta tu cara)", "Arriba izquierda", "Arriba derecha", "Abajo izquierda", "Abajo derecha"]
COLUMNS = ["Usar", "Nota", "Título", "Inicio (s)", "Fin (s)", "Descripción", "Hashtags"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise gr.Error("FFmpeg falló: " + r.stderr[-500:])


# ───────────────────────── Entrada: descarga, vídeo, transcripción ─────────────────────────

_pool = ThreadPoolExecutor(4)  # descargas de vídeo y chat en segundo plano


def ytdlp(args, url, what):
    r = subprocess.run(["yt-dlp", *args, "--no-simulate", "--print", "title", "--print", "live_status",
                        "--print", "after_move:filepath", url], capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines or not os.path.exists(lines[-1]):
        raise gr.Error(f"No se pudo descargar el {what}: " + r.stderr[-400:])
    return lines[-1], lines[0], lines[1] if len(lines) >= 3 else ""


RETRY = ["--retries", "20", "--fragment-retries", "50", "--retry-sleep", "5"]  # cortes de red en descargas largas


def download_audio(url, work):
    """Solo el audio: pesa ~20 veces menos que el vídeo y basta para transcribir y analizar.
    Devuelve (ruta, título, live_status)."""
    return ytdlp(["-f", "ba/b", *RETRY, "-o", f"{work}/audio.%(ext)s"], url, "audio")


def is_live(url):
    """¿Se está emitiendo ahora mismo? (consulta rápida, sin descargar nada)."""
    try:
        r = subprocess.run(["yt-dlp", "--simulate", "--no-warnings", "--print", "live_status", url],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    except subprocess.TimeoutExpired:
        return False  # si no contesta a tiempo se sigue; la descarga dará su propio error si hace falta
    return r.stdout.strip().splitlines()[-1:] == ["is_live"]


def video_kind(url, live_status, dur):
    """'directo' (VOD de Twitch o directo de YouTube) o 'editado' (vídeo normal de YouTube o archivo corto)."""
    if url and "twitch.tv" in url:
        return "directo"
    if live_status in ("was_live", "post_live", "is_live"):
        return "directo"
    if live_status == "not_live":
        return "editado"
    return "directo" if dur >= 45 * 60 else "editado"  # sin dato: un vídeo de 45 min o más suele ser un directo


def stream_urls(url):
    """Enlaces directos al vídeo (y al audio, si van aparte) en máxima calidad hasta 1080p. FFmpeg lee de ahí solo
    los trozos de cada clip: no se descarga el directo entero (un VOD de 8 h son ~28 GB y sin índice para saltar)."""
    err = "no responde"
    for _ in range(3):
        try:
            r = subprocess.run(["yt-dlp", "-f", "bv*+ba/b", "-S", "res:1080,fps,vcodec:h264", "-g", "--no-warnings", url],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        except subprocess.TimeoutExpired:
            continue
        urls = [line for line in r.stdout.splitlines() if line.startswith("http")]
        if urls:
            return urls[0], (urls[1] if len(urls) > 1 else None)
        err = r.stderr[-300:]
        time.sleep(5)
    raise gr.Error("No se pudo conectar con el vídeo: " + err)


def net(path):
    """Opciones de FFmpeg para leer de internet sin cortarse (solo para enlaces, no para archivos)."""
    return ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"] if str(path).startswith("http") else []


def probe(path):
    """(ancho, alto, fps de salida, duración); ancho/alto 0 si es solo audio. fps 60 si el original va a ≥50."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,avg_frame_rate:format=duration", "-of", "json", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        j = json.loads(r.stdout)
        s = (j.get("streams") or [{}])[0]
        a, b = s.get("avg_frame_rate", "30/1").split("/")
        fps = 60 if float(b) and float(a) / float(b) >= 50 else 30
        return int(s.get("width", 0)), int(s.get("height", 0)), fps, float(j["format"]["duration"])
    except (KeyError, ValueError, ZeroDivisionError):
        raise gr.Error("No se pudo leer el archivo (¿dañado o vacío?).")


_waits = {}  # hilo → hora hasta la que espera a una API (para enseñarlo en la barra de progreso)


def api_sleep(secs):
    """Duerme anotando la espera, para que la barra diga «esperando el límite gratis» y no parezca colgada."""
    me = threading.get_ident()
    _waits[me] = time.time() + secs
    try:
        time.sleep(secs)
    finally:
        _waits.pop(me, None)


def wait_note():
    left = max((t - time.time() for t in list(_waits.values())), default=0)
    return f" · esperando el límite gratis de la IA ({int(left // 60)}m {int(left % 60):02}s)" if left > 1 else ""


def track(futs, progress, lo, hi, text):
    """Espera a que acaben las tareas moviendo la barra de lo a hi; se refresca cada 2 s con las esperas de las APIs."""
    pending, total = set(futs), len(futs)
    while pending:
        _, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
        k = total - len(pending)
        progress(lo + (hi - lo) * k / max(total, 1), desc=f"{text} {k}/{total}{wait_note()}")
    return [f.result() for f in futs]


def wait_seconds(r, default=20.0):
    """Segundos que pide esperar una API: cabecera retry-after, «retryDelay» de Google o «try again in 1m2s» de Groq."""
    try:
        if r.headers.get("retry-after"):
            return float(re.sub(r"[^\d.]", "", r.headers["retry-after"])) + 1
    except ValueError:
        pass
    m = re.search(r'"retryDelay":\s*"([\d.]+)s"', r.text)
    if m:
        return float(m.group(1)) + 1
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", r.text)
    if m:
        h, mi, s = m.groups()
        return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s) + 1
    return default


GROQ_MAX_WAIT = 90 * 60  # el límite por hora del plan gratis puede pedir esperar bastante: se espera hasta 90 min


def groq_chunk(path):
    deadline = time.time() + GROQ_MAX_WAIT
    while True:
        try:
            with open(path, "rb") as f:
                r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                                  headers={"Authorization": f"Bearer {GROQ_KEY}"},
                                  files={"file": f},
                                  data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json",
                                        "timestamp_granularities[]": ["word", "segment"]},
                                  timeout=600)
        except requests.RequestException as e:  # cortes de red: se reintenta
            if time.time() > deadline:
                raise gr.Error(f"Groq no responde: {e}")
            time.sleep(15)
            continue
        if r.status_code == 200:
            return r.json()
        low = r.text.lower()
        if r.status_code == 429 and ("per day" in low or "(asd)" in low or "(rpd)" in low):
            raise gr.Error("Groq: has llegado al límite diario del plan gratis (unas 8 h de audio al día). "
                           "Vuelve mañana: lo ya transcrito queda guardado y no se repite.")
        if r.status_code in (429, 500, 502, 503) and time.time() < deadline:
            w = wait_seconds(r)
            print(f"Groq pide esperar {w:.0f} s (límite del plan gratis); sigo después…")
            api_sleep(min(w, max(1, deadline - time.time())))
            continue
        raise gr.Error("Groq: " + r.text[:300])


def cache_dir(key):
    """Carpeta de caché para un vídeo: en Drive si está conectado (sobrevive a que Colab se apague)."""
    base = f"{DRIVE}/ClipForge/.cache" if os.path.isdir(DRIVE) else os.path.expanduser("~/.clipforge_cache")
    d = f"{base}/{hashlib.sha1(key.encode()).hexdigest()[:16]}"
    os.makedirs(d, exist_ok=True)
    return d


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_json(path, data):
    tmp = path + ".tmp"  # se escribe aparte y se renombra: un corte a mitad no deja un archivo roto
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def transcribe(audio, work, cache=None, progress=lambda *a, **k: None, label=""):
    run(["ffmpeg", "-y", "-i", audio, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
         "-f", "segment", "-segment_time", str(CHUNK_SEC), f"{work}/a%03d.mp3"])
    chunks = sorted(glob.glob(f"{work}/a*.mp3"))

    def one(i_path):
        i, path = i_path
        saved = load_json(f"{cache}/chunk_{i:03d}.json") if cache else None
        if saved is not None:  # ya transcrito en un intento anterior: no se gasta cuota otra vez
            return saved
        j = groq_chunk(path)
        if cache:
            save_json(f"{cache}/chunk_{i:03d}.json", j)
        return j

    with ThreadPoolExecutor(4) as ex:  # los trozos de 20 min se transcriben a la vez (mismo modelo, mismo resultado)
        results = track([ex.submit(one, ic) for ic in enumerate(chunks)], progress, 0.15, 0.45,
                        f"{label}Transcribiendo trozos de 20 min:")
    segs, words = [], []
    for i, j in enumerate(results):
        off = i * CHUNK_SEC
        segs += [{"start": s["start"] + off, "end": s["end"] + off, "text": s["text"].strip()}
                 for s in j.get("segments") or []]
        words += [{"start": w["start"] + off, "end": w["end"] + off, "word": w["word"].strip()}
                  for w in j.get("words") or [] if w.get("word", "").strip()]
    return segs, words or words_from_segs(segs)


def words_from_segs(segs):
    """Si Whisper no da tiempos por palabra, se reparten a partes iguales dentro de cada frase."""
    out = []
    for s in segs:
        ws = s["text"].split()
        d = (s["end"] - s["start"]) / max(len(ws), 1)
        out += [{"start": s["start"] + k * d, "end": s["start"] + (k + 1) * d, "word": w} for k, w in enumerate(ws)]
    return out


# ───────────────────────── Señales: volumen y chat ─────────────────────────

def loud_peaks(db, rise=10, gap=15):
    """Gritos, risas, explosiones: segundos que suben `rise` dB o más sobre la mediana del minuto de alrededor.
    Compara con lo que sonaba justo antes y después, así que funciona aunque el directo cambie de volumen."""
    v = np.asarray(db, dtype=float)
    if len(v) < 61:
        return find_peaks(v, 1, floor=20)  # vídeo de menos de un minuto: comparación con todo el vídeo
    local = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(v, 30, mode="edge"), 61), axis=1)
    rel = v - local
    idx = np.where((rel >= rise) & (v > 20))[0]  # > 20 dB: no cuenta el ruido de fondo de un tramo silenciado
    picked = []
    for i in idx[np.argsort(-rel[idx])]:
        if all(abs(i - p) >= gap for p in picked):
            picked.append(i)
    return sorted(float(i) for i in picked)


def find_peaks(values, step, z=3.0, gap=15, floor=None):
    """Tiempos (s) donde el valor se sale de lo normal, separados al menos `gap` s. Usa mediana y MAD (no media y
    desviación): los tramos silenciados de Twitch hundían la media y ningún grito llegaba a destacar.
    `floor`: valores por debajo (silencio) no cuentan para saber qué es «normal»."""
    v = np.asarray(values, dtype=float)
    base = v[v > floor] if floor is not None else v
    if len(base) < 10:
        return []
    med = np.median(base)
    spread = np.median(np.abs(base - med)) * 1.4826 or base.std()
    if not spread:
        return []
    s = (v - med) / spread
    peaks = []
    for i in np.argsort(-s):
        if s[i] < z:
            break
        if all(abs(i * step - p) >= gap for p in peaks):
            peaks.append(float(i * step))
    return sorted(peaks)


def loudness(video):
    """Volumen (dB) de cada segundo del vídeo."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                         stdout=subprocess.PIPE)
    db = []
    while chunk := p.stdout.read(16000):  # 1 s = 8000 muestras × 2 bytes
        a = np.frombuffer(chunk[: len(chunk) // 2 * 2], np.int16).astype(np.float32)
        db.append(20 * np.log10(np.sqrt(np.mean(a * a)) + 1))
    p.wait()
    return db


MUSIC_RE = re.compile(r"♪|♫|\b(m[uú]sica|music|canci[oó]n|song)\b|amara\.org", re.I)  # amara: alucinación típica con música


def music_ranges(segs, words, db, edited):
    """Tramos con música [(desde, hasta)]: donde Whisper marca música o canto y, en vídeos editados, sonido
    fuerte sin voz durante 6 s o más (intros, transiciones, montajes musicales).
    shortcut: no detecta música flojita de fondo bajo la voz; para eso haría falta un modelo de audio."""
    ranges = [(s["start"], s["end"]) for s in segs if MUSIC_RE.search(s["text"])]
    if edited and len(db) > 10:
        speech = np.zeros(len(db), bool)
        for w in words:
            speech[int(w["start"]):int(w["end"]) + 1] = True
        loud = np.asarray(db) > np.median(db)
        start = None
        for i in range(len(db) + 1):
            m = i < len(db) and loud[i] and not speech[i]
            if m and start is None:
                start = i
            elif not m and start is not None:
                if i - start >= 6:
                    ranges.append((float(start), float(i)))
                start = None
    merged = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def music_share(c, ranges):
    """Fracción del clip que cae en tramos con música."""
    s, e = float(c["start"]), float(c["end"])
    return sum(max(0, min(e, b) - max(s, a)) for a, b in ranges) / max(e - s, 1e-6)


def adapt(src, n, min_s, max_s):
    """Los vídeos editados ya vienen sin relleno: menos clips y más cortos que en un directo."""
    if src.get("kind") != "editado":
        return n, min_s, max_s
    max_s = min(max_s, 40)
    return min(n, max(3, round(src["dur"] / 120))), min(min_s, 15, max_s), max_s


def chat_peaks(url, work):
    """Momentos en que el chat de Twitch/YouTube explota. Si no hay chat o falla, devuelve []."""
    try:
        subprocess.run(["yt-dlp", "--skip-download", "--write-subs", "--sub-langs", "live_chat,rechat",
                        "-o", f"{work}/chat", url], capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        return []
    times = []
    for f in glob.glob(f"{work}/chat*"):
        txt = open(f, encoding="utf-8", errors="ignore").read()
        times += [int(x) / 1000 for x in re.findall(r'"videoOffsetTimeMsec"\s*:\s*"?(\d+)', txt)]  # YouTube
        times += [float(x) for x in re.findall(r'"content_offset_seconds"\s*:\s*([\d.]+)', txt)]  # Twitch
    if len(times) < 50:
        return []
    return find_peaks(np.bincount((np.array(times) // 10).astype(int)), 10)


# ───────────────────────── IA: elegir momentos ─────────────────────────

_models = {}  # caché de la lista de modelos y del que funcionó


def gemini_models():
    """Modelos de texto de tu clave, del mejor al peor: Pro > Flash > Lite y versión más alta primero."""
    if "list" in _models:
        return _models["list"]
    r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_KEY}&pageSize=1000", timeout=30)
    if r.status_code != 200:
        raise gr.Error("Clave de Gemini no válida: " + r.text[:200])
    skip = ("image", "tts", "live", "audio", "embedding", "vision", "learnlm", "gemma", "robotics", "computer",
            "customtools")
    ranked = []
    for m in r.json().get("models", []):
        name = m["name"]
        if not name.startswith("models/gemini") or "generateContent" not in m.get("supportedGenerationMethods", []) \
                or any(s in name for s in skip):
            continue
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        tier = 0 if "lite" in name else 2 if "pro" in name else 1 if "flash" in name else -1
        ranked.append((tier, float(v.group(1)) if v else 0, "preview" not in name, name))
    models = [x[3] for x in sorted(ranked, reverse=True)][:6]
    if not models:
        raise gr.Error("Tu clave de Gemini no tiene modelos de texto disponibles.")
    _models["list"] = models
    return models


_dead = {}  # modelo → hasta cuándo no probarlo (sin acceso o sin cuota del día): no se gastan peticiones en balde


def ask_gemini(prompt):
    good = _models.get("good")
    errors = []
    for model in ([good] if good else []) + [m for m in gemini_models() if m != good]:
        if _dead.get(model, 0) > time.time():
            continue
        for attempt in range(8):  # «espera un poco» (límite por minuto): se espera con el MISMO modelo, no se baja
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={GEMINI_KEY}",
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"responseMimeType": "application/json"}}, timeout=300)
            except requests.RequestException:
                time.sleep(10)
                continue
            if r.status_code in (429, 500, 503) and "PerDay" not in r.text:
                w = min(wait_seconds(r, 20 * (attempt + 1)), 120)
                print(f"Gemini ({model}) pide esperar {w:.0f} s…")
                api_sleep(w)
                continue
            break
        else:
            errors.append(f"{model}: sigue saturado")
            continue
        if r.status_code == 200:
            if model != good:
                print("Gemini usado:", model)
                _models["good"] = model
            try:
                return parse_clips(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            except (KeyError, IndexError):
                return []  # respuesta vacía o cortada en este tramo: se salta, no se pierde el resto
        # sin acceso o sin cuota del día para este modelo: solo entonces se pasa al siguiente
        errors.append(f"{model}: {r.status_code}")
        _dead[model] = time.time() + (365 * 86400 if r.status_code in (403, 404) else 3600)
        print(f"Aviso: {model} no disponible ({r.status_code}); pruebo con el siguiente modelo.")
        if _models.get("good") == model:
            _models.pop("good")
    raise NoQuota("Gemini sin cuota hoy (" + (", ".join(errors) or "todos los modelos en pausa") + ")")


class NoQuota(gr.Error):
    """Sin cuota gratis del día: se pasa al modelo de reserva en vez de fallar."""


def parse_clips(text):
    """Lista de clips del JSON que devuelve la IA (aunque venga rodeado de texto)."""
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else []
    except ValueError:
        return []
    return [c for c in data if isinstance(c, dict) and "start" in c and "end" in c] if isinstance(data, list) else []


# Reserva gratis cuando Gemini se queda sin cuota: modelos de texto de Groq (misma clave, cuota aparte).
GROQ_LLMS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
BACKUP_WINDOW = 900  # Groq gratis admite pocos tokens por minuto: tramos de 15 min por petición


def groq_llms():
    """Modelos de texto que Groq ofrece hoy (cambian a menudo): primero los preferidos, luego el resto."""
    if "groq" not in _models:
        try:
            r = requests.get("https://api.groq.com/openai/v1/models", timeout=30,
                             headers={"Authorization": f"Bearer {GROQ_KEY}"})
            ids = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
        except (requests.RequestException, ValueError, KeyError):
            ids = []
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "distil", "compound", "prompt")
        ids = [i for i in ids if not any(s in i.lower() for s in skip)]
        big = sorted((i for i in ids if i not in GROQ_LLMS),
                     key=lambda i: -max([int(x) for x in re.findall(r"(\d+)b", i.lower())] or [0]))
        _models["groq"] = [m for m in GROQ_LLMS if m in ids or not ids] + big  # sin lista: se prueban los conocidos
    return _models["groq"]


def ask_groq_llm(prompt):
    for model in groq_llms():
        if _dead.get(model, 0) > time.time():
            continue
        for attempt in range(12):
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=300,
                                  headers={"Authorization": f"Bearer {GROQ_KEY}"},
                                  json={"model": model, "temperature": 0.4,
                                        "messages": [{"role": "user", "content": prompt}]})
            except requests.RequestException:
                time.sleep(10)
                continue
            if r.status_code == 429 and "per day" not in r.text.lower():  # límite por minuto: esperar
                api_sleep(min(wait_seconds(r), 120))
                continue
            break
        else:
            continue
        if r.status_code == 200:
            try:
                return parse_clips(r.json()["choices"][0]["message"]["content"])
            except (KeyError, IndexError):
                return []
        _dead[model] = time.time() + (365 * 86400 if r.status_code in (403, 404) else 3600)
        print(f"Aviso: reserva {model} no disponible ({r.status_code}).")
    raise NoQuota("Se ha acabado la cuota gratis de hoy de Gemini y de la reserva de Groq. Se renueva cada día "
                  "(Gemini a las 9:00, hora de España). Lo ya transcrito queda guardado: mañana pulsa el mismo botón "
                  "y seguirá donde lo dejó.")


KIND_HINT = {
    "directo": "Es un DIRECTO largo con partes de relleno: busca los picos (reacciones, risas, jugadas, polémica).",
    "editado": "Es un VÍDEO DE YOUTUBE YA EDITADO, sin relleno: busca la frase, reacción o momento clave que "
               "funcione solo, con planteamiento y remate en pocos segundos.",
}


def pick_window(segs, loud, chat, n, min_s, max_s, kind="directo", music=(), avoid=()):
    try:
        return ask_gemini(window_prompt(segs, loud, chat, n, min_s, max_s, kind, music, avoid))
    except NoQuota:
        print("Gemini sin cuota: elijo los momentos con la reserva gratis de Groq (más lento).")
    groups = {}
    for s in segs:
        groups.setdefault(int(s["start"] // BACKUP_WINDOW), []).append(s)
    per = -(-n // max(len(groups), 1)) + 1
    clips = []
    for g in groups.values():
        a, b = g[0]["start"], g[-1]["end"]
        clips += ask_groq_llm(window_prompt(g, [t for t in loud if a <= t <= b], [t for t in chat if a <= t <= b],
                                            per, min_s, max_s, kind, [(x, y) for x, y in music if x < b and y > a],
                                            avoid))
    return clips


def window_prompt(segs, loud, chat, n, min_s, max_s, kind="directo", music=(), avoid=()):
    transcript = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segs) or "(sin diálogo)"
    signals = ""
    if loud:
        signals += f"\nPicos de VOLUMEN (gritos, risas, explosiones) en los segundos: {', '.join(f'{t:.0f}' for t in loud)}"
    if chat:
        signals += f"\nPicos de actividad del CHAT en los segundos: {', '.join(f'{t:.0f}' for t in chat)}"
    if music:
        signals += ("\nTramos con MÚSICA (EVÍTALOS, dan problemas de derechos de autor): "
                    + ", ".join(f"{a:.0f}-{b:.0f}" for a, b in music))
    if avoid:  # en directo, los tramos nuevos siguen hablando de lo de antes: que no se repita el mismo momento
        signals += ("\nYA SE HAN HECHO CLIPS DE ESTOS MOMENTOS; NO propongas otro clip del mismo suceso aunque se "
                    "vuelva a comentar: " + " | ".join(avoid[-40:]))
    prompt = (f"{KIND_HINT[kind]}\nEres editor experto de YouTube Shorts y TikTok. Con esta transcripción (tiempos en segundos) y estas "
              f"señales, elige los {n} mejores momentos virales: gancho fuerte al inicio, humor, emoción, reacción, "
              f"polémica o jugada épica, que se entiendan sin contexto. Los picos de volumen y de chat suelen marcar "
              f"buenos momentos aunque haya poco diálogo: inclúyelos con unos segundos de contexto antes. "
              f"Cada clip entre {min_s} y {max_s} s, empezando y acabando en límites de frase. Usa el idioma del vídeo. "
              f"Devuelve SOLO JSON: [{{\"start\":num,\"end\":num,\"title\":\"título corto con gancho\","
              f"\"score\":1-100,\"description\":\"descripción de 1-2 frases para el post\","
              f"\"hashtags\":\"#tag1 #tag2 #shorts\"}}]\n{signals}\n\nTRANSCRIPCIÓN:\n{transcript}")
    return prompt


def dedupe(clips, dur, min_s, max_s):
    """Ajusta a la duración del vídeo y quita clips que se solapan >30% con otro de más puntuación."""
    out = []
    for c in sorted(clips, key=lambda c: -float(c.get("score") or 0)):
        try:
            s, e = max(0.0, float(c["start"])), min(dur, float(c["end"]))
        except (TypeError, ValueError):
            continue
        e = min(e, s + max_s)
        if e - s < max(3, min_s * 0.5):
            continue
        if any(min(e, o["end"]) - max(s, o["start"]) > 0.3 * (e - s) for o in out):
            continue
        out.append({**c, "start": s, "end": e})
    return out


def snap(c, w_starts, s_ends):
    """Cortes limpios: empieza justo al inicio de una palabra y acaba al final de una frase (no a media palabra)."""
    s, e = float(c["start"]), float(c["end"])
    i = np.searchsorted(w_starts, s)
    near = [w_starts[j] for j in (i - 1, i) if 0 <= j < len(w_starts) and abs(w_starts[j] - s) <= 2]
    if near:
        s = min(near, key=lambda x: abs(x - s))
    i = np.searchsorted(s_ends, e)
    near = [s_ends[j] for j in (i - 1, i) if 0 <= j < len(s_ends) and abs(s_ends[j] - e) <= 3]
    if near:
        e = min(near, key=lambda x: abs(x - e))
    return {**c, "start": max(0.0, s - 0.15), "end": e + 0.35}  # pequeño margen para no comerse sílabas


def judge(cands, src, min_s, max_s, progress=lambda *a, **k: None):
    """Segunda pasada: la IA revisa cada candidato con su texto exacto, lo puntúa con rigor y afina los cortes."""
    def context(c):
        a, b = c["start"] - 10, c["end"] + 10
        return " ".join(f"[{s['start']:.1f}] {s['text']}" for s in src["segs"] if a <= s["start"] <= b)

    def one(batch):
        items = "\n\n".join(f"ID {k} (clip propuesto {c['start']:.1f}-{c['end']:.1f} s): {context(c)}"
                            for k, c in enumerate(batch))
        prompt = (f"Eres el editor jefe de un canal viral de Shorts/TikTok. Revisa estos clips candidatos con rigor. "
                  f"Puntúa cada uno de 1 a 100 según: gancho en los 3 primeros segundos (lo más importante), se "
                  f"entiende sin contexto, tiene remate o punchline al final, emoción/humor/sorpresa, y sin tiempos "
                  f"muertos. Ajusta start/end para que empiece directamente en el gancho (sin saludos ni relleno) y "
                  f"acabe justo después del remate, entre {min_s} y {max_s} s. Mejora el título para que enganche. "
                  f"Sé duro: un clip mediocre no pasa de 50. Usa el idioma del vídeo. Devuelve SOLO JSON: "
                  f"[{{\"id\":num,\"start\":num,\"end\":num,\"score\":1-100,\"title\":\"...\","
                  f"\"description\":\"...\",\"hashtags\":\"#... #shorts\"}}]\n\n{items}")
        out = []
        try:
            answer = ask_gemini(prompt)
        except NoQuota:  # sin cuota para el juez: se quedan los candidatos tal cual en vez de fallar
            return batch
        for r in answer:
            try:
                k = int(r.get("id", -1))
                if not 0 <= k < len(batch):
                    continue
                c = batch[k]
                s = min(max(float(r["start"]), c["start"] - 10), c["end"])  # sin irse lejos del momento original
                e = min(max(float(r["end"]), s + min_s * 0.5), c["end"] + 10)
            except (TypeError, ValueError):
                continue
            out.append({**c, **{f: r[f] for f in ("score", "title", "description", "hashtags") if r.get(f)},
                        "start": s, "end": e})
        return out or batch  # si el juez falla en un lote, se conservan los candidatos de ese lote

    batches = [cands[i:i + 60] for i in range(0, len(cands), 60)]  # 60 por lote: normalmente 1 sola petición
    with ThreadPoolExecutor(4) as ex:
        res = track([ex.submit(one, b) for b in batches], progress, 0.82, 0.98,
                    f"Juez IA revisando {len(cands)} candidatos, lotes:")
    return [c for r in res for c in r]


def pick_highlights(src, n, min_s, max_s, progress, avoid_music=True):
    """1) Gemini propone el doble de candidatos por tramo (en paralelo), 2) cortes limpios, sin repetidos y sin
    música, 3) un juez IA los repuntúa y afina, 4) se quedan los n mejores."""
    music = src.get("music", []) if avoid_music else []
    kind = src.get("kind", "directo")
    no_music = lambda cs: [c for c in cs if music_share(c, music) <= 0.3]
    n_win = max(1, int(-(-src["dur"] // WINDOW_SEC)))
    per = min(60, -(-2 * n // n_win) + 2)  # el doble de lo pedido: más donde elegir = mejores clips
    w_starts = np.array([w["start"] for w in src["words"]])
    s_ends = np.array(sorted(s["end"] for s in src["segs"]))
    jobs = []
    with ThreadPoolExecutor(4) as ex:
        for i in range(n_win):
            a, b = i * WINDOW_SEC, (i + 1) * WINDOW_SEC
            segs = [s for s in src["segs"] if a <= s["start"] < b]
            loud = [t for t in src["loud"] if a <= t < b]
            chat = [t for t in src["chat"] if a <= t < b]
            mus = [(x, y) for x, y in music if x < b and y > a]
            if segs or loud or chat:
                jobs.append(ex.submit(pick_window, segs, loud, chat, per, min_s, max_s, kind, mus, src.get("avoid", ())))
        clips = [c for res in track(jobs, progress, 0.5, 0.8, "Buscando momentos:") for c in res]
    cands = no_music(dedupe([snap(c, w_starts, s_ends) for c in clips], src["dur"], min_s, max_s))[:2 * n + 10]
    if not cands:
        return []
    judged = [snap(c, w_starts, s_ends) for c in judge(cands, src, min_s, max_s, progress)]
    clips = no_music(dedupe(judged, src["dur"], min_s, max_s))[:n]  # el juez puede mover cortes: se vuelve a filtrar
    return sorted(clips, key=lambda c: c["start"])


# ───────────────────────── Encuadre ─────────────────────────

try:
    CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except AttributeError:  # OpenCV incompleto (pasa en algunos Colab): sin detección de caras, la web sigue funcionando
    cv2 = CASCADE = None
    print("Aviso: OpenCV sin detector de caras; instala opencv-python-headless para usar 'Seguir cara'.")


FACE_H = 360  # alto de las miniaturas en las que se buscan caras
FACE_FPS = 4  # muestras por segundo para seguir la cara


def biggest_face(gray, sc):
    """Cara más grande en una miniatura gris; devuelve (x, y, w, h) en píxeles del vídeo original."""
    faces = CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return x / sc, y / sc, w / sc, h / sc


def gray_frames(video, start, end, w0, h0, fps):
    """Miniaturas grises sacadas por FFmpeg (multihilo y sin decodificar a tamaño completo en Python)."""
    sw = even(w0 * FACE_H / h0)
    r = subprocess.run(["ffmpeg", "-v", "error", *net(video), "-ss", str(start), "-to", str(end), "-i", video, "-an",
                        "-vf", f"fps={fps},scale={sw}:{FACE_H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, np.uint8)[: len(r.stdout) // (sw * FACE_H) * sw * FACE_H].reshape(-1, FACE_H, sw)


def smooth_track(xs, frame_w, dead=0.06, speed=0.3):
    """Cámara virtual: no se mueve si la cara está cerca (zona muerta) y la sigue suave si se aleja."""
    known = [x for x in xs if x is not None]
    cam = known[0] if known else frame_w / 2
    out = []
    for x in xs:
        if x is not None and abs(x - cam) > dead * frame_w:
            cam += (x - cam) * speed
        out.append(cam)
    return out


def face_cmds(src, start, end, fps, crop_w, path):
    """Escribe un archivo sendcmd que mueve el recorte para seguir la cara, fotograma a fotograma."""
    sc = FACE_H / src["h"]
    xs = []
    off = src.get("offset", 0)  # en directo, el vídeo es un trozo que empieza en «offset»
    for g in gray_frames(src["video"], start - off, end - off, src["w"], src["h"], FACE_FPS):
        f = biggest_face(g, sc)
        xs.append(f[0] + f[2] / 2 if f else None)
    if not any(x is not None for x in xs):
        return False
    ts = [k / FACE_FPS for k in range(len(xs))]
    cx = smooth_track(xs, src["w"])
    frames = np.arange(0, end - start, 1 / fps)
    left = np.clip(np.interp(frames, ts, cx) - crop_w / 2, 0, src["w"] - crop_w)
    with open(path, "w") as f:
        f.writelines(f"{t:.3f} crop x {x:.0f};\n" for t, x in zip(frames, left))
    return True


def find_cam(src, pos):
    """Zona de la webcam (x, y, w, h). Auto: alrededor de la cara más repetida a lo largo del vídeo."""
    w0, h0 = src["w"], src["h"]
    if pos != CAM_POS[0]:
        cw, ch = w0 * 0.28, w0 * 0.28 * 9 / 16
        x = 0 if "izquierda" in pos else w0 - cw
        y = 0 if "Arriba" in pos else h0 - ch
        return x, y, cw, ch
    if not cv2:
        return None
    sc, faces = FACE_H / h0, []
    for k in range(1, 25):  # 24 muestras repartidas por todo el vídeo: posición de cámara más fiable
        t = src["dur"] * k / 25
        for g in gray_frames(src["video"], t, t + 0.1, w0, h0, 10)[:1]:
            if f := biggest_face(g, sc):
                faces.append(f)
    if len(faces) < 4:
        return None
    fx, fy, fw, fh = np.median(faces, axis=0)
    cw = min(w0, fw * 3.4)
    ch = min(h0, cw * 0.75)
    x = np.clip(fx + fw / 2 - cw / 2, 0, w0 - cw)
    y = np.clip(fy + fh / 2 - ch * 0.45, 0, h0 - ch)
    return x, y, cw, ch


def fpath(p):
    """Ruta segura dentro de un filtro de FFmpeg (en Windows «C:» rompería el filtro)."""
    return "'" + p.replace("\\", "/").replace(":", r"\:").replace("'", r"\'") + "'"


def even(v):
    return int(v) // 2 * 2


def video_chain(src, start, end, fps, layout, cam, cmd_path):
    """Cadena de filtros FFmpeg que deja la imagen en 1080x1920 según el encuadre elegido."""
    w0, h0 = src["w"], src["h"]
    if layout == LAYOUTS[1] and cam:
        x, y, cw, ch = cam
        gw = even(min(w0, h0 * W / (H - CAM_H)))
        return (f"[0:v]fps={fps},split[a][b];"
                f"[a]crop={even(cw)}:{even(ch)}:{int(x)}:{int(y)},scale={W}:{CAM_H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{CAM_H}[cam];"
                f"[b]crop={gw}:{h0}:{(w0 - gw) // 2}:0,scale={W}:{H - CAM_H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H - CAM_H}[game];[cam][game]vstack")
    crop_w = even(h0 * W / H)
    if crop_w >= w0:  # el vídeo ya es vertical o casi
        return f"[0:v]fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}"
    if layout == LAYOUTS[0] and cv2 and face_cmds(src, start, end, fps, crop_w, cmd_path):
        return f"[0:v]fps={fps},sendcmd=f={fpath(cmd_path)},crop={crop_w}:{h0}:0:0,scale={W}:{H}:flags=lanczos"
    return f"[0:v]fps={fps},crop={crop_w}:{h0}:{(w0 - crop_w) // 2}:0,scale={W}:{H}:flags=lanczos"


# ───────────────────────── Subtítulos karaoke + gancho ─────────────────────────

def ass_time(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02}:{t % 60:05.2f}"


def make_ass(words, start, end, title, hook, sub_margin, path):
    """Subtítulos de 3 palabras con la palabra que se dice resaltada en amarillo y el título-gancho arriba."""
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,DejaVu Sans,84,&H0000E8FF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,3,2,70,70,{sub_margin},1
Style: Hook,DejaVu Sans,66,&H00FFFFFF,&H00FFFFFF,&H00ED3A7C,&H00ED3A7C,-1,0,0,0,100,100,0,0,3,22,0,8,80,80,170,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    clean = lambda s: re.sub(r"[{}\\]", "", s)
    ws = [w for w in words if w["end"] > start and w["start"] < end]
    lines, group = [], []
    for w in ws:
        if group and (len(group) == 3 or w["start"] - group[-1]["end"] > 0.6):
            lines.append(group)
            group = []
        group.append(w)
    if group:
        lines.append(group)
    events = []
    for g in lines:
        a, b = max(g[0]["start"], start) - start, min(g[-1]["end"], end) - start
        text = ""
        for k, w in enumerate(g):
            nxt = g[k + 1]["start"] if k + 1 < len(g) else w["end"]
            text += f"{{\\k{max(1, round((nxt - max(w['start'], start)) * 100))}}}{clean(w['word']).upper()} "
        events.append(f"Dialogue: 0,{ass_time(a)},{ass_time(max(b, a + 0.3))},Sub,,0,0,0,,{{\\fad(60,0)}}{text.strip()}")
    if hook and title:
        events.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(3, end - start))},Hook,,0,0,0,,"
                      f"{{\\fad(150,300)}}{clean(title).upper()}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(events) + "\n")


# ───────────────────────── Render ─────────────────────────

HAS_GPU = shutil.which("nvidia-smi") is not None
GPU_ENC = ["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "19", "-b:v", "12M", "-maxrate", "16M"]
CPU_ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "16M", "-bufsize", "24M"]


def render(src, c, idx, layout, cam, hook):
    work = src["work"]
    s, e = float(c["start"]), float(c["end"])
    name = re.sub(r"[^\w\- ]", "", c.get("title") or "clip")[:40].strip() or "clip"
    out = f"{work}/out/{idx:04d}_{name}.mp4"
    os.makedirs(f"{work}/out", exist_ok=True)
    sub_margin = H - CAM_H - 260 if layout == LAYOUTS[1] and cam else 430  # en streamer: justo bajo la cámara
    make_ass(src["words"], s, e, c.get("title"), hook, sub_margin, f"{work}/c{idx}.ass")
    chain = video_chain(src, s, e, src["fps"], layout, cam, f"{work}/c{idx}.cmd")
    fc = f"{chain},ass={fpath(f'{work}/c{idx}.ass')},format=yuv420p[v]"
    off = src.get("offset", 0)  # en directo, el vídeo es un trozo que empieza en «offset»

    def cmd(hw):
        return (["ffmpeg", "-y"] + (["-hwaccel", "cuda"] if hw else []) +  # decodifica en GPU; el resultado es idéntico
                [*net(src["video"]), "-ss", str(s - off), "-to", str(e - off), "-i", src["video"]] +
                ([*net(src["audio_in"]), "-ss", str(s), "-to", str(e), "-i", src["audio_in"]] if src.get("audio_in") else []) +
                ["-filter_complex", fc, "-map", "[v]", "-map", "1:a" if src.get("audio_in") else "0:a?"] +
                (GPU_ENC if hw else CPU_ENC) +
                ["-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", out])
    done = False
    if HAS_GPU:
        try:
            run(cmd(True))
            done = True
        except gr.Error:
            pass  # sin NVENC/CUDA en este entorno: se hace por CPU con la misma calidad
    if not done:
        run(cmd(False))
    with open(out[:-4] + ".txt", "w", encoding="utf-8") as f:
        f.write(f"{c.get('title', '')}\n\n{c.get('description', '')}\n\n{c.get('hashtags', '')}\n")
    return out


# ───────────────────────── Flujo ─────────────────────────

def sources(video, urls):
    out = [u.strip() for u in (urls or "").splitlines() if u.strip()]
    return out + ([video] if video else [])


def prepare(item, progress, label=""):
    """Analiza el vídeo. Con enlaces baja primero solo el audio y lo analiza mientras el vídeo completo
    y el chat se descargan en segundo plano: el análisis no espera a la descarga."""
    if not GROQ_KEY or not GEMINI_KEY:
        raise gr.Error("Faltan GROQ_API_KEY o GEMINI_API_KEY en los Secretos.")
    free_disk()
    work = tempfile.mkdtemp()
    _works.append(work)
    is_url = re.match(r"https?://", item)
    key = item if is_url else f"{os.path.basename(item)}:{os.path.getsize(item)}"
    cache = cache_dir(key)
    if is_url and is_live(item):  # un directo en emisión se descargaría durante horas y quedaría a medias
        raise gr.Error("Este directo aún se está emitiendo. Espera a que termine y pega el enlace del vídeo guardado "
                       "(en Twitch: Vídeos › Emisiones anteriores).")
    # _v2: análisis con picos de volumen por contraste local (los viejos no tenían picos); la transcripción se reutiliza
    saved = load_json(f"{cache}/analysis_v2.json")
    if saved:  # ya analizado antes (aunque Colab se apagara): se reutiliza al instante y sin gastar cuota
        progress(0.4, desc=f"{label}Usando el análisis guardado de este vídeo…")
        return dict(saved, work=work, video=None if is_url else item, cache=cache, url=item if is_url else None)
    chat_job, status = None, ""
    if is_url:
        chat_job = _pool.submit(chat_peaks, item, work)
        progress(0.02, desc=f"{label}Descargando audio…")
        audio, title, status = download_audio(item, work)
        video = None
    else:
        audio = video = item
        title = os.path.splitext(os.path.basename(item))[0]
    dur = probe(audio)[3]
    kind = video_kind(item if is_url else "", status, dur)
    progress(0.15, desc=f"{label}Transcribiendo ({'directo' if kind == 'directo' else 'vídeo editado'})…")
    db_job = _pool.submit(loudness, audio)  # el volumen se mide mientras Groq transcribe
    segs, words = transcribe(audio, work, cache, progress, label)
    db = db_job.result()
    loud = loud_peaks(db)
    progress(0.45, desc=f"{label}Leyendo el chat…")
    chat = chat_job.result() if chat_job else []
    if not segs and not loud:
        raise gr.Error("No se detectó voz ni sonido en el vídeo.")
    data = dict(title=title, dur=dur, segs=segs, words=words, loud=loud, chat=chat, kind=kind,
                music=[list(r) for r in music_ranges(segs, words, db, kind == "editado")])
    save_json(f"{cache}/analysis_v2.json", data)
    return dict(data, work=work, video=video, cache=cache, url=item if is_url else None)


_works = []  # carpetas de trabajo de esta sesión, de la más antigua a la más nueva


def free_disk(keep=2):
    """Borra el audio descargado de análisis antiguos para no llenar el disco de Colab.
    Los clips ya hechos (carpeta out/) se conservan."""
    for w in _works[:-keep] if len(_works) > keep else []:
        for f in glob.glob(f"{w}/source.*") + glob.glob(f"{w}/audio.*") + glob.glob(f"{w}/a*.mp3"):
            try:
                os.remove(f)
            except OSError:
                pass


def cached_pick(src, n, min_s, max_s, progress, avoid_music):
    """Momentos elegidos, guardados por vídeo y ajustes: repetir la búsqueda es instantáneo y gratis."""
    path = f"{src['cache']}/picks_v2_{n}_{min_s}_{max_s}_{int(bool(avoid_music))}.json" if src.get("cache") else None
    saved = load_json(path) if path else None
    if saved:
        progress(0.9, desc="Usando los momentos ya encontrados…")
        return saved
    clips = pick_highlights(src, n, min_s, max_s, progress, avoid_music)
    if path and clips:
        save_json(path, clips)
    return clips


def ensure_video(src, progress, label=""):
    """Deja listo de dónde leer el vídeo y sus medidas. Con enlaces se piden cada vez: caducan a las pocas horas."""
    if src.get("url"):
        progress(0, desc=f"{label}Conectando con el vídeo…")
        src["video"], src["audio_in"] = stream_urls(src["url"])
    elif not src.get("video") or not os.path.exists(src["video"]):
        raise gr.Error("El vídeo ya no está en Colab: vuelve a subirlo.")
    if "w" not in src:
        src["w"], src["h"], src["fps"], _ = probe(src["video"])
        if not src["w"]:
            raise gr.Error("El archivo no tiene imagen de vídeo.")


def to_rows(clips):
    return [[True, c.get("score"), c.get("title", ""), round(c["start"], 1), round(c["end"], 1),
             c.get("description", ""), c.get("hashtags", "")] for c in clips]


def from_rows(rows):
    clips = []
    for r in rows:
        try:
            use, score, title, s, e, desc, tags = list(r)[:7]
            s, e = float(s), float(e)
        except (TypeError, ValueError):
            continue
        if use in (True, "true", "True", 1) and e > s:
            clips.append(dict(score=score, title=str(title or ""), start=s, end=e,
                              description=str(desc or ""), hashtags=str(tags or "")))
    return clips


def drive_dest(src, to_drive, folder):
    if not to_drive:
        return None
    if not os.path.isdir(DRIVE):
        raise gr.Error("Google Drive no está conectado. Ejecuta la celda de Colab que monta Drive y vuelve a probar.")
    # carpeta elegida dentro de "Mi unidad"; se quitan ".." para no salir de Drive
    folder = "/".join(p for p in (folder or "").replace("\\", "/").split("/") if p.strip() not in ("", ".", ".."))
    sub = re.sub(r'[\\/:*?"<>|]', "", src["title"]).strip()[:100] or time.strftime("%Y-%m-%d_%H-%M")
    dest = f"{DRIVE}/{folder or 'ClipForge'}/{sub}"
    os.makedirs(dest, exist_ok=True)
    return dest


def export(src, clips, layout, cam_pos, hook, to_drive, folder, progress, label=""):
    if not clips:
        raise gr.Error("No hay clips marcados para exportar.")
    dest = drive_dest(src, to_drive, folder)
    ensure_video(src, progress, label)
    if layout == LAYOUTS[1] and cam_pos == CAM_POS[0]:
        progress(0, desc=f"{label}Buscando la webcam del streamer…")
    cam = find_cam(src, cam_pos) if layout == LAYOUTS[1] else None
    if layout == LAYOUTS[1] and not cam:
        gr.Warning("No encontré tu cámara: elige su posición a mano. Uso encuadre centrado.")

    def job(i, c):
        f = render(src, c, i, layout, cam, hook)
        if dest:  # se copia al momento: si Colab se corta, lo hecho ya está en Drive
            shutil.copy(f, dest)
            shutil.copy(f[:-4] + ".txt", dest)
        return f

    files, errors = [None] * len(clips), []
    with ThreadPoolExecutor(3 if HAS_GPU else 2) as ex:  # varios clips a la vez, cada uno con la misma calidad
        futs = {ex.submit(job, i, c): i - 1 for i, c in enumerate(clips, 1)}
        t0 = time.time()
        progress(0, desc=f"{label}Creando clips 0/{len(clips)}…")
        for k, fut in enumerate(as_completed(futs), 1):
            left = (time.time() - t0) / k * (len(clips) - k)  # media de lo que llevan los terminados
            eta = f" · quedan ~{int(left // 60)}m {int(left % 60):02}s" if k < len(clips) else ""
            progress(k / len(clips), desc=f"{label}Creando clips {k}/{len(clips)}{eta}")
            try:
                files[futs[fut]] = (fut.result(), clips[futs[fut]])
            except gr.Error as e:  # un clip que falla no tira los demás
                errors.append(str(e))
    files = [f for f in files if f]
    if errors:
        gr.Warning(f"{len(errors)} clip(s) fallaron al renderizar: {errors[0][:200]}")
    if not files:
        raise gr.Error("No se pudo renderizar ningún clip. " + (errors[0][:300] if errors else ""))
    if dest:
        gr.Info(f"{len(files)} clips guardados en Drive: {dest[len(DRIVE) + 1:]}")
    return files


def results(files, work):
    gallery = [(f, f"{c.get('score')} · {c.get('title')}") for f, c in files[:MAX_SHOWN]]
    zip_path = f"{work}/clips.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # sin comprimir: el MP4 ya va comprimido
        for f, _ in files:
            z.write(f, os.path.basename(f))
            z.write(f[:-4] + ".txt", os.path.basename(f)[:-4] + ".txt")
    return gallery, zip_path


def fmt_dur(s):
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02}m" if s >= 3600 else f"{s // 60}m {s % 60:02}s"


def stats_html(items):
    """Fila de cifras: [(etiqueta, valor), ...]."""
    cells = "".join(f'<div class="stat"><small>{l}</small><b>{v}</b></div>' for l, v in items)
    return f'<div class="stats">{cells}</div>'


def analysis_stats(src, clips):
    return stats_html([("Tipo", "Directo" if src.get("kind") == "directo" else "Vídeo editado"),
                       ("Duración", fmt_dur(src["dur"])),
                       ("Tramos con música", len(src.get("music", []))),
                       ("Palabras", f"{len(src['words']):,}".replace(",", ".")),
                       ("Picos de volumen", len(src["loud"])),
                       ("Picos de chat", len(src["chat"]) if src["chat"] else "—"),
                       ("Momentos", len(clips))])


def done_stats(files, to_drive, videos=1):
    total = sum(float(c["end"]) - float(c["start"]) for _, c in files)
    best = max((float(c.get("score") or 0) for _, c in files), default=0)
    return stats_html([("Clips", len(files)), ("Vídeos", videos), ("Duración total", fmt_dur(total)),
                       ("Mejor nota", f"{best:.0f}"), ("Guardado en", "Drive" if to_drive else "Descarga")])


# ───────────────────────── Publicar: cuentas ─────────────────────────

YT_SCOPE = "https://www.googleapis.com/auth/youtube"  # el único de YouTube que permite el login por código
TT_API = "https://open.tiktokapis.com/v2"
PRIVACY = {"Público": "public", "Oculto": "unlisted", "Privado": "private"}
PLATFORMS = ["YouTube Shorts", "TikTok (bandeja de borradores)"]
_pending = {}  # login de YouTube a medias (código del dispositivo)


def accounts_path():
    """Las cuentas se guardan en tu Drive (si está conectado) para no tener que conectar cada vez."""
    base = f"{DRIVE}/ClipForge" if os.path.isdir(DRIVE) else os.path.expanduser("~")
    return f"{base}/.clipforge_cuentas.json"


def load_accounts():
    try:
        with open(accounts_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_accounts(acc):
    p = accounts_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(acc, f)
    try:
        os.chmod(p, 0o600)  # solo tu usuario puede leer los tokens
    except OSError:
        pass


def need_env(*keys):
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise gr.Error("Faltan estos Secretos en Colab: " + ", ".join(missing) + ". Mira la guía en el botón «Cuentas» (arriba a la derecha).")
    return [os.environ[k] for k in keys]


def yt_start():
    cid, = need_env("YT_CLIENT_ID")
    r = requests.post("https://oauth2.googleapis.com/device/code", data={"client_id": cid, "scope": YT_SCOPE}, timeout=30)
    if r.status_code != 200:
        raise gr.Error("Google rechazó el login: " + r.text[:300])
    j = _pending["yt"] = r.json()
    return (f'<div class="code-box"><span>1. Abre</span><a href="{j["verification_url"]}" target="_blank">'
            f'{j["verification_url"]}</a><span>2. Escribe este código</span><b>{j["user_code"]}</b>'
            f'<span>3. Elige tu canal, acepta y pulsa <b style="font-size:1em">Ya lo autoricé</b></span></div>')


def yt_finish():
    cid, secret = need_env("YT_CLIENT_ID", "YT_CLIENT_SECRET")
    j = _pending.get("yt")
    if not j:
        raise gr.Error("Primero pulsa «Conectar YouTube».")
    wait, deadline = j.get("interval", 5), time.time() + 90
    while time.time() < deadline:
        d = requests.post("https://oauth2.googleapis.com/token", timeout=30, data={
            "client_id": cid, "client_secret": secret, "device_code": j["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}).json()
        if "access_token" in d:
            acc = load_accounts()
            acc["youtube"] = {"refresh_token": d.get("refresh_token"), "access_token": d["access_token"],
                              "expires": time.time() + d.get("expires_in", 3600) - 60}
            ch = requests.get("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                              headers={"Authorization": f"Bearer {d['access_token']}"}, timeout=30).json()
            items = ch.get("items") or []
            acc["youtube"]["name"] = items[0]["snippet"]["title"] if items else "tu canal"
            save_accounts(acc)
            _pending.pop("yt", None)
            return accounts_html()
        if d.get("error") == "slow_down":
            wait += 5
        elif d.get("error") != "authorization_pending":
            raise gr.Error("YouTube: " + d.get("error_description", d.get("error", "error desconocido")))
        time.sleep(wait)
    raise gr.Error("Aún no veo la autorización. Termina en la página de Google y vuelve a pulsar «Ya lo autoricé».")


def yt_token():
    yt = load_accounts().get("youtube")
    if not yt:
        raise gr.Error("YouTube no está conectado (botón «Cuentas», arriba a la derecha).")
    if time.time() < yt.get("expires", 0):
        return yt["access_token"]
    cid, secret = need_env("YT_CLIENT_ID", "YT_CLIENT_SECRET")
    d = requests.post("https://oauth2.googleapis.com/token", timeout=30, data={
        "client_id": cid, "client_secret": secret, "refresh_token": yt.get("refresh_token"),
        "grant_type": "refresh_token"}).json()
    if "access_token" not in d:
        raise gr.Error("La sesión de YouTube caducó: vuelve a conectarla en el botón «Cuentas» (arriba a la derecha).")
    acc = load_accounts()
    acc["youtube"].update(access_token=d["access_token"], expires=time.time() + d.get("expires_in", 3600) - 60)
    save_accounts(acc)
    return d["access_token"]


def tt_login_url():
    key, redirect = need_env("TIKTOK_CLIENT_KEY", "TIKTOK_REDIRECT_URI")
    q = requests.compat.urlencode({"client_key": key, "scope": "user.info.basic,video.upload",
                                   "response_type": "code", "redirect_uri": redirect, "state": os.urandom(8).hex()})
    url = f"https://www.tiktok.com/v2/auth/authorize/?{q}"
    return (f'<div class="code-box"><span>1. Abre</span><a href="{url}" target="_blank">Iniciar sesión en TikTok</a>'
            f'<span>2. Acepta. Te llevará a otra página: copia la dirección completa de la barra del navegador '
            f'y pégala abajo.</span></div>')


def tt_connect(pasted):
    key, secret, redirect = need_env("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "TIKTOK_REDIRECT_URI")
    pasted = (pasted or "").strip()
    code = parse_qs(urlparse(pasted).query).get("code", [pasted])[0]
    if not code:
        raise gr.Error("Pega la dirección a la que te llevó TikTok tras aceptar.")
    d = requests.post(f"{TT_API}/oauth/token/", timeout=30, data={
        "client_key": key, "client_secret": secret, "code": code, "grant_type": "authorization_code",
        "redirect_uri": redirect}).json()
    if "access_token" not in d:
        raise gr.Error("TikTok: " + d.get("error_description", d.get("error", "código no válido o caducado")))
    acc = load_accounts()
    acc["tiktok"] = {"access_token": d["access_token"], "refresh_token": d.get("refresh_token"),
                     "expires": time.time() + d.get("expires_in", 86400) - 60}
    me = requests.get(f"{TT_API}/user/info/?fields=display_name", timeout=30,
                      headers={"Authorization": f"Bearer {d['access_token']}"}).json()
    acc["tiktok"]["name"] = (me.get("data") or {}).get("user", {}).get("display_name") or "tu cuenta"
    save_accounts(acc)
    return accounts_html(), ""


def tt_token():
    tt = load_accounts().get("tiktok")
    if not tt:
        raise gr.Error("TikTok no está conectado (botón «Cuentas», arriba a la derecha).")
    if time.time() < tt.get("expires", 0):
        return tt["access_token"]
    key, secret = need_env("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET")
    d = requests.post(f"{TT_API}/oauth/token/", timeout=30, data={
        "client_key": key, "client_secret": secret, "grant_type": "refresh_token",
        "refresh_token": tt.get("refresh_token")}).json()
    if "access_token" not in d:
        raise gr.Error("La sesión de TikTok caducó: vuelve a conectarla en el botón «Cuentas» (arriba a la derecha).")
    acc = load_accounts()
    acc["tiktok"].update(access_token=d["access_token"], refresh_token=d.get("refresh_token", tt.get("refresh_token")),
                         expires=time.time() + d.get("expires_in", 86400) - 60)
    save_accounts(acc)
    return d["access_token"]


def disconnect(which):
    acc = load_accounts()
    acc.pop(which, None)
    save_accounts(acc)
    return accounts_html()


def accounts_html():
    acc = load_accounts()

    def card(key, name):
        a = acc.get(key)
        state = f'<span class="dot on"></span>Conectado como {a.get("name", "")}' if a else \
                '<span class="dot"></span>Sin conectar'
        return f'<div class="acc"><b>{name}</b><span class="acc-state">{state}</span></div>'
    return f'<div class="accs">{card("youtube", "YouTube")}{card("tiktok", "TikTok")}</div>'


# ───────────────────────── Publicar: subidas ─────────────────────────

def post_text(c):
    tags = " ".join(t for t in str(c.get("hashtags") or "").split() if t.startswith("#"))
    return str(c.get("title") or "").strip(), str(c.get("description") or "").strip(), tags


def yt_upload(path, c, privacy):
    tok = yt_token()
    title, desc, tags = post_text(c)
    title = (title[:88].rstrip() + " #shorts").strip()  # YouTube: máx. 100 caracteres
    body = {"snippet": {"title": title, "description": f"{desc}\n\n{tags}".strip(), "categoryId": "24",
                        "tags": [t.lstrip("#") for t in tags.split()][:15]},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    r = requests.post("https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                      json=body, timeout=60, headers={"Authorization": f"Bearer {tok}",
                                                      "X-Upload-Content-Type": "video/mp4",
                                                      "X-Upload-Content-Length": str(os.path.getsize(path))})
    if r.status_code != 200 or "Location" not in r.headers:
        raise gr.Error("YouTube: " + r.text[:250])
    with open(path, "rb") as f:
        up = requests.put(r.headers["Location"], data=f, headers={"Content-Type": "video/mp4"}, timeout=3600)
    if up.status_code not in (200, 201):
        raise gr.Error("YouTube: " + up.text[:250])
    return f"https://youtube.com/shorts/{up.json()['id']}"


TT_CHUNK = 10 * 1024 * 1024  # TikTok: trozos de 5–64 MB; un vídeo ≤64 MB va de una vez


def tt_chunks(size):
    """[(desde, hasta)] según las reglas de TikTok: el último trozo se queda con los bytes sobrantes."""
    if size <= 64 * 1024 * 1024:
        return [(0, size - 1)]
    n = size // TT_CHUNK
    return [(i * TT_CHUNK, size - 1 if i == n - 1 else (i + 1) * TT_CHUNK - 1) for i in range(n)]


def tt_upload(path):
    tok = tt_token()
    size = os.path.getsize(path)
    parts = tt_chunks(size)
    r = requests.post(f"{TT_API}/post/publish/inbox/video/init/", timeout=60,
                      headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json; charset=UTF-8"},
                      json={"source_info": {"source": "FILE_UPLOAD", "video_size": size,
                                            "chunk_size": parts[0][1] - parts[0][0] + 1,
                                            "total_chunk_count": len(parts)}})
    d = r.json()
    if (d.get("error") or {}).get("code") != "ok":
        raise gr.Error("TikTok: " + (d.get("error") or {}).get("message", r.text[:250]))
    with open(path, "rb") as f:
        for a, b in parts:
            f.seek(a)
            up = requests.put(d["data"]["upload_url"], data=f.read(b - a + 1), timeout=1800,
                              headers={"Content-Type": "video/mp4", "Content-Range": f"bytes {a}-{b}/{size}"})
            if up.status_code not in (200, 201, 206):
                raise gr.Error(f"TikTok: fallo al subir ({up.status_code})")
    return "en tu bandeja de TikTok: ábrela en el móvil para publicarlo"


def clip_label(f, c):
    return f"{os.path.basename(f)[:4]} · nota {c.get('score')} · {c.get('title')}"


def on_publish(made, picked, platforms, privacy, progress=gr.Progress()):
    sel = [(f, c) for f, c in (made or []) if clip_label(f, c) in (picked or [])]
    if not sel:
        raise gr.Error("Elige al menos un clip.")
    if not platforms:
        raise gr.Error("Elige YouTube, TikTok o ambos.")
    rows, total, done = [], len(sel) * len(platforms), 0
    for f, c in sel:
        for p in platforms:
            progress(done / total, desc=f"Subiendo a {p}: {c.get('title')}…")
            try:
                res = yt_upload(f, c, PRIVACY[privacy]) if p == PLATFORMS[0] else tt_upload(f)
                link = f'<a href="{res}" target="_blank">{res}</a>' if res.startswith("http") else res
                rows.append(f"<tr><td class='ok'>Subido</td><td>{p}</td><td>{c.get('title')}</td><td>{link}</td></tr>")
            except gr.Error as e:  # un fallo no para el resto de subidas
                rows.append(f"<tr><td class='err'>Error</td><td>{p}</td><td>{c.get('title')}</td><td>{getattr(e, 'message', e)}</td></tr>")
            done += 1
    return f'<table class="pub-log"><tr><th>Estado</th><th>Dónde</th><th>Clip</th><th>Resultado</th></tr>{"".join(rows)}</table>'


# ───────────────────────── Acciones de la web ─────────────────────────

def made_outputs(files, stats):
    """Salidas comunes al terminar de exportar: galería, zip, cifras, lista para publicar."""
    gallery, zip_path = results(files, files[0][0].rsplit("/out/", 1)[0] if files else tempfile.mkdtemp())
    labels = [clip_label(f, c) for f, c in files]
    return (gr.update(value=gallery, visible=True), gr.update(value=zip_path, visible=True), stats, files,
            gr.update(choices=labels, value=labels), "")  # "" limpia el registro de publicaciones anterior


def check_sources(video, urls):
    if not sources(video, urls):
        raise gr.Error("Pega un enlace o sube un vídeo para continuar.")


def on_analyze(video, urls, n, min_s, max_s, avoid_music, progress=gr.Progress()):
    items = sources(video, urls)
    check_sources(video, urls)
    if len(items) > 1:
        gr.Info("Reviso solo el primer enlace. Para procesar toda la lista usa «Hacerlo todo automático».")
    src = prepare(items[0], progress)
    n2, a2, b2 = adapt(src, int(n or 10), min_s, max_s)
    if (n2, a2, b2) != (int(n or 10), min_s, max_s):
        gr.Info(f"Es un vídeo editado: busco {n2} clips de {a2}–{b2} s.")
    clips = cached_pick(src, n2, a2, b2, progress, avoid_music)
    if not clips:
        raise gr.Error("No encontré momentos destacados en este vídeo.")
    return src, to_rows(clips), analysis_stats(src, clips)


def on_export(src, rows, layout, cam_pos, hook, to_drive, folder, progress=gr.Progress()):
    if not src:
        raise gr.Error("Primero busca los momentos.")
    rows = rows.values.tolist() if hasattr(rows, "values") else rows
    files = export(src, from_rows(rows), layout, cam_pos, hook, to_drive, folder, progress)
    return made_outputs(files, done_stats(files, to_drive))


def on_auto(video, urls, n, min_s, max_s, avoid_music, layout, cam_pos, hook, to_drive, folder,
            progress=gr.Progress()):
    items = sources(video, urls)
    check_sources(video, urls)
    all_files, failed = [], []
    for k, item in enumerate(items, 1):
        label = f"[{k}/{len(items)}] "
        try:
            src = prepare(item, progress, label)
            clips = cached_pick(src, *adapt(src, int(n or 10), min_s, max_s), progress, avoid_music)
            all_files += export(src, clips, layout, cam_pos, hook, to_drive, folder, progress, label)
        except gr.Error as e:  # un vídeo que falla no para la cola
            failed.append(f"{item[:60]}: {e}")
            gr.Warning(f"Falló {item[:60]}")
    if not all_files:
        raise gr.Error("No se generó ningún clip. " + " | ".join(failed)[:400])
    if failed:
        gr.Warning(f"{len(failed)} vídeo(s) fallaron: " + " | ".join(failed)[:300])
    return made_outputs(all_files, done_stats(all_files, to_drive, len(items) - len(failed)))


# ───────────────────────── Directo en vivo ─────────────────────────
# Twitch va grabando el directo en un vídeo que crece. Se leen solo las piezas (de 10 s) del tramo que interesa:
# FFmpeg no sabe saltar dentro de un vídeo que sigue creciendo, pero las piezas sí se pueden bajar sueltas.

LIVE_WINDOW = 1800   # cada 30 min de directo se buscan clips
LIVE_BACKLOG = 7200  # lo ya emitido se analiza en tramos de 2 h (menos peticiones a la IA)
LIVE_LAG = 60        # se deja 1 min de margen con el directo: las últimas piezas aún se están escribiendo


def http_get(url):
    for k in range(4):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(3 * (k + 1))
    raise gr.Error("Twitch no responde al bajar un trozo del directo.")


def hls_index(m3u8):
    """(cabecera, [(desde, hasta, url)], duración) de la lista de piezas de un vídeo de Twitch."""
    init, segs, t, d = None, [], 0.0, 0.0
    for line in http_get(m3u8).text.splitlines():
        if line.startswith("#EXT-X-MAP:") and 'URI="' in line:
            init = urljoin(m3u8, line.split('URI="')[1].split('"')[0])
        elif line.startswith("#EXTINF:"):
            d = float(line[8:].split(",")[0])
        elif line and not line.startswith("#"):
            segs.append((t, t + d, urljoin(m3u8, line)))
            t += d
    return init, segs, t


def fetch_section(m3u8, a, b, out):
    """Baja solo las piezas que cubren [a, b] y devuelve el segundo del directo en que empieza el archivo."""
    init, segs, _ = hls_index(m3u8)
    pick = [s for s in segs if s[1] > a and s[0] < b]
    if not pick:
        raise gr.Error("Ese tramo del directo aún no está disponible.")
    urls = ([init] if init else []) + [u for _, _, u in pick]
    with ThreadPoolExecutor(8) as ex, open(out, "wb") as f:  # 8 piezas a la vez, escritas en orden
        for r in ex.map(http_get, urls):
            f.write(r.content)
    return pick[0][0]


def live_vod(channel):
    """Vídeo que Twitch está grabando del directo en curso del canal."""
    m = re.search(r"twitch\.tv/([A-Za-z0-9_]+)", channel or "")
    if not m or m.group(1).lower() == "videos":
        raise gr.Error("Para el modo directo pega el enlace del canal, por ejemplo https://www.twitch.tv/illojuan")
    r = subprocess.run(["yt-dlp", "--flat-playlist", "--playlist-end", "1", "--no-warnings", "--print", "url",
                        "--print", "title", f"https://www.twitch.tv/{m.group(1)}/videos?filter=archives&sort=time"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    lines = r.stdout.strip().splitlines()
    if len(lines) < 2:
        raise gr.Error("No encuentro el vídeo del directo en Twitch (¿tiene los VODs desactivados?).")
    return f"https://www.twitch.tv/{m.group(1)}", lines[0], lines[1]


def hls_url(vod, fmt):
    """Lista de piezas del vídeo del directo (se pide cada vez: el enlace caduca)."""
    r = subprocess.run(["yt-dlp", "-f", fmt, "-S", "res:1080,fps,vcodec:h264", "-g", "--no-warnings", vod],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    urls = [x for x in r.stdout.split() if x.startswith("http")]
    if not urls:
        raise gr.Error("No se pudo leer el directo: " + r.stderr[-200:])
    return urls[0]


def live_analyze(vod, a, b, work, kind_n, min_s, max_s, avoid_music, progress, avoid=()):
    """Transcribe y elige momentos del tramo [a, b] del directo."""
    sec = f"{work}/win_{int(a)}.mp4"
    off = fetch_section(hls_url(vod, "ba/b"), a, b, sec)
    mp3 = f"{work}/win_{int(a)}.mp3"
    run(["ffmpeg", "-y", "-v", "error", "-i", sec, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", mp3])
    j = groq_chunk(mp3)
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in j.get("segments") or []]
    words = [{"start": w["start"], "end": w["end"], "word": w["word"].strip()}
             for w in j.get("words") or [] if w.get("word", "").strip()] or words_from_segs(segs)
    db = loudness(mp3)
    music = music_ranges(segs, words, db, False)
    shift = lambda xs: [{**x, "start": x["start"] + off, "end": x["end"] + off} for x in xs]
    src = dict(dur=b, segs=[s for s in shift(segs) if s["start"] >= a], words=shift(words), chat=[], kind="directo",
               loud=[t + off for t in loud_peaks(db) if t + off >= a], music=[[x + off, y + off] for x, y in music])
    src["avoid"] = list(avoid)  # títulos de clips ya hechos de este directo
    return src, (pick_highlights(src, kind_n, min_s, max_s, progress, avoid_music) if src["segs"] else [])


def live_render(vod, clips, src, work, first, layout, cam_pos, hook, dest, cam_box):
    """Crea los clips del tramo bajando solo sus piezas de vídeo. Devuelve [(archivo, clip)] y la webcam encontrada."""
    m3u8 = hls_url(vod, "bv*+ba/b")
    made = []
    for i, c in enumerate(clips, first):
        f = f"{work}/clip_{i}.mp4"
        off = fetch_section(m3u8, float(c["start"]) - 2, float(c["end"]) + 2, f)
        w, h, fps, _ = probe(f)  # la duración que da es «hasta dónde llega en el directo», no lo que dura el trozo
        length = float(c["end"]) + 2 - off  # lo que dura el trozo de verdad: así la webcam se busca dentro de él
        clip_src = dict(src, work=work, video=f, offset=off, audio_in=None, w=w, h=h, fps=fps, dur=length)
        if layout == LAYOUTS[1] and not cam_box[0]:  # hasta encontrarla: en un clip puede no mirar a cámara
            cam_box[0] = find_cam(clip_src, cam_pos)
        out = render(clip_src, c, i, layout, cam_box[0] or None, hook)
        if dest:
            shutil.copy(out, dest)
            shutil.copy(out[:-4] + ".txt", dest)
        made.append((out, c))
    return made


def limit_hit(msg):
    """¿Error de límite de la plataforma (no del clip)? Entonces no tiene sentido seguir subiendo hoy ahí."""
    m = msg.lower()
    return any(k in m for k in ("quota", "exceeded", "uploadlimit", "spam_risk", "no está conectado", "caducó"))


def live_publish(f, c, platforms, privacy, blocked):
    """Sube un clip a las plataformas elegidas. Si YouTube se queda sin cuota del día, deja de intentarlo."""
    notes = []
    for p in platforms:
        if p in blocked:
            continue
        try:
            res = yt_upload(f, c, PRIVACY[privacy]) if p == PLATFORMS[0] else tt_upload(f)
            notes.append(f"{p}: {res}")
        except gr.Error as e:
            msg = str(getattr(e, "message", e))
            notes.append(f"{p}: error ({msg[:80]})")
            if limit_hit(msg):
                blocked.add(p)
    return notes


def on_live(channel, per_window, min_s, max_s, avoid_music, layout, cam_pos, hook, to_drive, folder, from_start,
            auto_pub, platforms, privacy, progress=gr.Progress()):
    """Saca clips de un directo mientras se emite. Es un generador: la web se actualiza con cada tanda de clips."""
    if not GROQ_KEY or not GEMINI_KEY:
        raise gr.Error("Faltan GROQ_API_KEY o GEMINI_API_KEY en los Secretos.")
    chan, vod, title = live_vod(channel)
    if not is_live(chan):
        raise gr.Error("Ese canal no está en directo ahora. Para un directo ya terminado usa el modo normal.")
    work = tempfile.mkdtemp()
    dest = drive_dest({"title": f"{title} (directo)"}, to_drive, folder)
    made, log, blocked, cam_box, seen = [], [], set(), [None], []  # seen: tramos ya clipeados de este vídeo
    _, _, length = hls_index(hls_url(vod, "ba/b"))
    pos = 0.0 if from_start else max(0.0, length - LIVE_WINDOW)

    def update(status):
        rows = "".join(f"<li>{x}</li>" for x in log[-12:][::-1])
        html = (f'<div class="note"><b>En directo:</b> {title}<br>{status}<br>Clips hechos: <b>{len(made)}</b> · '
                f'Analizado hasta el minuto <b>{int(pos // 60)}</b></div><ul class="hint">{rows}</ul>')
        labels = [clip_label(f, c) for f, c in made]
        return (gr.update(value=[(f, f"{c.get('score')} · {c.get('title')}") for f, c in made[-MAX_SHOWN:]],
                          visible=True), gr.update(), html, made, gr.update(choices=labels, value=labels), "")

    yield update("Empezando…")
    while True:
        now_vod = live_vod(chan)[1] if is_live(chan) else None
        live = now_vod == vod  # si el streamer reinicia el directo, Twitch abre otro vídeo: se acaba este y se pasa al nuevo
        _, _, length = hls_index(hls_url(vod, "ba/b"))
        ready = length - LIVE_LAG if live else length
        size = LIVE_BACKLOG if ready - pos > LIVE_BACKLOG else LIVE_WINDOW
        if ready - pos >= (size if live else 30):
            b = min(ready, pos + size)
            yield update(f"Analizando del minuto {int(pos // 60)} al {int(b // 60)}…")
            try:
                n = max(1, round(int(per_window or 3) * (b - pos) / LIVE_WINDOW))
                src, clips = live_analyze(vod, pos, b, work, n, min_s, max_s, avoid_music, progress)
                clips = [c for c in clips if not any(min(e, c["end"]) - max(s, c["start"]) > 5 for s, e in seen)]
                seen += [(float(c["start"]), float(c["end"])) for c in clips]
                yield update(f"Creando {len(clips)} clips del minuto {int(pos // 60)} al {int(b // 60)}…")
                new = live_render(vod, clips, src, work, len(made) + 1, layout, cam_pos, hook, dest, cam_box)
                for f, c in new:
                    made.append((f, c))
                    notes = live_publish(f, c, platforms, privacy, blocked) if auto_pub else []
                    log.append(f"{time.strftime('%H:%M')} · {c.get('title')}" + (" · " + " · ".join(notes) if notes else ""))
                    yield update(f"Clip listo: {c.get('title')}")
                pos = b
            except NoQuota as e:
                log.append(f"{time.strftime('%H:%M')} · Sin cuota de IA: {getattr(e, 'message', e)}")
                yield update("Sin cuota de IA por hoy: se para. Lo hecho está guardado.")
                return
            except gr.Error as e:  # un tramo que falla no para el directo: se reintenta en la siguiente vuelta
                log.append(f"{time.strftime('%H:%M')} · Error en el tramo: {str(getattr(e, 'message', e))[:120]}")
                yield update("Hubo un error en este tramo; lo reintento en un minuto…")
                time.sleep(60)
            continue
        if not live:
            if now_vod:  # directo reiniciado: se sigue con el vídeo nuevo desde su principio
                vod, pos, seen = now_vod, 0.0, []
                log.append(f"{time.strftime('%H:%M')} · El directo se ha reiniciado; sigo con el nuevo.")
                continue
            yield update(f"El directo ha terminado. {len(made)} clips en total.")
            return
        wait = int(pos + LIVE_WINDOW + LIVE_LAG - length)
        yield update(f"Esperando a que avance el directo (próxima tanda en ~{max(1, wait // 60)} min)…")
        time.sleep(min(max(wait, 30), 300))


# ───────────────────────── Interfaz ─────────────────────────

ACCENT = "#f5c518"  # mismo amarillo que los subtítulos karaoke
STEPS = ["Vídeo", "Ajustes", "Revisar", "Clips", "Publicar"]

THEME = gr.themes.Base(
    primary_hue="yellow", secondary_hue="zinc", neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"], radius_size="md",
).set(
    body_background_fill="#0e0e10", body_background_fill_dark="#0e0e10",
    body_text_color="#ececee", body_text_color_dark="#ececee",
    body_text_color_subdued="#8b8b93", body_text_color_subdued_dark="#8b8b93",
    background_fill_primary="#161618", background_fill_primary_dark="#161618",
    background_fill_secondary="#121214", background_fill_secondary_dark="#121214",
    block_background_fill="#161618", block_background_fill_dark="#161618",
    block_border_color="#26262a", block_border_color_dark="#26262a",
    block_label_text_color="#a1a1aa", block_label_text_color_dark="#a1a1aa",
    block_title_text_color="#ececee", block_title_text_color_dark="#ececee",
    input_background_fill="#1c1c1f", input_background_fill_dark="#1c1c1f",
    input_border_color="#2e2e33", input_border_color_dark="#2e2e33",
    border_color_primary="#2e2e33", border_color_primary_dark="#2e2e33",
    button_primary_background_fill=ACCENT, button_primary_background_fill_dark=ACCENT,
    button_primary_background_fill_hover="#ffd84d", button_primary_background_fill_hover_dark="#ffd84d",
    button_primary_text_color="#111", button_primary_text_color_dark="#111",
    button_secondary_background_fill="#232326", button_secondary_background_fill_dark="#232326",
    button_secondary_background_fill_hover="#2c2c30", button_secondary_background_fill_hover_dark="#2c2c30",
    button_secondary_text_color="#ececee", button_secondary_text_color_dark="#ececee",
    slider_color=ACCENT, slider_color_dark=ACCENT,
    checkbox_background_color_selected=ACCENT, checkbox_background_color_selected_dark=ACCENT,
    checkbox_border_color_selected=ACCENT, checkbox_border_color_selected_dark=ACCENT,
    shadow_drop="none", shadow_drop_lg="none", block_shadow="none", block_shadow_dark="none",
)

CSS = """
.gradio-container {max-width: 980px !important; margin: auto !important;}
footer {display:none !important;}
/* barra superior */
#top {align-items:center !important; border-bottom:1px solid #222226; padding:10px 0 12px !important; margin-bottom:4px;}
#top .brand {font-size:1.25rem; font-weight:800; color:#fff;} #top .brand span {color:%(a)s;}
#top .status {display:flex; gap:16px; flex-wrap:wrap; justify-content:flex-end; font-size:.84rem; color:#a1a1aa;}
.dot {display:inline-block; width:8px; height:8px; border-radius:50%%; background:#6b6b73; margin-right:6px; vertical-align:1px;}
.dot.on {background:#22c55e;} .dot.warn {background:%(a)s;}
.status small {color:#6b6b73; margin-left:4px;}
/* barra de pasos */
.stepper {display:flex; gap:0; margin:14px 0 6px; counter-reset:s;}
.stepper .st {flex:1; display:flex; flex-direction:column; gap:8px; font-size:.82rem; color:#6b6b73; font-weight:600;}
.stepper .bar {height:4px; border-radius:2px; background:#26262a; margin-right:6px;}
.stepper .st.done {color:#a1a1aa;} .stepper .st.done .bar {background:#6b6b73;}
.stepper .st.now {color:#fff;} .stepper .st.now .bar {background:%(a)s;}
/* cabecera de cada paso */
.head h2 {font-size:1.6rem; font-weight:800; color:#fff; margin:18px 0 4px; letter-spacing:-.01em;}
.head p {color:#8b8b93; margin:0 0 16px; font-size:.95rem;}
.label {font-size:.76rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:#8b8b93; margin:2px 0 8px;}
.hint {color:#8b8b93; font-size:.85rem; margin:2px 0 6px;}
#urls textarea {font-size:1.02rem !important; padding:14px !important; background:#121214 !important;}
.group {background:#161618 !important; border:1px solid #26262a !important; border-radius:10px !important; padding:16px !important;}
/* barra de navegación: atrás a la izquierda, acción principal a la derecha */
.nav {margin-top:18px !important; padding-top:16px !important; border-top:1px solid #222226; align-items:center !important;}
.nav button {min-height:48px; font-weight:700 !important; font-size:.98rem !important;}
.or {color:#6b6b73; text-align:center; font-size:.85rem;}
#layout .wrap {display:grid !important; grid-template-columns:repeat(3,1fr); gap:8px;}
#layout label {justify-content:center; text-align:center; padding:12px 6px !important; border-radius:8px !important;
  border:1px solid #2e2e33 !important; background:#1c1c1f !important; font-weight:600;}
#layout label:has(input:checked), #layout label.selected {border-color:%(a)s !important; color:#fff !important;
  background:rgba(245,197,24,.08) !important;}
#layout input {display:none;}
/* cifras */
.stats {display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); border:1px solid #26262a;
  border-radius:10px; overflow:hidden; margin:0 0 14px;}
.stat {padding:12px 14px; border-right:1px solid #26262a; display:flex; flex-direction:column; gap:2px;}
.stat:last-child {border-right:none;}
.stat small {color:#8b8b93; font-size:.76rem;} .stat b {color:#fff; font-size:1.2rem; font-weight:700;}
#review-table, #review-table * {font-size:.9rem; font-family: Inter, system-ui, sans-serif !important;}
/* galería vertical */
#gallery .grid-wrap {height:auto !important; max-height:none !important; overflow:visible !important;}
#gallery .grid-container {grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)) !important; height:auto !important;}
#gallery .thumbnail-item {aspect-ratio: 9 / 16 !important; height:auto !important; border-radius:8px !important;}
#gallery video, #gallery img {object-fit:cover !important; width:100%%; height:100%%;}
/* publicar y cuentas */
.note {border-left:3px solid %(a)s; background:#161618; padding:10px 14px; color:#c4c4cc; font-size:.88rem; margin:0 0 14px;}
.accs {display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin:0 0 12px;}
.acc {display:flex; justify-content:space-between; align-items:center; background:#161618; border:1px solid #26262a;
  border-radius:10px; padding:14px 16px;}
.acc b {color:#fff;} .acc-state {color:#a1a1aa; font-size:.88rem;}
.code-box {display:flex; flex-direction:column; gap:6px; background:#121214; border:1px solid #2e2e33; border-radius:10px;
  padding:14px; margin:8px 0; color:#c4c4cc;}
.code-box b {font-size:1.7rem; letter-spacing:.15em; color:#fff;} .code-box a {color:%(a)s; font-weight:600;}
.pub-log {width:100%%; border-collapse:collapse; margin-top:12px; font-size:.9rem;}
.pub-log td, .pub-log th {border-bottom:1px solid #26262a; padding:9px 8px; text-align:left;}
.pub-log th {color:#8b8b93; font-weight:600;} .pub-log a {color:%(a)s;}
.pub-log .ok {color:#22c55e; font-weight:600;} .pub-log .err {color:#f87171; font-weight:600;}
.guide {color:#c4c4cc; font-size:.9rem; line-height:1.6;} .guide ol {margin:4px 0 12px 18px;} .guide code {color:%(a)s;}
#accounts {border:1px solid #2e2e33 !important; border-radius:12px !important; padding:6px 18px 18px !important;
  background:#121214 !important; margin-top:10px;}
@media (max-width: 700px) {#layout .wrap {grid-template-columns:1fr;} .stepper .st span {display:none;}}
""" % {"a": ACCENT}


def status_html():
    gpu = ('<span><span class="dot on"></span>GPU activa</span>' if HAS_GPU else
           '<span><span class="dot warn"></span>Sin GPU<small>(más lento)</small></span>')
    drive = ('<span><span class="dot on"></span>Drive conectado</span>' if os.path.isdir(DRIVE) else
             '<span><span class="dot warn"></span>Drive sin conectar</span>')
    return f'<div class="status">{gpu}{drive}</div>'


def stepper_html(now):
    cells = "".join(f'<div class="st {"done" if i < now else "now" if i == now else ""}"><div class="bar"></div>'
                    f'<span>{i}. {name}</span></div>' for i, name in enumerate(STEPS, 1))
    return f'<div class="stepper">{cells}</div>'


def head(title, sub):
    return gr.HTML(f'<div class="head"><h2>{title}</h2><p>{sub}</p></div>')


GUIDE = """<div class="guide">
<b>YouTube</b> (una vez, unos 10 minutos)
<ol><li>En <code>console.cloud.google.com</code> crea un proyecto y activa <b>YouTube Data API v3</b>.</li>
<li>Pantalla de consentimiento OAuth: tipo <b>Externo</b> y añádete como <b>usuario de prueba</b>.</li>
<li>Credenciales › Crear ID de cliente OAuth › <b>TV y dispositivos de entrada limitada</b>.</li>
<li>Guarda en los Secretos de Colab <code>YT_CLIENT_ID</code> y <code>YT_CLIENT_SECRET</code>.</li></ol>
<b>TikTok</b>
<ol><li>En <code>developers.tiktok.com</code> crea una app con <b>Login Kit</b> y <b>Content Posting API</b>.</li>
<li>Redirect URI: una web con https, por ejemplo <code>https://www.google.com/</code>.</li>
<li>Añade tu cuenta como usuario de prueba (Sandbox).</li>
<li>Guarda en los Secretos de Colab <code>TIKTOK_CLIENT_KEY</code>, <code>TIKTOK_CLIENT_SECRET</code> y <code>TIKTOK_REDIRECT_URI</code>.</li></ol>
</div>"""

with gr.Blocks(title="ClipForge") as demo:
    src_state, made_state = gr.State(None), gr.State([])

    with gr.Row(elem_id="top"):
        gr.HTML('<div class="brand">Clip<span>Forge</span></div>')
        gr.HTML(status_html())
        acc_open = gr.Button("Cuentas", size="sm", scale=0, min_width=100)

    # ── Cuentas (panel que se abre desde arriba) ──
    with gr.Column(visible=False, elem_id="accounts") as accounts:
        head("Cuentas", "Conecta YouTube y TikTok una sola vez; la conexión se guarda en tu Drive.")
        acc_status = gr.HTML(accounts_html())
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="group"):
                gr.HTML('<div class="label">YouTube</div>')
                with gr.Row():
                    yt_btn = gr.Button("Conectar YouTube", variant="primary")
                    yt_done = gr.Button("Ya lo autoricé")
                yt_code = gr.HTML()
                yt_out = gr.Button("Desconectar", size="sm")
            with gr.Column(elem_classes="group"):
                gr.HTML('<div class="label">TikTok</div>')
                tt_btn = gr.Button("Conectar TikTok", variant="primary")
                tt_link = gr.HTML()
                tt_paste = gr.Textbox(label="Dirección a la que te llevó TikTok", placeholder="https://…?code=…")
                with gr.Row():
                    tt_ok = gr.Button("Guardar conexión")
                    tt_out = gr.Button("Desconectar", size="sm")
        with gr.Accordion("Cómo preparar YouTube y TikTok (solo la primera vez)", open=False):
            gr.HTML(GUIDE)
        acc_close = gr.Button("Cerrar")

    stepper = gr.HTML(stepper_html(1))

    # ── Paso 1: vídeo ──
    with gr.Column(visible=True) as s1:
        head("¿De qué vídeo sacamos clips?",
             "Directos de Twitch o YouTube, o vídeos normales de YouTube. Detecto el tipo y busco los momentos "
             "de la forma adecuada.")
        with gr.Tabs():
            with gr.Tab("Pegar enlace"):
                urls = gr.Textbox(show_label=False, lines=3, max_lines=12, elem_id="urls",
                                  placeholder="https://www.twitch.tv/videos/123456789\nhttps://www.youtube.com/watch?v=…")
                gr.HTML('<div class="hint">Puedes pegar varios, uno por línea. ¿Directo en emisión ahora mismo? '
                        'Pega el canal (por ejemplo https://www.twitch.tv/illojuan) y en el paso 2 pulsa '
                        '«Clipear el directo en vivo».</div>')
            with gr.Tab("Subir archivo"):
                # gr.File y no gr.Video: Gradio no reconvierte el archivo al subirlo (ahorra minutos en vídeos largos)
                vid = gr.File(show_label=False, file_types=["video"], type="filepath", height=130)
        with gr.Row(elem_classes="nav"):
            gr.HTML("")
            next1 = gr.Button("Siguiente: ajustes", variant="primary", scale=0, min_width=220)

    # ── Paso 2: ajustes y cómo hacerlo ──
    with gr.Column(visible=False) as s2:
        head("Ajustes", "Revisa cómo quieres los clips y elige cómo seguir.")
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="group"):
                gr.HTML('<div class="label">Momentos</div>')
                n = gr.Number(10, label="Clips por vídeo", minimum=1, maximum=5000, precision=0)
                with gr.Row():
                    min_s = gr.Slider(10, 60, 20, step=5, label="Mínimo (s)")
                    max_s = gr.Slider(20, 90, 55, step=5, label="Máximo (s)")
                gr.HTML('<div class="hint">En vídeos editados de YouTube uso menos clips y más cortos (15–40 s).</div>')
            with gr.Column(elem_classes="group"):
                gr.HTML('<div class="label">Encuadre</div>')
                layout = gr.Radio(LAYOUTS, value=LAYOUTS[0], show_label=False, elem_id="layout")
                gr.HTML('<div class="hint">Seguir cara: el recorte sigue a quien habla. Streamer: webcam arriba y '
                        'juego abajo. Centro: recorte fijo.</div>')
                cam_pos = gr.Dropdown(CAM_POS, value=CAM_POS[0], label="Webcam (solo modo Streamer)")
        with gr.Row(equal_height=True):
            with gr.Column(elem_classes="group"):
                gr.HTML('<div class="label">Contenido</div>')
                hook = gr.Checkbox(True, label="Título del clip en los 3 primeros segundos")
                avoid_music = gr.Checkbox(True, label="Evitar tramos con música (menos reclamaciones)")
                gr.HTML('<div class="hint">Los subtítulos karaoke van siempre.</div>')
            with gr.Column(elem_classes="group"):
                gr.HTML('<div class="label">Guardar</div>')
                to_drive = gr.Checkbox(True, label="Guardar en Google Drive")
                folder = gr.Textbox("ClipForge", label="Carpeta", info="Dentro, una subcarpeta por vídeo")
        with gr.Accordion("En directo (solo si pegaste el canal de un directo en emisión)", open=False):
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="group"):
                    live_n = gr.Number(3, label="Clips por cada 30 min de directo", minimum=1, maximum=20, precision=0)
                    from_start = gr.Checkbox(True, label="Empezar desde el principio del directo (si no, desde ahora)")
                with gr.Column(elem_classes="group"):
                    auto_pub = gr.Checkbox(False, label="Subir cada clip automáticamente")
                    live_platforms = gr.CheckboxGroup(PLATFORMS, value=PLATFORMS, label="Dónde")
                    live_privacy = gr.Radio(list(PRIVACY), value="Público", label="Visibilidad en YouTube")
        with gr.Row(elem_classes="nav"):
            back2 = gr.Button("Atrás", scale=0, min_width=110)
            live_btn = gr.Button("Clipear el directo en vivo", scale=1)
            auto_btn = gr.Button("Hacerlo todo automático", scale=1)
            analyze_btn = gr.Button("Buscar momentos y revisarlos", variant="primary", scale=1)

    # ── Paso 3: revisar ──
    with gr.Column(visible=False) as s3:
        head("Revisa los momentos", "Desmarca los que no quieras. Haz clic en una celda para cambiar tiempos, "
                                    "título, descripción o hashtags.")
        review_stats = gr.HTML()
        table = gr.Dataframe(headers=COLUMNS, datatype=["bool", "number", "str", "number", "number", "str", "str"],
                             interactive=True, wrap=True, max_height=480, elem_id="review-table")
        with gr.Row(elem_classes="nav"):
            back3 = gr.Button("Atrás", scale=0, min_width=110)
            gr.HTML("")
            export_btn = gr.Button("Crear los clips marcados", variant="primary", scale=0, min_width=240)

    # ── Paso 4: clips ──
    with gr.Column(visible=False) as s4:
        head("Tus clips", "Haz clic en uno para verlo. Cada clip va con un .txt con título, descripción y hashtags.")
        clip_stats = gr.HTML()
        gallery = gr.Gallery(show_label=False, columns=5, height="auto", object_fit="cover", preview=False,
                             elem_id="gallery")
        out = gr.File(label="Descargar todo (.zip)")
        with gr.Row(elem_classes="nav"):
            restart = gr.Button("Empezar con otro vídeo", scale=0, min_width=220)
            stop_live = gr.Button("Parar el directo", variant="stop", scale=0, min_width=180)
            gr.HTML("")
            next4 = gr.Button("Siguiente: publicar", variant="primary", scale=0, min_width=220)

    # ── Paso 5: publicar ──
    with gr.Column(visible=False) as s5:
        head("Publicar", "Sube los clips a tus cuentas.")
        gr.HTML('<div class="note">YouTube deja en <b>privado</b> lo que suben apps sin verificar: cámbialo a público '
                'en YouTube Studio. El plan gratis permite unas 6 subidas al día. TikTok los deja en tu '
                '<b>bandeja de borradores</b>: ábrela en el móvil para publicarlos.</div>')
        pub_accs = gr.HTML(accounts_html())
        picked = gr.CheckboxGroup([], label="Clips")
        with gr.Row():
            platforms = gr.CheckboxGroup(PLATFORMS, value=[PLATFORMS[0]], label="Dónde")
            privacy = gr.Radio(list(PRIVACY), value="Público", label="Visibilidad en YouTube")
        pub_log = gr.HTML()
        with gr.Row(elem_classes="nav"):
            back5 = gr.Button("Atrás", scale=0, min_width=110)
            acc_open2 = gr.Button("Conectar cuentas", scale=0, min_width=180)
            gr.HTML("")
            publish_btn = gr.Button("Publicar", variant="primary", scale=0, min_width=180)

    # ── Navegación ──
    panels = [s1, s2, s3, s4, s5]
    nav_out = panels + [stepper]

    def go(i):
        return [gr.update(visible=k == i) for k in range(1, 6)] + [stepper_html(i)]

    def to(i):
        return lambda: go(i)

    next1.click(check_sources, [vid, urls], None).success(to(2), None, nav_out)
    back2.click(to(1), None, nav_out)
    back3.click(to(2), None, nav_out)
    next4.click(to(5), None, nav_out).then(accounts_html, None, pub_accs)
    back5.click(to(4), None, nav_out)
    restart.click(lambda: (None, ""), None, [vid, urls]).then(to(1), None, nav_out)
    show_acc = lambda: gr.update(visible=True)
    acc_open.click(show_acc, None, accounts)
    acc_open2.click(show_acc, None, accounts)
    acc_close.click(lambda: gr.update(visible=False), None, accounts).then(accounts_html, None, pub_accs)

    # ── Acciones ──
    made_out = [gallery, out, clip_stats, made_state, picked, pub_log]
    # Primero se cambia de paso y luego se trabaja: así la barra de progreso se ve en el paso nuevo.
    # Si algo falla, sale el aviso en rojo y el botón «Atrás» devuelve al paso anterior.
    analyze_btn.click(to(3), None, nav_out) \
        .then(on_analyze, [vid, urls, n, min_s, max_s, avoid_music], [src_state, table, review_stats])
    export_btn.click(to(4), None, nav_out) \
        .then(on_export, [src_state, table, layout, cam_pos, hook, to_drive, folder], made_out)
    auto_btn.click(to(4), None, nav_out) \
        .then(on_auto, [vid, urls, n, min_s, max_s, avoid_music, layout, cam_pos, hook, to_drive, folder], made_out)
    live_job = live_btn.click(to(4), None, nav_out).then(
        on_live, [urls, live_n, min_s, max_s, avoid_music, layout, cam_pos, hook, to_drive, folder, from_start,
                  auto_pub, live_platforms, live_privacy], made_out)
    stop_live.click(None, None, None, cancels=[live_job])
    publish_btn.click(on_publish, [made_state, picked, platforms, privacy], pub_log)
    yt_btn.click(yt_start, None, yt_code)
    yt_done.click(yt_finish, None, acc_status).success(lambda: "", None, yt_code)
    yt_out.click(lambda: disconnect("youtube"), None, acc_status)
    tt_btn.click(tt_login_url, None, tt_link)
    tt_ok.click(tt_connect, tt_paste, [acc_status, tt_paste])
    tt_out.click(lambda: disconnect("tiktok"), None, acc_status)
    demo.load(accounts_html, None, acc_status)

if __name__ == "__main__":
    demo.queue().launch(share="COLAB_RELEASE_TAG" in os.environ, theme=THEME, css=CSS)  # en Colab da un enlace público
