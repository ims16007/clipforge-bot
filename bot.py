"""ClipForge en piloto automático (GitHub Actions, cada 30 min).

Para cada streamer de streamers.txt: si está en directo (o su último directo tiene un tramo sin revisar), analiza
los tramos nuevos de 30 min, crea SOLO los clips que el juez IA puntúa alto y los sube a YouTube y TikTok.
El estado (cuentas conectadas y por dónde va cada directo) se guarda cifrado en state.enc dentro del repositorio.
"""
import json, os, sys, tempfile, time
from cryptography.fernet import Fernet

import app

MIN_SCORE = 70           # solo se crean los momentos que el juez puntúa 70/100 o más
CANDIDATES = 6           # momentos que se proponen por cada 30 min (luego se filtran por nota)
MIN_S, MAX_S = 20, 45    # duración de los clips
LAYOUT = app.LAYOUTS[1]  # Streamer: webcam arriba, juego abajo (si no hay webcam, recorte centrado)
PLATFORMS = app.PLATFORMS
WORK_BUDGET = 40 * 60    # trabajo máximo por ejecución; lo que falte se hace en la siguiente
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.enc")
OUT = os.environ.get("CLIPS_DIR", "clips")


def mask(value):
    """Oculta un valor en los registros públicos de GitHub."""
    if value and os.environ.get("GITHUB_ACTIONS"):
        print(f"::add-mask::{value}")


def load_state(fernet):
    if os.path.exists(STATE):
        with open(STATE, "rb") as f:
            return json.loads(fernet.decrypt(f.read()))
    # primera vez: las cuentas vienen del secreto ACCOUNTS_JSON (el archivo .clipforge_cuentas.json de tu Drive)
    return {"accounts": json.loads(os.environ.get("ACCOUNTS_JSON") or "{}"), "vods": {}}


def save_state(fernet, state):
    tmp = STATE + ".tmp"
    with open(tmp, "wb") as f:
        f.write(fernet.encrypt(json.dumps(state).encode()))
    os.replace(tmp, STATE)


def streamers():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamers.txt")
    with open(path, encoding="utf-8") as f:
        return [l.strip().lstrip("@").lower() for l in f if l.strip() and not l.startswith("#")]


def good(clips):
    return [c for c in clips if float(c.get("score") or 0) >= MIN_SCORE]


def process(name, state, deadline, blocked):
    """Revisa los tramos pendientes del directo de un streamer. Devuelve cuántos clips ha subido."""
    chan = f"https://www.twitch.tv/{name}"
    live = app.is_live(chan)
    try:
        _, vod, title = app.live_vod(chan)
    except app.gr.Error:
        return 0
    info = state["vods"].setdefault(vod, {"pos": 0.0, "done": [], "title": title, "seen": time.time()})
    if not live and info.get("finished"):
        return 0
    _, _, length = app.hls_index(app.hls_url(vod, "ba/b"))
    ready = length - app.LIVE_LAG if live else length
    made = 0
    work = tempfile.mkdtemp()
    while time.time() < deadline:
        pos = info["pos"]
        size = app.LIVE_BACKLOG if ready - pos > app.LIVE_BACKLOG else app.LIVE_WINDOW
        if ready - pos < (size if live else 30):
            break
        b = min(ready, pos + size)
        print(f"[{name}] analizando minuto {int(pos // 60)}–{int(b // 60)} de «{title}»")
        n = max(1, round(CANDIDATES * (b - pos) / app.LIVE_WINDOW))
        src, clips = app.live_analyze(vod, pos, b, work, n, MIN_S, MAX_S, True, lambda *a, **k: None)
        clips = [c for c in good(clips)
                 if not any(min(e, c["end"]) - max(s, c["start"]) > 5 for s, e in info["done"])]
        print(f"[{name}] {len(clips)} momentos con nota ≥ {MIN_SCORE}")
        for f, c in app.live_render(vod, clips, src, work, len(info["done"]) + 1, LAYOUT, app.CAM_POS[0], True,
                                    None, [None]):
            notes = app.live_publish(f, c, PLATFORMS, "Público", blocked)
            print(f"[{name}] «{c.get('title')}» (nota {c.get('score')}): " + " · ".join(notes))
            os.makedirs(OUT, exist_ok=True)
            os.replace(f, os.path.join(OUT, f"{name}_{os.path.basename(f)}"))
            made += 1
        info["done"] += [[float(c["start"]), float(c["end"])] for c in clips]
        info["pos"] = b
    if not live and ready - info["pos"] < 30:
        info["finished"] = True
    return made


def main():
    key = os.environ.get("STATE_KEY")
    if not key:
        sys.exit("Falta el secreto STATE_KEY.")
    fernet = Fernet(key.encode())
    state = load_state(fernet)
    for acc in state["accounts"].values():
        for k in ("access_token", "refresh_token"):
            mask(acc.get(k))
    # las funciones de subida de app.py leen y renuevan las cuentas en este archivo
    with open(app.accounts_path(), "w", encoding="utf-8") as f:
        json.dump(state["accounts"], f)
    deadline, blocked, total = time.time() + WORK_BUDGET, set(), 0
    try:
        for name in streamers():
            try:
                total += process(name, state, deadline, blocked)
            except app.NoQuota as e:
                print(f"Sin cuota de IA por hoy: {getattr(e, 'message', e)}")
                break
            except app.gr.Error as e:  # un streamer que falla no para a los demás
                print(f"[{name}] error: {getattr(e, 'message', e)}")
    finally:
        state["accounts"] = app.load_accounts()  # tokens renovados
        cutoff = time.time() - 7 * 86400  # se olvidan los directos de hace más de una semana
        state["vods"] = {k: v for k, v in state["vods"].items() if v.get("seen", 0) > cutoff}
        save_state(fernet, state)
    print(f"Listo: {total} clips subidos en esta vuelta.")


if __name__ == "__main__":
    main()
