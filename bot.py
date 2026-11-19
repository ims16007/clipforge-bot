"""ClipForge en piloto automático (GitHub Actions, cada 30 min).

1) Para cada streamer de streamers.txt: si está en directo (o a su último directo le falta un tramo por revisar),
   analiza los tramos nuevos y apunta en una COLA solo los momentos que el juez IA puntúa alto.
2) Sube la cola a YouTube y TikTok de mejor a peor nota. Lo que no cabe hoy (límites diarios de las plataformas)
   se queda en la cola y se sube en las siguientes vueltas.
El estado (cuentas, por dónde va cada directo y la cola) se guarda cifrado en state.enc dentro del repositorio.
"""
import json, os, sys, tempfile, time
from cryptography.fernet import Fernet

import app

MIN_SCORE = 70           # solo entran en la cola los momentos con nota 70/100 o más
CANDIDATES = 6           # momentos que se proponen por cada 30 min (luego se filtran por nota)
MIN_S, MAX_S = 20, 45    # duración de los clips
LAYOUT = app.LAYOUTS[1]  # Streamer: webcam arriba, juego abajo (si no hay webcam, recorte centrado)
PLATFORMS = app.PLATFORMS
QUEUE_HOURS = 48         # un momento que no se ha podido subir en 2 días ya no es noticia: se descarta
WORK_BUDGET = 40 * 60    # trabajo máximo por vuelta; lo que falte se hace en la siguiente
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.enc")
OUT = os.environ.get("CLIPS_DIR", "clips")


def mask(value):
    """Oculta un valor en los registros públicos de GitHub."""
    if value and os.environ.get("GITHUB_ACTIONS"):
        print(f"::add-mask::{value}")


def load_state(fernet):
    if os.path.exists(STATE):
        with open(STATE, "rb") as f:
            state = json.loads(fernet.decrypt(f.read()))
    else:  # primera vez: las cuentas vienen del secreto ACCOUNTS_JSON (el archivo .clipforge_cuentas.json de tu Drive)
        state = {"accounts": json.loads(os.environ.get("ACCOUNTS_JSON") or "{}"), "vods": {}}
    state.setdefault("queue", [])
    state.setdefault("n", 0)
    return state


def save_state(fernet, state):
    tmp = STATE + ".tmp"
    with open(tmp, "wb") as f:
        f.write(fernet.encrypt(json.dumps(state).encode()))
    os.replace(tmp, STATE)


def streamers():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamers.txt")
    with open(path, encoding="utf-8") as f:
        return [l.strip().lstrip("@").lower() for l in f if l.strip() and not l.startswith("#")]


def analyze(name, state, deadline):
    """Revisa los tramos pendientes del directo de un streamer y mete en la cola los momentos buenos."""
    chan = f"https://www.twitch.tv/{name}"
    live = app.is_live(chan)
    try:
        _, vod, title = app.live_vod(chan)
    except app.gr.Error:
        return 0
    info = state["vods"].setdefault(vod, {"pos": 0.0, "done": [], "titles": [], "title": title})
    info.setdefault("titles", [])  # directos guardados por la versión anterior del bot
    info["seen"] = time.time()
    if not live and info.get("finished"):
        return 0
    _, _, length = app.hls_index(app.hls_url(vod, "ba/b"))
    ready = length - app.LIVE_LAG if live else length
    added, work = 0, tempfile.mkdtemp()
    while time.time() < deadline:
        pos = info["pos"]
        size = app.LIVE_BACKLOG if ready - pos > app.LIVE_BACKLOG else app.LIVE_WINDOW
        if ready - pos < (size if live else 30):
            break
        b = min(ready, pos + size)
        print(f"[{name}] analizando minuto {int(pos // 60)}–{int(b // 60)} de «{title}»")
        n = max(1, round(CANDIDATES * (b - pos) / app.LIVE_WINDOW))
        src, clips = app.live_analyze(vod, pos, b, work, n, MIN_S, MAX_S, True, lambda *a, **k: None,
                                      avoid=info["titles"])
        clips = [c for c in clips if float(c.get("score") or 0) >= MIN_SCORE
                 and not any(min(e, c["end"]) - max(s, c["start"]) > 5 for s, e in info["done"])]
        for c in clips:
            s, e = float(c["start"]), float(c["end"])
            state["queue"].append({"vod": vod, "name": name, "added": time.time(), "todo": list(PLATFORMS),
                                   "clip": {k: c.get(k) for k in ("start", "end", "title", "description",
                                                                  "hashtags", "score")},
                                   "words": [w for w in src["words"] if w["end"] > s - 1 and w["start"] < e + 1]})
            info["done"].append([s, e])
            info["titles"].append(str(c.get("title") or ""))
            print(f"[{name}] a la cola: «{c.get('title')}» (nota {c.get('score')})")
        added += len(clips)
        info["pos"] = b
    if not live and ready - info["pos"] < 30:
        info["finished"] = True
    return added


def publish(state, deadline, blocked):
    """Sube la cola de mejor a peor nota. Devuelve cuántas subidas se han hecho."""
    cutoff = time.time() - QUEUE_HOURS * 3600
    state["queue"] = [q for q in state["queue"] if q["added"] > cutoff]
    state["queue"].sort(key=lambda q: -float(q["clip"].get("score") or 0))
    uploads, cams, work = 0, {}, tempfile.mkdtemp()
    for q in list(state["queue"]):
        todo = [p for p in q["todo"] if p not in blocked]
        if time.time() > deadline or len(blocked) == len(PLATFORMS):
            break
        if not todo:
            continue
        c = q["clip"]
        try:
            state["n"] += 1
            cam = cams.setdefault(q["vod"], [None])  # la webcam se busca una vez por directo
            [(f, _)] = app.live_render(q["vod"], [c], {"words": q["words"]}, work, state["n"], LAYOUT,
                                       app.CAM_POS[0], True, None, cam)
        except app.gr.Error as e:  # el vídeo ya no está o falla al crearlo: se quita de la cola
            print(f"[{q['name']}] no se pudo crear «{c.get('title')}»: {getattr(e, 'message', e)}")
            state["queue"].remove(q)
            continue
        for p in todo:
            try:
                res = app.yt_upload(f, c, "public") if p == PLATFORMS[0] else app.tt_upload(f)
                print(f"[{q['name']}] «{c.get('title')}» (nota {c.get('score')}) → {p}: {res}")
                q["todo"].remove(p)
                uploads += 1
            except app.gr.Error as e:
                msg = str(getattr(e, "message", e))
                if app.limit_hit(msg):  # límite del día: se intenta en otra vuelta
                    blocked.add(p)
                    print(f"{p}: límite alcanzado por ahora ({msg[:90]}); lo que falta queda en la cola.")
                else:  # error de este clip en concreto: no se reintenta
                    q["todo"].remove(p)
                    print(f"[{q['name']}] «{c.get('title')}» → {p}: error ({msg[:120]})")
        os.makedirs(OUT, exist_ok=True)
        os.replace(f, os.path.join(OUT, f"{q['name']}_{os.path.basename(f)}"))
        if not q["todo"]:
            state["queue"].remove(q)
    return uploads


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
    deadline, blocked, added, uploads = time.time() + WORK_BUDGET, set(), 0, 0
    try:
        for name in streamers():
            try:
                added += analyze(name, state, deadline)
            except app.NoQuota as e:
                print(f"Sin cuota de IA por hoy: {getattr(e, 'message', e)}")
                break
            except app.gr.Error as e:  # un streamer que falla no para a los demás
                print(f"[{name}] error: {getattr(e, 'message', e)}")
        uploads = publish(state, deadline + 15 * 60, blocked)
    finally:
        state["accounts"] = app.load_accounts()  # tokens renovados
        cutoff = time.time() - 7 * 86400  # se olvidan los directos de hace más de una semana
        state["vods"] = {k: v for k, v in state["vods"].items() if v.get("seen", 0) > cutoff}
        save_state(fernet, state)
    print(f"Listo: {added} momentos nuevos · {uploads} subidas · {len(state['queue'])} clips esperando en la cola.")


if __name__ == "__main__":
    main()
