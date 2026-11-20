"""ClipForge en piloto automático (GitHub Actions, cada 30 min).

1) Para cada streamer de streamers.txt: si está en directo (o a su último directo le falta un tramo por revisar),
   analiza los tramos nuevos y apunta en una COLA solo los momentos que el juez IA puntúa alto.
2) Sube la cola a YouTube y TikTok de mejor a peor nota. Lo que no cabe hoy (límites diarios de las plataformas)
   se queda en la cola y se sube en las siguientes vueltas.
El estado (cuentas, por dónde va cada directo y la cola) se guarda cifrado en state.enc dentro del repositorio.
"""
import hashlib, json, os, sys, tempfile, time
from cryptography.fernet import Fernet

import app

MIN_SCORE = 70           # solo entran en la cola los momentos con nota 70/100 o más
CANDIDATES = 6           # momentos que se proponen por cada 30 min (luego se filtran por nota)
KEEP_PER_30 = 3          # como mucho se crean los 3 mejores de cada 30 min de directo
AUTO_PER_30 = 1          # y solo el mejor de cada 30 min se sube solo (si además llega a AUTO_SCORE)
MIN_S, MAX_S = 20, 45    # duración de los clips
LAYOUT = app.LAYOUTS[1]  # Streamer: webcam arriba, juego abajo (si no hay webcam, recorte centrado)
PLATFORMS = app.PLATFORMS
QUEUE_HOURS = 48         # un momento que no se ha podido subir en 2 días ya no es noticia: se descarta
YT_SPACING = 2 * 3600    # en YouTube, publicaciones repartidas a lo largo del día (rinden más que todas de golpe)
TT_FOLDER = "ClipForge (TikTok)"  # TikTok no se sube solo: los buenos van a esta carpeta de Drive y los subes tú
WORK_BUDGET = 40 * 60    # trabajo máximo por tanda dentro de una vuelta
RUN_BUDGET = 320 * 60    # una vuelta dura como mucho 5 h 20 (GitHub corta a las 6 h); si sigue el directo, se relanza
CHECK_EVERY = 180        # mientras haya directo, se mira cada 3 min si hay tramo nuevo
DRIVE = "drive"          # destino extra de la cola: tu Drive, carpeta «ClipForge (bot)», para revisar desde la web
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
    for q in state["queue"]:  # clips apuntados antes de existir lo de Drive: también se guardan allí
        if not q.get("drive_checked"):
            q["drive_checked"] = True
            q.setdefault("top", True)
            if DRIVE not in q["todo"]:
                q["todo"].insert(0, DRIVE)
    # si cambias el secreto ACCOUNTS_JSON (al reconectar cuentas), manda la conexión nueva
    fresh = os.environ.get("ACCOUNTS_JSON") or ""
    sig = hashlib.sha256(fresh.encode()).hexdigest()
    if fresh and state.get("accounts_sig") != sig:
        state["accounts"], state["accounts_sig"] = json.loads(fresh), sig
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
    """Revisa los tramos pendientes del directo de un streamer y mete en la cola los momentos buenos.
    Devuelve (momentos nuevos, ¿sigue en directo?)."""
    chan = f"https://www.twitch.tv/{name}"
    live = app.is_live(chan)
    try:
        _, vod, title = app.live_vod(chan)
    except app.gr.Error:
        return 0, live
    info = state["vods"].setdefault(vod, {"pos": 0.0, "done": [], "titles": [], "title": title})
    info.setdefault("titles", [])  # directos guardados por la versión anterior del bot
    info["seen"] = time.time()
    if not live and info.get("finished"):
        return 0, False
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
                                      avoid=info["titles"], about=f"{name}: {title}")
        clips = [c for c in clips if float(c.get("score") or 0) >= MIN_SCORE
                 and not any(min(e, c["end"]) - max(s, c["start"]) > 5 for s, e in info["done"])]
        clips.sort(key=lambda c: -float(c.get("score") or 0))
        per = (b - pos) / app.LIVE_WINDOW
        clips = clips[:max(1, round(KEEP_PER_30 * per))]
        n_auto = max(1, round(AUTO_PER_30 * per))
        for rank, c in enumerate(clips):
            s, e = float(c["start"]), float(c["end"])
            # los muy buenos se suben solos; todos van además a tu Drive para verlos en la Biblioteca
            top = rank < n_auto and float(c.get("score") or 0) >= app.AUTO_SCORE
            state["queue"].append({"vod": vod, "name": name, "added": time.time(), "top": top,
                                   "todo": [DRIVE] + (list(PLATFORMS) if top else []),
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
    return added, live


def publish(state, deadline, blocked):
    """Sube la cola de mejor a peor nota. Devuelve cuántas subidas se han hecho."""
    cutoff = time.time() - QUEUE_HOURS * 3600
    state["queue"] = [q for q in state["queue"] if q["added"] > cutoff]
    state["queue"].sort(key=lambda q: -float(q["clip"].get("score") or 0))
    uploads, cams, work = 0, {}, tempfile.mkdtemp()
    last = state.setdefault("last_up", {})
    due = lambda p: p != PLATFORMS[0] or time.time() - last.get("yt", 0) >= YT_SPACING
    for q in list(state["queue"]):
        todo = [p for p in q["todo"] if p not in blocked and due(p)]
        if time.time() > deadline or len(blocked) == len(PLATFORMS) + 1:
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
                if p == DRIVE:
                    link = app.drive_upload(f)
                    res = f"en tu Drive ({app.BOT_FOLDER})"
                    if not q.get("top"):
                        app.notify(f"Clip para revisar ({q['name']}): «{c.get('title')}» (nota {c.get('score')})\n"
                                   f"Verlo: {link}\nSi te gusta, súbelo desde la web › Biblioteca.", video=f)
                elif p == PLATFORMS[1]:
                    link = app.drive_upload(f, TT_FOLDER)
                    res = f"en tu Drive ({TT_FOLDER}) para subirlo tú"
                    app.notify(f"Para TikTok ({q['name']}): «{c.get('title')}» (nota {c.get('score')})\n"
                               f"Descárgalo: {link}\nTexto: {c.get('description') or ''} {c.get('hashtags') or ''}",
                               video=f)
                else:
                    res = app.yt_upload(f, c, "public")
                    app.notify(f"Subido ({q['name']}): «{c.get('title')}» (nota {c.get('score')}) → {p}: {res}")
                print(f"[{q['name']}] «{c.get('title')}» (nota {c.get('score')}) → {p}: {res}")
                q["todo"].remove(p)
                uploads += 1
                if p == PLATFORMS[0]:
                    last["yt"] = time.time()
            except app.gr.Error as e:
                msg = str(getattr(e, "message", e))
                if app.limit_hit(msg) or p != PLATFORMS[0]:  # límite del día o sin permiso de Drive: otra vuelta
                    blocked.add(p)
                    print(f"{p}: límite alcanzado por ahora ({msg[:90]}); lo que falta queda en la cola.")
                    app.notify(f"⚠️ {p}: no puedo subir más por ahora ({msg[:150]}). Lo pendiente queda en cola.")
                else:  # error de este clip en concreto: no se reintenta
                    q["todo"].remove(p)
                    print(f"[{q['name']}] «{c.get('title')}» → {p}: error ({msg[:120]})")
                    app.notify(f"⚠️ «{c.get('title')}» → {p}: error ({msg[:150]})")
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
    end, blocked, added, uploads, live_any = time.time() + RUN_BUDGET, set(), 0, 0, False
    try:
        while True:  # mientras alguien esté en directo, la vuelta sigue trabajando
            deadline, live_any, no_quota = min(end, time.time() + WORK_BUDGET), False, False
            for name in streamers():
                try:
                    n, live = analyze(name, state, deadline)
                    added, live_any = added + n, live_any or live
                except app.NoQuota as e:
                    print(f"Sin cuota de IA por ahora: {getattr(e, 'message', e)}")
                    no_quota = live_any = True  # se reintenta más tarde, sin perder el directo
                    break
                except app.gr.Error as e:  # un streamer que falla no para a los demás
                    print(f"[{name}] error: {getattr(e, 'message', e)}")
                    app.notify(f"⚠️ [{name}] error: {str(getattr(e, 'message', e))[:200]}")
                    # si sigue en directo, se reintenta en la siguiente comprobación en vez de dar la vuelta por acabada
                    live_any = live_any or app.is_live(f"https://www.twitch.tv/{name}")
            uploads += publish(state, min(end, deadline + 15 * 60), blocked)
            save_state(fernet, state)
            if not live_any or time.time() > end - CHECK_EVERY:
                break
            time.sleep(CHECK_EVERY * (10 if no_quota else 1))
    finally:
        state["accounts"] = app.load_accounts()  # tokens renovados
        cutoff = time.time() - 7 * 86400  # se olvidan los directos de hace más de una semana
        state["vods"] = {k: v for k, v in state["vods"].items() if v.get("seen", 0) > cutoff}
        save_state(fernet, state)
    if live_any:  # el directo sigue: la siguiente vuelta se lanza ya, sin esperar al horario de GitHub
        open("relaunch", "w").close()
    summary = f"{added} momentos nuevos · {uploads} subidas · {len(state['queue'])} clips esperando en la cola."
    print("Listo: " + summary)
    if added or uploads:
        app.notify("Resumen de la vuelta: " + summary)


if __name__ == "__main__":
    main()
