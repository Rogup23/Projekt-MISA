import time
import os
from dotenv import load_dotenv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from obsws_python import ReqClient

load_dotenv()

# --- KONFIGURATION ---
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = os.getenv("OBS_PASSWORD")      # <-- OBS-WebSocket-Passwort in .env

# Szenen & Quellen in OBS
SZENEN_NAME = "Szene"
QUELLE_HAND = "PyCharm_TestText"    # Text bei gehobener Hand
QUELLE_LINKS = "LinksTest"          # Text beim Zeigen nach links (5 Sek.)
QUELLE_RECHTS = "RechtsTest"        # Text beim Zeigen nach rechts (5 Sek.)

KAMERA_INDEX = 1
MODEL_PATH = "hand_landmarker.task"

# Parameter für Zeigegeste & Timer
ZEIGE_SCHWELLE_X = 0.12             # Mindestabstand auf der X-Achse zwischen Fingerbasis und Spitze
ANZEIGE_DAUER_SEK = 5.0             # Wie lange LinksTest / RechtsTest eingeblendet bleiben

# Verbindung zu OBS herstellen & IDs aller drei Textquellen abrufen
client = ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)

id_hand = client.get_scene_item_id(SZENEN_NAME, QUELLE_HAND).scene_item_id
id_links = client.get_scene_item_id(SZENEN_NAME, QUELLE_LINKS).scene_item_id
id_rechts = client.get_scene_item_id(SZENEN_NAME, QUELLE_RECHTS).scene_item_id

# Alle Texte zu Beginn ausblenden
client.set_scene_item_enabled(SZENEN_NAME, id_hand, False)
client.set_scene_item_enabled(SZENEN_NAME, id_links, False)
client.set_scene_item_enabled(SZENEN_NAME, id_rechts, False)

# Status- & Timer-Variablen
status_hand = False
status_links = False
status_rechts = False

links_ausblenden_um = 0.0
rechts_ausblenden_um = 0.0

# MediaPipe HandLandmarker initialisieren
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --- METHODE: Offene, gehobene Hand (alle 5 Finger gestreckt) erkennen ---
def erkenne_gehobene_hand(hand_landmarks) -> bool:
    """
    Prüft, ob die Hand aufrecht in die Kamera gehalten wird und alle 5 Finger
    vollständig ausgestreckt sind (offene Handfläche).
    """
    handgelenk_y = hand_landmarks[0].y

    # 1. Prüfen, ob die 4 Langfinger nach oben gestreckt sind (Spitze über dem Mittelgelenk)
    # Paare: (Fingerspitze, Mittelgelenk)
    finger_paare = [
        (8, 6),    # Zeigefinger
        (12, 10),  # Mittelfinger
        (16, 14),  # Ringfinger
        (20, 18)   # Kleiner Finger
    ]

    for spitze_idx, gelenk_idx in finger_paare:
        if hand_landmarks[spitze_idx].y >= hand_landmarks[gelenk_idx].y:
            return False  # Mindestens ein Finger ist eingeklappt

    # 2. Prüfen, ob der Daumen seitlich abgespreizt/gestreckt ist (funktioniert für linke & rechte Hand)
    daumen_spitze_x = hand_landmarks[4].x
    daumen_gelenk_x = hand_landmarks[3].x
    zeigefinger_basis_x = hand_landmarks[5].x

    daumen_abstand_spitze = abs(daumen_spitze_x - zeigefinger_basis_x)
    daumen_abstand_gelenk = abs(daumen_gelenk_x - zeigefinger_basis_x)

    if daumen_abstand_spitze <= daumen_abstand_gelenk:
        return False  # Daumen ist eingeklappt

    # 3. Sicherstellen, dass die Hand insgesamt aufrecht nach oben gehalten wird
    if hand_landmarks[12].y >= (handgelenk_y - 0.15):
        return False

    return True


# --- METHODE: Zeigegeste (Links / Rechts zeigen) erkennen ---
def erkenne_zeigegeste(hand_landmarks) -> str | None:
    """
    Prüft, ob der Zeigefinger deutlich nach links oder rechts zeigt,
    während Mittelfinger (12) und Ringfinger (16) eingeklappt sind.
    Gibt 'LINKS', 'RECHTS' oder None zurück.
    """
    zeigefinger_basis_x = hand_landmarks[5].x
    zeigefinger_spitze_x = hand_landmarks[8].x
    mittelfinger_spitze_x = hand_landmarks[12].x
    ringfinger_spitze_x = hand_landmarks[16].x

    differenz_x = zeigefinger_spitze_x - zeigefinger_basis_x

    # Zeigen nach RECHTS (im gespiegelten Bild wandert die Spitze zu größeren X-Werten)
    if differenz_x > ZEIGE_SCHWELLE_X:
        # Sicherstellen, dass Mittel- und Ringfinger nicht genauso weit nach rechts gestreckt sind
        if mittelfinger_spitze_x < (zeigefinger_spitze_x - 0.05) and ringfinger_spitze_x < (zeigefinger_spitze_x - 0.05):
            return "RECHTS"

    # Zeigen nach LINKS (im gespiegelten Bild wandert die Spitze zu kleineren X-Werten)
    elif differenz_x < -ZEIGE_SCHWELLE_X:
        # Sicherstellen, dass Mittel- und Ringfinger nicht genauso weit nach links gestreckt sind
        if mittelfinger_spitze_x > (zeigefinger_spitze_x + 0.05) and ringfinger_spitze_x > (zeigefinger_spitze_x + 0.05):
            return "LINKS"

    return None


# Kamera starten
cap = cv2.VideoCapture(KAMERA_INDEX, cv2.CAP_DSHOW)

# Variablen für Frame-Skipping initialisieren
SKIP_RATE = 3                 # Nur jeden 3. Frame durch MediaPipe schicken
frame_zaehler = 0
letzte_ergebnisse = None
ki_delay_ms = 0.0             # Startwert, damit die Anzeige nicht abstürzt

print("Starte Live-Analyse... Drücke 'q' im Kamerafenster zum Beenden.")

while cap.isOpened():
    start_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        print("Kein Kamerabild empfangen.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Frame-Zähler hochzählen
    frame_zaehler += 1

    # KI-Inferenz NUR bei jedem 3. Frame ausführen (spart Bildkonvertierung & CPU-Zeit!)
    if frame_zaehler % SKIP_RATE == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.perf_counter() * 1000)

        ki_start = time.perf_counter()
        letzte_ergebnisse = landmarker.detect_for_video(mp_image, timestamp_ms)
        ki_delay_ms = (time.perf_counter() - ki_start) * 1000

    # Für die weitere Auswertung (Skelett zeichnen & Gesten prüfen) das letzte Ergebnis nutzen
    results = letzte_ergebnisse

    hand_gehoben = False
    zeige_richtung = None

    # Prüfen, ob results existiert (beim allerersten Frame ist es noch None)
    if results and results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]

        # Skelett-Linien zeichnen
        punkte = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, punkte[start_idx], punkte[end_idx], (0, 255, 255), 2)
        for px, py in punkte:
            cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

        # Aufruf Methode Hand gehoben
        hand_gehoben = erkenne_gehobene_hand(hand_landmarks)

        # Aufruf Methode Zeigen nach links/rechts
        zeige_richtung = erkenne_zeigegeste(hand_landmarks)

    jetzt = time.perf_counter()

    # --- STEUERUNG 1: PyCharm_TestText (Hand gehoben) ---
    if hand_gehoben != status_hand:
        obs_start = time.perf_counter()
        client.set_scene_item_enabled(SZENEN_NAME, id_hand, hand_gehoben)
        obs_delay_ms = (time.perf_counter() - obs_start) * 1000
        status_hand = hand_gehoben
        print(f"'{QUELLE_HAND}' -> {hand_gehoben} | OBS-Delay: {obs_delay_ms:.2f} ms")

    # --- STEUERUNG 2: LinksTest (für 5 Sekunden beim Zeigen nach links) ---
    if zeige_richtung == "LINKS":
        links_ausblenden_um = jetzt + ANZEIGE_DAUER_SEK
        if not status_links:
            obs_start = time.perf_counter()
            client.set_scene_item_enabled(SZENEN_NAME, id_links, True)
            obs_delay_ms = (time.perf_counter() - obs_start) * 1000
            status_links = True
            print(f"Zeige LINKS -> '{QUELLE_LINKS}' eingeblendet (5s) | OBS-Delay: {obs_delay_ms:.2f} ms")

    if status_links and jetzt >= links_ausblenden_um:
        client.set_scene_item_enabled(SZENEN_NAME, id_links, False)
        status_links = False
        print(f"'{QUELLE_LINKS}' nach 5s ausgeblendet.")

    # --- STEUERUNG 3: RechtsTest (für 5 Sekunden beim Zeigen nach rechts) ---
    if zeige_richtung == "RECHTS":
        rechts_ausblenden_um = jetzt + ANZEIGE_DAUER_SEK
        if not status_rechts:
            obs_start = time.perf_counter()
            client.set_scene_item_enabled(SZENEN_NAME, id_rechts, True)
            obs_delay_ms = (time.perf_counter() - obs_start) * 1000
            status_rechts = True
            print(f"Zeige RECHTS -> '{QUELLE_RECHTS}' eingeblendet (5s) | OBS-Delay: {obs_delay_ms:.2f} ms")

    if status_rechts and jetzt >= rechts_ausblenden_um:
        client.set_scene_item_enabled(SZENEN_NAME, id_rechts, False)
        status_rechts = False
        print(f"'{QUELLE_RECHTS}' nach 5s ausgeblendet.")

    # Delays & Status im Vorschaufenster anzeigen
    gesamt_delay_ms = (time.perf_counter() - start_time) * 1000

    if zeige_richtung:
        status_text = f"ZEIGE {zeige_richtung}"
        farbe = (255, 200, 0)
    elif hand_gehoben:
        status_text = "HAND GEHOBEN"
        farbe = (0, 255, 0)
    else:
        status_text = "Warte auf Geste..."
        farbe = (0, 0, 255)

    cv2.putText(frame, f"Status: {status_text}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, farbe, 2)
    cv2.putText(frame, f"KI-Inferenz: {ki_delay_ms:.1f} ms | Gesamt-Loop: {gesamt_delay_ms:.1f} ms", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("MISA Toolbox - Gestensteuerung", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Aufräumen beim Beenden
landmarker.close()
client.set_scene_item_enabled(SZENEN_NAME, id_hand, False)
client.set_scene_item_enabled(SZENEN_NAME, id_links, False)
client.set_scene_item_enabled(SZENEN_NAME, id_rechts, False)
cap.release()
cv2.destroyAllWindows()