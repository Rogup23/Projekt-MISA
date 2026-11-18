import time
from obsws_python import ReqClient

#OBS-WebSocket-Einstellungen
client = ReqClient(host="localhost", port=4455, password="Das OBS Websocket Passwort hier rein")

#Abfrage OBS Version
version = client.get_version()
print(f"Verbunden mit OBS Studio Version: {version.obs_version}")

#Test Text bei OBS für 3 Sekunden einblenden und wieder ausblenden
szenen_name = "Szene"
quellen_name = "PyCharm_TestText"

response = client.get_scene_item_id(szenen_name, quellen_name)
item_id = response.scene_item_id

client.set_scene_item_enabled(szenen_name, item_id, True)
print(f"'{quellen_name}' wurde in OBS eingeblendet!")

time.sleep(3)

client.set_scene_item_enabled(szenen_name, item_id, False)
print(f"'{quellen_name}' wurde wieder ausgeblendet!")