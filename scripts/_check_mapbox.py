import os
import urllib.request

token = os.environ["MAPBOX_TOKEN"]
url = (
    "https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/static/"
    f"pin-l-fire-station+ff5722(105.5,21)/105.5,21,9/640x400@2x?access_token={token}"
)
print("url len", len(url))
try:
    with urllib.request.urlopen(url, timeout=15) as response:
        data = response.read()
    print("bytes", len(data))
except Exception as exc:  # noqa: BLE001
    print("error", type(exc).__name__, exc)
