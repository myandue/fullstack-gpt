import requests

API_BASE = "http://localhost:8000"


def api_request(method, endpoint, **kwargs):
    url = f"{API_BASE}{endpoint}"
    response = requests.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()
