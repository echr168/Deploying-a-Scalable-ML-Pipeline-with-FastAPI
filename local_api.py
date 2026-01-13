import json

import requests

BASE_URL = "http://127.0.0.1:8001"
# TODO: send a GET using the URL http://127.0.0.1:8000


def main() -> None:
    r = requests.get(f"{BASE_URL}/", timeout=10)
    print("GET status code:", r.status_code)
    try:
        print("GET response:", r.json())
    except json.JSONDecodeError:
        print("GET response (raw):", r.text)

    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    r = requests.post(f"{BASE_URL}/data/", json=data, timeout=10)
    print("POST status code:", r.status_code)
    try:
        print("POST response:", r.json())
    except json.JSONDecodeError:
        print("POST response (raw):", r.text)


if __name__ == "__main__":
    main()
