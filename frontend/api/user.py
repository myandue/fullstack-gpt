from .common import api_request


def sign_up(username, email, password, full_name):
    return api_request(
        "POST",
        "/users/signup",
        json={
            "username": username,
            "email": email,
            "password": password,
            "full_name": full_name,
        },
    )


def login(username, password):
    return api_request(
        "POST",
        "/auth/login",
        json={"username": username, "password": password},
    )
