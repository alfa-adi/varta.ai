import pytest

def test_print_500(client, make_token, fake_redis, monkeypatch):
    import web.server as srv
    from starlette.testclient import TestClient
    client = TestClient(srv.app, raise_server_exceptions=True)
    monkeypatch.setattr(srv, "_redis", fake_redis)
    token = make_token("u1")

    r = client.post("/session/create_durable",
                    data={"customer_id": "c1"},
                    headers={"Authorization": f"Bearer {token}"})
    print("STATUS", r.status_code)
    print("CONTENT", r.text)
    assert r.status_code == 200
