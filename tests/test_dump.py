def test_dump_db(client):
    from web.storage.mongo import get_db
    db = get_db()
    turns = list(db['translation_turns'].find({'turn_id': {'$regex': 'req-dual'}}))
    print(f"\\n\\nTURNS: {turns}\\n\\n")
    assert False
