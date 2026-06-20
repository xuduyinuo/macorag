from macorag.io_utils import normalize_text, sha1_text, read_jsonl, write_jsonl


def test_normalize_text_collapses_whitespace_but_keeps_case():
    assert normalize_text("  Alice\n  Smith\tFounded  X. ") == "Alice Smith Founded X."


def test_sha1_text_uses_normalized_text():
    assert sha1_text("Alice  Smith") == sha1_text(" Alice Smith ")


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "items.jsonl"
    write_jsonl(path, [{"id": "a"}, {"id": "b"}])

    assert list(read_jsonl(path)) == [{"id": "a"}, {"id": "b"}]
