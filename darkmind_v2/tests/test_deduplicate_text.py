from darkmind_v2.corpus.deduplicate_text import Document, deduplicate_documents, exact_hash, paragraph_hashes


def test_exact_duplicate_detection() -> None:
    docs = [
        Document("a", "Türkiye güvenli veri saklar.", {"source": "fixture"}),
        Document("b", " Türkiye   güvenli veri saklar. ", {"source": "fixture"}),
    ]
    accepted, rejected, mapping = deduplicate_documents(docs)
    assert len(accepted) == 1
    assert rejected[0].reason == "exact_duplicate"
    assert mapping["b"] == "a"


def test_near_duplicate_detection() -> None:
    docs = [
        Document("a", "python is a programming language for users", {}),
        Document("b", "python is a programming language for developers", {}),
    ]
    accepted, rejected, mapping = deduplicate_documents(docs, threshold=0.5, shingle_size=2)
    assert len(accepted) == 1
    assert rejected[0].reason == "near_duplicate"
    assert mapping["b"] == "a"


def test_hash_helpers_are_deterministic() -> None:
    assert exact_hash("A  B") == exact_hash("a b")
    assert paragraph_hashes("one\n\nTwo") == paragraph_hashes("one\n\nTwo")

