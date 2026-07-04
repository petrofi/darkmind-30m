from darkmind_v2.tokenizer.test_roundtrip import run_roundtrip


class MockTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(item) for item in ids)


def test_mock_tokenizer_roundtrip_for_turkish_and_english() -> None:
    results = run_roundtrip(MockTokenizer(), ["Türkiye", "The capital of Turkey is"])
    assert all(item.exact_match for item in results)
    assert [item.token_count for item in results] == [7, 24]

