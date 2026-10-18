from repomaro.tokenization import TokenCounter


def test_count_allows_endoftext_in_repo_files():
    n = TokenCounter("gpt-5.6-luna").count("hello <|endoftext|> world")
    assert n >= 1
