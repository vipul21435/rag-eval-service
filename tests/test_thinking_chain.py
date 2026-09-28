import pytest

from ragsvc.features.thinking_chain import ThinkingSplit, split_thinking


def test_trailing_think_block_is_separated_from_the_answer():
    result = split_thinking("Answer.<think>step 1\nstep 2</think> Done.")

    assert result == ThinkingSplit(answer="Answer. Done.", reasoning="step 1\nstep 2")


def test_leading_think_block_keeps_the_answer_after_it():
    # Ollama and OpenAI-compatible servers emit reasoning first (deepseek-r1, qwen3).
    result = split_thinking("<think>The summary already answers it.</think>\n\nNO_FURTHER_QUERY")

    assert result.answer == "NO_FURTHER_QUERY"
    assert result.reasoning == "The summary already answers it."


def test_multiple_think_blocks_are_joined():
    result = split_thinking("<think>first</think>Part one. <think>second</think>Part two.")

    assert result.answer == "Part one. Part two."
    assert result.reasoning == "first\n\nsecond"


def test_plain_answers_are_returned_verbatim_without_html_escaping():
    text = "if x < 10 and y > 3: use Vec<String>"

    assert split_thinking(text) == ThinkingSplit(answer=text, reasoning=None)


def test_markup_in_the_answer_is_not_interpreted():
    text = '<details open ontoggle="alert(1)">x</details><script>alert(1)</script>'

    assert split_thinking(text).answer == text


def test_unclosed_think_block_is_all_reasoning():
    result = split_thinking("<think>cut off by the token limit")

    assert result == ThinkingSplit(answer="", reasoning="cut off by the token limit")


def test_close_tag_without_open_tag_treats_the_prefix_as_reasoning():
    # Chat templates that pre-fill <think> make the model start with reasoning.
    result = split_thinking("weighing the options</think>final answer")

    assert result == ThinkingSplit(answer="final answer", reasoning="weighing the options")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, ThinkingSplit(answer="")),
        (42, ThinkingSplit(answer="42")),
        ("<think></think>   ", ThinkingSplit(answer="")),
    ],
)
def test_none_non_string_and_empty_inputs(text, expected):
    assert split_thinking(text) == expected
