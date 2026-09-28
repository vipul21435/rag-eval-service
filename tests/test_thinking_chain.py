from features.thinking_chain import process_thinking_content


def test_think_block_becomes_collapsible_details():
    result = process_thinking_content("Answer.<think>step 1\nstep 2</think> Done.")

    assert result.startswith("Answer.")
    assert "<details>" in result
    assert "<summary>Reasoning (click to expand)</summary>" in result
    assert "step 1\nstep 2" in result
    assert "</details>" in result
    assert result.endswith("Done.")
    assert "<think>" not in result


def test_other_tags_are_escaped_but_details_are_kept():
    result = process_thinking_content("<script>alert(1)</script><think>r</think>")

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in result
    assert "<details>" in result and "</details>" in result


def test_none_and_non_string_inputs():
    assert process_thinking_content(None) == ""
    assert process_thinking_content(42) == "42"


def test_unbalanced_think_tags_are_escaped_not_dropped():
    result = process_thinking_content("</think>before<think>after")

    assert result == "&lt;/think&gt;before&lt;think&gt;after"
