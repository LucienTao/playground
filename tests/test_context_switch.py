from rag_agent.detector.context_switch import is_context_switch


def test_context_switch_detection() -> None:
    assert is_context_switch("聊聊天气", "讨论机器学习", threshold=0.5)
    assert not is_context_switch("你好", "你好", threshold=0.5)
