from src.router.composer import ContextComposer


def test_compose_orders_by_descending_score(sample_modules):
    composer = ContextComposer(budget_tokens=8000)
    modules = sample_modules[:3]
    scores = [0.5, 0.9, 0.7]  # module[1] should come first, then [2], then [0]
    context = composer.compose(modules, scores)

    pos1 = context.find(modules[1].name)
    pos2 = context.find(modules[2].name)
    pos0 = context.find(modules[0].name)
    assert pos1 < pos2 < pos0


def test_compose_includes_all_modules_within_budget(sample_modules):
    composer = ContextComposer(budget_tokens=8000)
    modules = sample_modules
    scores = [1.0] * len(modules)
    context = composer.compose(modules, scores)
    for module in modules:
        assert module.name in context
        assert module.content.split("\n")[0] in context


def test_compose_respects_token_budget(sample_modules):
    composer = ContextComposer(budget_tokens=8000)
    # A tiny budget should truncate to well under the full content size.
    tiny_composer = ContextComposer(budget_tokens=10)
    modules = sample_modules
    scores = [1.0] * len(modules)

    full_context = composer.compose(modules, scores)
    tiny_context = tiny_composer.compose(modules, scores)

    assert tiny_composer.count_tokens(tiny_context) <= 10
    assert tiny_composer.count_tokens(tiny_context) < composer.count_tokens(full_context)


def test_compose_empty_modules_returns_empty_string():
    composer = ContextComposer()
    assert composer.compose([], []) == ""


def test_count_tokens_empty_string_is_zero():
    composer = ContextComposer()
    assert composer.count_tokens("") == 0


def test_count_tokens_positive_for_nonempty_text():
    composer = ContextComposer()
    assert composer.count_tokens("hello world") > 0
