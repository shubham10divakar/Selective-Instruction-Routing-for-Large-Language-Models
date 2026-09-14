"""Tests for the offline template-based generators and the on-disk library
loader/saver, using tmp_path so they don't touch the real data/ directory."""
from src.evaluation.task_generator import EVAL_DOMAINS, generate_tasks_for_domain
from src.instruction.generator import TAXONOMY, generate_library_modules, generate_module
from src.instruction.library import InstructionLibrary


def test_generate_library_modules_produces_500_unique_modules():
    modules = generate_library_modules()
    ids = [m.module_id for m in modules]
    assert len(modules) == 500
    assert len(set(ids)) == 500  # all unique


def test_generated_module_content_within_target_token_range():
    modules = generate_library_modules()
    for m in modules[:20]:  # sample, full sweep is slow-ish but still fast; keep test quick
        assert 100 <= m.token_count <= 2500


def test_generate_module_is_deterministic():
    a = generate_module("coding", "python", "conventions", "{subject} Conventions")
    b = generate_module("coding", "python", "conventions", "{subject} Conventions")
    assert a.content == b.content
    assert a.capabilities == b.capabilities


def test_library_save_and_load_roundtrip(tmp_path):
    modules = generate_library_modules()[:10]
    library = InstructionLibrary(modules)
    library.save(tmp_path)

    reloaded = InstructionLibrary.load(tmp_path)
    assert len(reloaded) == len(modules)
    for m in modules:
        reloaded_m = reloaded.get(m.module_id)
        assert reloaded_m is not None
        assert reloaded_m.content == m.content
        assert reloaded_m.domain == m.domain


def test_library_stats_reports_per_domain_counts():
    modules = generate_library_modules()
    library = InstructionLibrary(modules)
    stats = library.stats()
    assert stats["n_modules"] == 500
    assert stats["n_domains"] == len(TAXONOMY)
    for domain in TAXONOMY:
        assert stats["modules_per_domain"][domain] == 50


def test_generate_tasks_for_domain_produces_80_tasks_with_valid_module_refs():
    modules = {m.module_id for m in generate_library_modules()}
    for domain in EVAL_DOMAINS:
        tasks = generate_tasks_for_domain(domain)
        assert len(tasks) == 80
        n_simple = sum(1 for t in tasks if t.complexity == "simple")
        n_compound = sum(1 for t in tasks if t.complexity == "compound")
        assert n_simple == 40
        assert n_compound == 40

        task_ids = [t.task_id for t in tasks]
        assert len(set(task_ids)) == len(task_ids)  # no duplicate task ids

        for task in tasks:
            for module_id in task.relevant_modules:
                assert module_id in modules
            for module_id in task.distractor_modules:
                assert module_id in modules


def test_compound_tasks_reference_two_domains():
    tasks = generate_tasks_for_domain("coding")
    compound = [t for t in tasks if t.complexity == "compound"]
    for task in compound:
        assert len(task.domains) == 2
        assert 2 <= len(task.relevant_modules) <= 4
