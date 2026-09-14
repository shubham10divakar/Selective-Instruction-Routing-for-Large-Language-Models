"""Uses the offline hashing backend so these run with no network access."""
import numpy as np

from src.router.encoder import DualEncoder


def test_encode_module_is_unit_norm(sample_modules):
    encoder = DualEncoder(backend="hashing")
    emb = encoder.encode_module(sample_modules[0])
    assert emb.shape == (encoder.dim,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_encode_query_is_unit_norm():
    encoder = DualEncoder(backend="hashing")
    emb = encoder.encode_query("perform a hypothesis test on this dataset")
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)


def test_encode_library_shape(sample_modules):
    encoder = DualEncoder(backend="hashing")
    matrix = encoder.encode_library(sample_modules)
    assert matrix.shape == (len(sample_modules), encoder.dim)
    assert matrix.dtype == np.float32


def test_similar_queries_rank_relevant_module_higher(sample_modules):
    """A query about hypothesis testing should score the hypothesis-testing
    module higher than a totally unrelated one (secrets management)."""
    encoder = DualEncoder(backend="hashing")
    query_emb = encoder.encode_query("hypothesis testing p-value significance")

    stats_module = next(m for m in sample_modules if m.module_id == "statistics_hypothesis_testing_best_practices")
    security_module = next(m for m in sample_modules if m.module_id == "security_secrets_management_best_practices")

    stats_score = float(np.dot(query_emb, encoder.encode_module(stats_module)))
    security_score = float(np.dot(query_emb, encoder.encode_module(security_module)))

    assert stats_score > security_score


def test_alpha_zero_uses_only_functional_view(sample_modules):
    """At alpha=0 the module embedding should equal the functional-only embedding."""
    encoder = DualEncoder(backend="hashing", alpha=0.0)
    module = sample_modules[0]
    combined = encoder.encode_module(module)
    functional_only = encoder._embedder.encode(module.functional_text)
    functional_only = functional_only / np.linalg.norm(functional_only)
    assert np.allclose(combined, functional_only, atol=1e-5)
