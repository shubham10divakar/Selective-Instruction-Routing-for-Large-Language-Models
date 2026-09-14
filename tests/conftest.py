import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instruction.module import InstructionModule


@pytest.fixture
def sample_modules() -> list[InstructionModule]:
    return [
        InstructionModule(
            module_id="statistics_hypothesis_testing_best_practices",
            name="Hypothesis Testing Best Practices",
            domain="statistics",
            description="Guidelines for conducting statistical hypothesis tests correctly.",
            capabilities=["hypothesis testing", "t-tests", "p-value interpretation", "significance"],
            content="1. State the null and alternative hypotheses explicitly.\n"
                    "2. Check assumptions before running the test.\n"
                    "3. Report effect size alongside p-values.\n"
                    "4. Correct for multiple comparisons when relevant.",
        ),
        InstructionModule(
            module_id="coding_python_conventions",
            name="Python Coding Conventions",
            domain="coding",
            description="Style and structure conventions for Python code.",
            capabilities=["python", "pep8", "naming", "formatting"],
            content="1. Use snake_case for functions and variables.\n"
                    "2. Keep functions under 50 lines where practical.\n"
                    "3. Prefer explicit imports over wildcard imports.",
        ),
        InstructionModule(
            module_id="writing_technical_writing_best_practices",
            name="Technical Writing Best Practices",
            domain="writing",
            description="Guidelines for writing clear technical documentation.",
            capabilities=["technical writing", "clarity", "audience", "structure"],
            content="1. Lead with the conclusion, not the process.\n"
                    "2. Define jargon on first use.\n"
                    "3. Use active voice.",
        ),
        InstructionModule(
            module_id="coding_python_best_practices",
            name="Python Best Practices",
            domain="coding",
            description="Best practices for writing maintainable Python.",
            capabilities=["python", "maintainability", "testing", "error handling"],
            content="1. Write a test for every new function.\n"
                    "2. Handle exceptions explicitly, never with a bare except.\n"
                    "3. Avoid mutable default arguments.",
        ),
        InstructionModule(
            module_id="security_secrets_management_best_practices",
            name="Secrets Management Best Practices",
            domain="security",
            description="Guidelines for handling credentials and secrets safely.",
            capabilities=["secrets", "credentials", "encryption", "security"],
            content="1. Never hard-code credentials in source.\n"
                    "2. Rotate secrets on a fixed schedule.\n"
                    "3. Use a secrets manager, not environment files, in production.",
        ),
    ]
