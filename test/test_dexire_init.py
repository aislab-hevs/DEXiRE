import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dexire.dexire import DEXiRE
from dexire.core.dexire_abstract import Mode, RuleExtractorEnum


@pytest.mark.parametrize(
    "rule_extraction_method, expected_extractors",
    [
        ("oneR", {RuleExtractorEnum.ONERULE}),
        ("treeR", {RuleExtractorEnum.TREERULE}),
        ("mixed", {RuleExtractorEnum.ONERULE, RuleExtractorEnum.TREERULE}),
    ],
)
def test_init_with_string_rule_extractor_creates_default_extractors(
    rule_extraction_method, expected_extractors
):
    dexire = DEXiRE(
        model=None,
        mode=Mode.CLASSIFICATION,
        rule_extraction_method=rule_extraction_method,
    )
    assert dexire.rule_extraction_method == RuleExtractorEnum(rule_extraction_method)
    assert isinstance(dexire.rule_extractor, dict)
    assert set(dexire.rule_extractor.keys()) == expected_extractors


def test_init_with_invalid_rule_extractor_string_raises():
    with pytest.raises(NotImplementedError):
        DEXiRE(model=None, mode=Mode.CLASSIFICATION, rule_extraction_method="invalid")
