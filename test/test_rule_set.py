#TODO: test for rule set behavior (e.g. rule_set.add_rule())
import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule_set import RuleSet
from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause
from dexire.core.rule_set import RuleSet



@pytest.fixture
def create_rule_set():
    return RuleSet()

def test_rule_set(create_rule_set):
    rule_set = create_rule_set
    assert rule_set.get_feature_idx() == []
    assert rule_set.get_feature_name() == []
    assert len(rule_set) == 0
    assert rule_set == RuleSet()