#TODO: Add tests for rule extraction with One rule method. 
import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule import Rule
from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause
from dexire.rule_extractors.one_rule_extracctor import OneRuleExtractor
from dexire.core.rule_set import RuleSet



@pytest.fixture
def create_rule():
    return Rule([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')])



@pytest.fixture
def create_rule_set():  
    
    return RuleSet([Rule([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')]), Rule([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')]), Rule([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')]), Rule([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')]), Rule([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')])])

def test_rule_set(create_rule_set):
    rule_set = create_rule_set
    assert rule_set.get_feature_idx() == [0, 1]
    assert rule_set.get_feature_name() == ['feature_0', 'feature_1']
    assert len(rule_set) == 5