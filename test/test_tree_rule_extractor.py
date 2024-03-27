#TODO: test for rule extractor based on decision tree. 
import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexire.core.rule import Rule
from dexire.core.expression import Expr
from dexire.core.clause import DisjunctiveClause, ConjunctiveClause
from dexire.rule_extractors.tree_rule_extractor import TreeRuleExtractor
from dexire.core.rule_set import RuleSet



@pytest.fixture
def create_rule():
    return Rule([Expr(0, 0.0, '==', 'feature_0'), Expr(0, 0.0, '==', 'feature_1')])



@pytest.fixture
def create_rule_set():
    pass