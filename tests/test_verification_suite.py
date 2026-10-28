"""
tests/test_verification_suite.py - Wrapper ensuring test_verification runs
even when pytest is invoked with positional 'tests/'.
"""
from test_verification import (
    test_factor_coding_reversibility,
    test_design_matrix_properties,
    test_desirability_function_properties,
    test_canonical_spectral_identity,
)

def test_verification_suite():
    test_factor_coding_reversibility()
    test_design_matrix_properties()
    test_desirability_function_properties()
    test_canonical_spectral_identity()
