import pytest

from ndbench.codes import BB72, GROSS144


@pytest.mark.parametrize("code, n, k", [(BB72, 72, 12), (GROSS144, 144, 12)])
def test_bb_parameters(code, n, k):
    assert code.n == n and code.k == k
    assert not ((code.hx @ code.hz.T) % 2).any()
    assert (code.hx.sum(axis=1) == 6).all()
