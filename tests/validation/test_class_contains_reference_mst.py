from __future__ import annotations

import pytest

from core_sg.core_sg import CoreSG
from tests.validate import validate_mst_in_core_sg

pytestmark = [pytest.mark.validation, pytest.mark.slow]


@pytest.fixture(scope="module")
def fitted_class_core_sg(validation_dataset):
    X = validation_dataset["X"]
    core_sg = CoreSG(
        metric="euclidean", p=2, debug=True, match_reference_implementation=True
    )
    core_sg.fit(X, 30, test_only=True)
    return core_sg


class TestClassContainsReferenceMST:
    def test_class_core_sg_contains_reference_mst_for_each_k(
        self, fitted_class_core_sg, validation_dataset, reference_mst_builder
    ):
        n = validation_dataset["n"]

        for k_iter in range(30, 2, -2):
            mst_hdb = reference_mst_builder(fitted_class_core_sg._D, k_iter)
            result = validate_mst_in_core_sg(
                fitted_class_core_sg._core_sg, mst_hdb, n, k_iter
            )

            if not result.ok:
                print(result)
                raise ValueError(f"A MST para k = {k_iter} nao esta contida no Core-SG")
