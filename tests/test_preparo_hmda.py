import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "scripts"))
from preparo_hmda import FEATURES_BASE, carregar_amostra_modelagem, dividir_amostra


class CarregarAmostraModelagemTests(unittest.TestCase):
    def test_prepares_shared_features_and_excludes_ineligible_rows(self):
        frame = pd.DataFrame(
            {
                "state_code": ["CA", "NY", "TX", None],
                "income": [50.0, 0.0, 75.0, 90.0],
                "combined_loan_to_value_ratio": [80.0, 90.0, 110.0, 70.0],
                "loan_type": ["1", "2", "1", "3"],
                "dti": ["<20%", "20%-<30%", "36%-<50%", "30%-<36%"],
                "applicant_age": ["25-34", "35-44", "45-54", "55-64"],
                "negado": [0, 1, 0, 1],
                "mulher": [0, 1, 1, 0],
                "extra": ["keep", "drop", "keep", "drop"],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.parquet"
            frame.to_parquet(path, index=False)
            prepared = carregar_amostra_modelagem(path, colunas_extras=["extra"])

        self.assertEqual(prepared["extra"].tolist(), ["keep", "keep"])
        self.assertEqual(prepared["negado"].tolist(), [0, 0])
        self.assertAlmostEqual(prepared.loc[0, "log_renda"], math.log(50.0))
        self.assertEqual(prepared["dti"].dtype.name, "category")
        self.assertTrue(set(FEATURES_BASE).issubset(prepared.columns))

    def test_split_is_reproducible_stratified_and_disjoint(self):
        frame = pd.DataFrame(
            {
                "negado": [outcome for outcome in (0, 1) for _ in range(50)],
                "mulher": [gender for _ in (0, 1) for gender in (0, 1) for _ in range(25)],
            }
        )
        train, test, train_indices, test_indices = dividir_amostra(frame)
        train_again, test_again, _, _ = dividir_amostra(frame)

        self.assertEqual(len(test), 30)
        self.assertEqual(len(set(train_indices) & set(test_indices)), 0)
        self.assertEqual(set(train_indices) | set(test_indices), set(frame.index))
        self.assertTrue(train.equals(train_again))
        self.assertTrue(test.equals(test_again))
        stratum_counts = train.groupby(["negado", "mulher"], observed=True).size()
        self.assertEqual(sum(stratum_counts), len(train))
        self.assertLessEqual(stratum_counts.max() - stratum_counts.min(), 1)


if __name__ == "__main__":
    unittest.main()
