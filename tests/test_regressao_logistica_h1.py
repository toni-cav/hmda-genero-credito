import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "scripts"))
from regressao_logistica_h1 import (
    criar_especificacoes,
    formula_especificacao,
    preparar_dados,
)


class PrepararDadosH1Tests(unittest.TestCase):
    def test_fixed_effects_use_supported_string_dtype(self):
        n = 40
        frame = pd.DataFrame(
            {
                "lei": [f"lei-{i % 5}" for i in range(n)],
                "state_code": ["CA" if i % 2 else "NY" for i in range(n)],
                "income": [50.0 + i for i in range(n)],
                "combined_loan_to_value_ratio": [70.0 + i for i in range(n)],
                "loan_type": [str(i % 4 + 1) for i in range(n)],
                "dti": ["<20%", "20%-<30%", "30%-<36%", "36%-<50%", "50%-60%", ">60%"]
                * 6
                + ["<20%", "20%-<30%", "30%-<36%", "36%-<50%"],
                "applicant_age": ["25-34", "<25", "35-44", "45-54", "55-64", "65-74", ">74"]
                * 5
                + ["25-34", "<25", "35-44", "45-54", "55-64"],
                "negado": [i % 2 for i in range(n)],
                "mulher": [(i // 2) % 2 for i in range(n)],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "h1.parquet"
            frame.to_parquet(path, index=False)
            todas, base, principal, controles, idade, _, _ = preparar_dados(path)

        self.assertEqual(str(principal["lei"].dtype), "string")
        self.assertEqual(str(principal["state_code"].dtype), "string")
        especificacoes = criar_especificacoes(todas, base, principal, controles, idade)
        self.assertIn("state_code", formula_especificacao(especificacoes[4]))
        self.assertEqual(len(especificacoes), 8)


if __name__ == "__main__":
    unittest.main()
