# Rodar: python src/01_dados.py

import os
import urllib.request
import zipfile
from datetime import date

import duckdb

# Pasta dos dados (fora do OneDrive, pois são vários GB)
PASTA = os.path.expanduser("~/hmda_dados")
os.makedirs(PASTA, exist_ok=True)

URL = "https://files.ffiec.cfpb.gov/static-data/snapshot/2025/"
LAR_ZIP = f"{PASTA}/2025_public_lar_csv.zip"   # pedidos de crédito
TS_ZIP = f"{PASTA}/2025_public_ts_csv.zip"     # lista de instituições
PARQUET = f"{PASTA}/hmda_2025.parquet"

# 1. Baixar (pula se já existir)
for arquivo in [LAR_ZIP, TS_ZIP]:
    if not os.path.exists(arquivo):
        print("Baixando", os.path.basename(arquivo), "...")
        urllib.request.urlretrieve(URL + os.path.basename(arquivo), arquivo)
print("Download feito em", date.today())

# 2. Descompactar
zipfile.ZipFile(LAR_ZIP).extract("2025_public_lar_csv.csv", PASTA)
zipfile.ZipFile(TS_ZIP).extract("2025_public_ts_csv.csv", PASTA)
lar_csv = f"{PASTA}/2025_public_lar_csv.csv"
ts_csv = f"{PASTA}/2025_public_ts_csv.csv"

# 3. Checar integridade
con = duckdb.connect()
con.execute(f"CREATE VIEW lar AS SELECT * FROM read_csv('{lar_csv}', all_varchar = true)")
con.execute(f"CREATE VIEW ts AS SELECT * FROM read_csv('{ts_csv}')")

linhas, bancos_lar = con.execute("SELECT count(*), count(DISTINCT lei) FROM lar").fetchone()
colunas = len(con.execute("DESCRIBE lar").fetchall())
bancos_ts, linhas_ts = con.execute("SELECT count(*), sum(lar_count) FROM ts").fetchone()

print(f"Linhas:       {linhas:,}  (esperado pelo TS: {linhas_ts:,})")
print(f"Colunas:      {colunas}  (esperado: 99)")
print(f"Instituições: {bancos_lar:,}  (esperado pelo TS: {bancos_ts:,})")
assert linhas == linhas_ts and colunas == 99 and bancos_lar == bancos_ts, "Base incompleta!"

# 4. Salvar em Parquet, sem as colunas _2 a _5 (respostas múltiplas que não usamos)
todas = [c[0] for c in con.execute("DESCRIBE lar").fetchall()]
manter = [c for c in todas if c[-2:] not in ("_2", "_3", "_4", "_5") and c != "multifamily_affordable_units"]
con.execute(f"COPY (SELECT {', '.join(manter)} FROM lar) TO '{PARQUET}' (FORMAT parquet, COMPRESSION zstd)")
con.close()
print("Parquet salvo em", PARQUET)

# 5. Apagar os CSVs extraídos (5 GB); o zip continua guardado
os.remove(lar_csv)
os.remove(ts_csv)
