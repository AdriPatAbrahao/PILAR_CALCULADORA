"""
Inspeciona o CSV: mostra as primeiras linhas e tenta detectar a linha de cabeçalho.
Uso: python scripts\inspect_csv.py
"""
from pathlib import Path

FILE = Path(__file__).parent.parent / "data" / "dados_pilares.csv"

def print_first_lines(n=20):
    print(f"File: {FILE}")
    with FILE.open("r", encoding="latin-1", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            print(f"{i:02d}: {line.rstrip()}")

def find_candidate_header(tokens=("fck","pe","pe direito","largura","altura","as","n","mx","my")):
    with FILE.open("r", encoding="latin-1", errors="replace") as fh:
        for i, line in enumerate(fh):
            low = line.lower()
            if any(t in low for t in tokens):
                print(f"Possible header at line {i}: {line.rstrip()}")

if __name__ == "__main__":
    print_first_lines(40)
    print("\n--- Searching for candidate header lines ---\n")
    find_candidate_header()