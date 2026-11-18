from pathlib import Path

# CSV file in the same directory
path = Path(__file__).resolve().parent / "golden_fda_eval.csv"

# Create a backup before conversion
backup = path.with_suffix(".csv.bak")
backup.write_bytes(path.read_bytes())

# Read the original Windows-encoded CSV
content = path.read_text(encoding="cp1252")

# Save permanently as UTF-8
path.write_text(content, encoding="utf-8-sig")

print("CSV successfully converted to UTF-8.")
print("Backup saved at:", backup)