"""
Helper script to set up bcrypt-hashed passwords in the Pengguna table.

Use this to:
  1. Generate a bcrypt hash for a plaintext password (to insert manually), OR
  2. Directly update/insert a staff record in your MySQL database with a
     properly hashed password.

Run:
    python setup_staff_password.py

Make sure mysql-connector-python and bcrypt are installed:
    pip install mysql-connector-python bcrypt

Credentials are read from the SAME .streamlit/secrets.toml file your
Streamlit app already uses — no need to duplicate them here. Expected keys:

    db_host     = "your_host"
    db_port     = 3306
    db_name     = "pcad_db"
    db_user     = "root"
    db_password = "your_actual_password"
"""
import os
import sys
import bcrypt
import mysql.connector

# ── Load DB credentials from .streamlit/secrets.toml ───────────────────────
def load_secrets():
    """
    Looks for .streamlit/secrets.toml relative to this script's location.
    Uses built-in tomllib on Python 3.11+, falls back to the 'toml' package
    on older versions.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    secrets_path = os.path.join(script_dir, ".streamlit", "secrets.toml")

    if not os.path.exists(secrets_path):
        print(f"ERROR: Could not find secrets file at: {secrets_path}")
        print("Make sure '.streamlit/secrets.toml' exists next to this script,")
        print("with the same db_host/db_port/db_name/db_user/db_password keys")
        print("used by your Streamlit app.")
        sys.exit(1)

    try:
        import tomllib  # Python 3.11+
        with open(secrets_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import toml  # fallback for Python < 3.11
        except ImportError:
            print("ERROR: Need either Python 3.11+ (tomllib) or the 'toml' package.")
            print("Install with: pip install toml")
            sys.exit(1)
        with open(secrets_path, "r") as f:
            return toml.load(f)


_secrets = load_secrets()

DB_CONFIG = {
    "host":     _secrets["db_host"],
    "port":     _secrets["db_port"],
    "database": _secrets["db_name"],
    "user":     _secrets["db_user"],
    "password": _secrets["db_password"],
}


def hash_password(plain_password: str) -> str:
    """Returns a bcrypt hash string suitable for storing in kata_laluan."""
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def upsert_staff(nama_staf: str, emel: str, plain_password: str):
    """
    Inserts a new staff record, or updates the password if nama_staf/emel
    already exists, using a bcrypt hash (never stores plaintext).
    """
    hashed_pw = hash_password(plain_password)

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("SELECT id_staf FROM Pengguna WHERE nama_staf = %s OR emel = %s", (nama_staf, emel))
    existing = cursor.fetchone()

    if existing:
        cursor.execute(
            "UPDATE Pengguna SET kata_laluan = %s, nama_staf = %s, emel = %s WHERE id_staf = %s",
            (hashed_pw, nama_staf, emel, existing[0])
        )
        print(f"Updated existing staff (id_staf={existing[0]}) with new hashed password.")
    else:
        cursor.execute(
            "INSERT INTO Pengguna (nama_staf, emel, kata_laluan) VALUES (%s, %s, %s)",
            (nama_staf, emel, hashed_pw)
        )
        print(f"Inserted new staff record (id_staf={cursor.lastrowid}).")

    conn.commit()
    cursor.close()
    conn.close()


def rehash_all_plaintext_passwords():
    """
    Scans the Pengguna table and re-hashes any password that isn't already
    a valid bcrypt hash. Useful for migrating existing plaintext rows
    (like the auto-created 'Default Staff' / 'changeme' account).

    WARNING: This assumes any non-bcrypt-looking value is plaintext and
    rehashes it as-is. Review your data before running this in production.
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id_staf, nama_staf, kata_laluan FROM Pengguna")
    rows = cursor.fetchall()

    updated = 0
    for row in rows:
        current_value = row["kata_laluan"]
        # bcrypt hashes always start with $2a$, $2b$, or $2y$ and are 60 chars long
        looks_hashed = (
            isinstance(current_value, str)
            and len(current_value) == 60
            and current_value.startswith(("$2a$", "$2b$", "$2y$"))
        )
        if not looks_hashed:
            new_hash = hash_password(current_value)
            cursor.execute(
                "UPDATE Pengguna SET kata_laluan = %s WHERE id_staf = %s",
                (new_hash, row["id_staf"])
            )
            print(f"Re-hashed password for '{row['nama_staf']}' (id_staf={row['id_staf']})")
            updated += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"\nDone. {updated} password(s) re-hashed.")


if __name__ == "__main__":
    print("=" * 60)
    print("Pengguna Password Setup Helper")
    print("=" * 60)
    print("\nChoose an option:")
    print("1. Hash a single password (just print the hash, no DB changes)")
    print("2. Create/update one staff record with a hashed password")
    print("3. Re-hash ALL existing plaintext passwords in Pengguna table")
    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        pw = input("Enter plaintext password to hash: ")
        print(f"\nBcrypt hash:\n{hash_password(pw)}")

    elif choice == "2":
        name = input("Staff name (nama_staf): ").strip()
        email = input("Email (emel): ").strip()
        pw = input("Plaintext password: ").strip()
        upsert_staff(name, email, pw)

    elif choice == "3":
        confirm = input("This will re-hash ALL non-bcrypt passwords in Pengguna. Continue? (yes/no): ")
        if confirm.lower() == "yes":
            rehash_all_plaintext_passwords()
        else:
            print("Cancelled.")
    else:
        print("Invalid choice.")
