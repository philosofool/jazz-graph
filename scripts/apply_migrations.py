import re
import os

import psycopg

def apply_migrations(conn, path="/workspace/queries"):
    """Read files in path, applying them as updates to the DB.

    The DB records the names of file read and applied; only files
    not in the DB's record will be applied.
    """
    cur = conn.cursor()

    # Create a schema migrations table to track which migration files
    # have been applied.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,  -- file name applied.
            applied_at TIMESTAMP DEFAULT now()
        )
    """)

    applied = {
        row[0]
        for row in cur.execute("SELECT version FROM schema_migrations")
    }

    for file in sorted(os.listdir(path)):
        if not re.match('\d\d\d', file):
            # all valid migration files should start with three numerals, e.g. 001_update.sql
            continue
        if file not in applied:
            print(f"Applying {file}.")
            with open(os.path.join(path, file)) as f:
                sql = f.read()
                cur.execute(sql)

            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)",
                (file,)
            )

if __name__ == '__main__':
    with psycopg.connect("dbname=musicbrainz_db user=philosofool") as connection:
        connection.autocommit = False
        apply_migrations(connection)
        connection.commit()
