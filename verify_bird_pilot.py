import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path("data/benchmarks/bird")


def main() -> None:
    questions_file = DATA_DIR / "mini_dev_sqlite_50.json"
    gold_file = DATA_DIR / "mini_dev_sqlite_50_gold.sql"
    tables_file = DATA_DIR / "dev_tables.json"
    databases_dir = DATA_DIR / "dev_databases"

    with questions_file.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    print("=" * 60)
    print("BIRD Mini-Dev Pilot Verification")
    print("=" * 60)

    print(f"Question count     : {len(questions)}")
    print(
        f"Unique question IDs: "
        f"{len({question['question_id'] for question in questions})}"
    )

    print("\nDatabase distribution:")
    for database, count in sorted(
        Counter(question["db_id"] for question in questions).items()
    ):
        print(f"  {database}: {count}")

    print("\nDifficulty distribution:")
    for difficulty, count in sorted(
        Counter(question["difficulty"] for question in questions).items()
    ):
        print(f"  {difficulty}: {count}")

    print("\nFiles:")
    for file_path in [questions_file, gold_file, tables_file]:
        print(
            f"  {file_path}: "
            f"{file_path.stat().st_size:,} bytes"
        )

    print("\nDatabases:")
    for database_dir in sorted(databases_dir.iterdir()):
        if not database_dir.is_dir():
            continue

        total_size = sum(
            file.stat().st_size
            for file in database_dir.rglob("*")
            if file.is_file()
        )

        print(
            f"  {database_dir.name}: "
            f"{total_size / (1024 * 1024):.2f} MB"
        )


if __name__ == "__main__":
    main()