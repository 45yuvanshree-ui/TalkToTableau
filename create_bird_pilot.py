import json
import random
import shutil
from pathlib import Path


SOURCE_DIR = Path("minidev/MINIDEV")
OUTPUT_DIR = Path("data/benchmarks/bird")

DATABASES = [
    "california_schools",
    "debit_card_specializing",
    "financial",
    "formula_1",
    "student_club",
    "superhero",
]

QUESTIONS_PER_DATABASE = {
    "simple": 3,
    "moderate": 4,
    "challenging": 1,
}

TARGET_QUESTION_COUNT = 50
RANDOM_SEED = 42


def main() -> None:
    source_questions = SOURCE_DIR / "mini_dev_sqlite.json"
    source_tables = SOURCE_DIR / "dev_tables.json"
    source_databases = SOURCE_DIR / "dev_databases"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_databases = OUTPUT_DIR / "dev_databases"
    output_databases.mkdir(parents=True, exist_ok=True)

    with source_questions.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    rng = random.Random(RANDOM_SEED)

    selected = []
    selected_ids = set()

    # ---------------------------------------------------------
    # Step 1: Select the planned difficulty mix from each
    # database.
    # ---------------------------------------------------------
    for database in DATABASES:
        database_questions = [
            question
            for question in questions
            if question["db_id"] == database
        ]

        for difficulty, count in QUESTIONS_PER_DATABASE.items():
            candidates = [
                question
                for question in database_questions
                if question["difficulty"] == difficulty
                and question["question_id"] not in selected_ids
            ]

            sample_count = min(count, len(candidates))

            chosen = rng.sample(candidates, sample_count)

            for question in chosen:
                selected.append(question)
                selected_ids.add(question["question_id"])

    # ---------------------------------------------------------
    # Step 2: If the planned distribution produces fewer than
    # 50 questions, fill the remaining slots with additional
    # unique questions from the same six databases.
    # ---------------------------------------------------------
    remaining_needed = TARGET_QUESTION_COUNT - len(selected)

    if remaining_needed > 0:
        remaining_candidates = [
            question
            for question in questions
            if question["db_id"] in DATABASES
            and question["question_id"] not in selected_ids
        ]

        if len(remaining_candidates) < remaining_needed:
            raise ValueError(
                "Not enough unique questions are available "
                "to create the requested BIRD pilot size."
            )

        additional_questions = rng.sample(
            remaining_candidates,
            remaining_needed,
        )

        selected.extend(additional_questions)
        selected_ids.update(
            question["question_id"]
            for question in additional_questions
        )

    # ---------------------------------------------------------
    # Step 3: Final safety checks.
    # ---------------------------------------------------------
    if len(selected) != TARGET_QUESTION_COUNT:
        raise ValueError(
            f"Expected exactly {TARGET_QUESTION_COUNT} questions, "
            f"but selected {len(selected)}."
        )

    if len(selected_ids) != TARGET_QUESTION_COUNT:
        raise ValueError(
            "Duplicate question IDs detected in the BIRD pilot."
        )

    selected_databases = sorted(
        {question["db_id"] for question in selected}
    )

    if not set(selected_databases).issubset(set(DATABASES)):
        raise ValueError(
            "The pilot contains a database outside the configured "
            "six-database selection."
        )

    # ---------------------------------------------------------
    # Step 4: Write selected questions.
    # ---------------------------------------------------------
    output_questions = OUTPUT_DIR / "mini_dev_sqlite_50.json"

    with output_questions.open("w", encoding="utf-8") as file:
        json.dump(
            selected,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------
    # Step 5: Copy schema metadata.
    # ---------------------------------------------------------
    shutil.copy2(
        source_tables,
        OUTPUT_DIR / "dev_tables.json",
    )

    # ---------------------------------------------------------
    # Step 6: Copy only the six selected databases.
    # ---------------------------------------------------------
    for database in selected_databases:
        source_database = source_databases / database
        destination_database = output_databases / database

        if not source_database.exists():
            raise FileNotFoundError(
                f"Source database not found: {source_database}"
            )

        shutil.copytree(
            source_database,
            destination_database,
            dirs_exist_ok=True,
        )

    # ---------------------------------------------------------
    # Step 7: Create gold SQL file.
    # ---------------------------------------------------------
    gold_sql = OUTPUT_DIR / "mini_dev_sqlite_50_gold.sql"

    with gold_sql.open("w", encoding="utf-8") as file:
        for question in selected:
            file.write(
                f"-- question_id: {question['question_id']}\n"
            )
            file.write(question["SQL"].strip())
            file.write(";\n\n")

    # ---------------------------------------------------------
    # Step 8: Print summary.
    # ---------------------------------------------------------
    print("=" * 60)
    print("BIRD Mini-Dev pilot created successfully")
    print("=" * 60)

    print(f"Selected questions : {len(selected)}")
    print(
        f"Unique question IDs: "
        f"{len(selected_ids)}"
    )

    print(f"Databases           : {len(selected_databases)}")

    print("Database names      :")
    for database in selected_databases:
        print(f"  - {database}")

    print()
    print("Difficulty distribution:")

    difficulty_counts = {}

    for question in selected:
        difficulty = question["difficulty"]
        difficulty_counts[difficulty] = (
            difficulty_counts.get(difficulty, 0) + 1
        )

    for difficulty in ["simple", "moderate", "challenging"]:
        print(
            f"  {difficulty}: "
            f"{difficulty_counts.get(difficulty, 0)}"
        )

    print()
    print(f"Output directory    : {OUTPUT_DIR.resolve()}")

    print()
    print("Files created/updated:")
    print(f"  - {output_questions}")
    print(f"  - {OUTPUT_DIR / 'mini_dev_sqlite_50_gold.sql'}")
    print(f"  - {OUTPUT_DIR / 'dev_tables.json'}")
    print(f"  - {output_databases}")


if __name__ == "__main__":
    main()