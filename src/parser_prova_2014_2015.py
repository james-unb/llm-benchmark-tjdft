import re
import pathlib

import pandas as pd
import pymupdf


# Data directories (relative to the repository root)
root = pathlib.Path(__file__).resolve().parent.parent
exams_dir = root / "data" / "raw" / "provas"
txt_dir = root / "data" / "interim" / "txt"
txt_dir.mkdir(parents=True, exist_ok=True)

# Input files
names = ['prova_2014', 'prova_2015_1', 'prova_2015_2']
exams = pd.DataFrame()

for name in names:
    path = exams_dir / (name + ".pdf")
    # Extract the text from the PDF
    with pymupdf.open(path) as doc:
        content = chr(12).join(page.get_text() for page in doc)


    # Save the extracted text
    (txt_dir / (name + ".txt")).write_text(
        content,
        encoding="utf-8"
    )


    # Regex to identify each question
    question_regex = (
        r"(?ms)^QUESTÃO\s+(?P<number>\d+)\s*\n"
        r"(?P<content>.*?)(?=^QUESTÃO\s+\d+|\Z)"
    )


    # Regex to identify the options
    options_regex = (
        r"(?ms)^(?P<option>[A-E])\s*\n"
        r"(?P<text>.*?)(?=^[A-E]\s*$|\Z)"
    )


    # Extract the questions
    questions = re.finditer(question_regex, content)

    question_rows = []

    for question in questions:

        number = question.group("number")
        question_content = question.group("content")

        # Find where the first option starts
        options_start = re.search(
            r"(?m)^[A-E]\s*$",
            question_content
        )

        if options_start:
            enunciado = question_content[:options_start.start()].strip()
            options_text = question_content[options_start.start():]
        else:
            enunciado = question_content.strip()
            options_text = ""

        # Extract the options
        options = re.finditer(
            options_regex,
            options_text
        )

        record = {
            "Nquestao": number,
            "enunciado": enunciado
        }

        for option in options:
            letter = option.group("option")
            text = option.group("text").strip()

            record[letter] = text

        question_rows.append(record)


    # Build the DataFrame
    df = pd.DataFrame(question_rows)

    # Ensure column order
    df = df[
        ["Nquestao", "enunciado", "A", "B", "C", "D", "E"]
    ]

    df["questao_completa"] = (
        df["enunciado"] + "\n"
        + "A) " + df["A"] + "\n"
        + "B) " + df["B"] + "\n"
        + "C) " + df["C"] + "\n"
        + "D) " + df["D"] + "\n"
        + "E) " + df["E"]
    )

    id_prova = '_'.join(re.findall(r'\d+', name))
    df['id_prova'] = id_prova

    # Show the first questions
    print(df.head())
    print(f"Total questions extracted from {name}: {df.shape[0]}")
    exams = pd.concat([exams, df])

print(f"Total questions in the dataset: {exams.shape[0]}")
exams.to_csv(root / 'data' / 'interim' / 'provas_tjdft.csv', index=False, encoding='utf-8')
