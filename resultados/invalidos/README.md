# Invalid results — do not use in metrics

The two CSVs in this folder were produced with the old merge based only on
`Nquestao`. Since different exams repeat question numbers, each file has 900 rows
for 300 questions, with incorrect associations between exams and answers.

Their contents were preserved in full only as history. **They must not be used to
compute accuracy or other metrics**, nor as a source for resuming. No attempt was
made to recover the old associations.

They will be replaced by new runs of the fixed benchmark, merged by `id_prova` and
`Nquestao`, with `one_to_one` validation. No new run was made as part of this fix.
