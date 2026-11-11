Continuing a race stopped by `max_time` could leave an uninfected contact's
`infection_time` `NaN` where a straight run to the same later `max_time` kept
a finite one. Every cut, including one the race is kept to continue, marks
an undecided or already-excluded contact as though the run had ended there.
A branching process now puts the drawn time back first where it still has a
record of it, and a `contacts_of` model does the same for a contact it
created. This replaces the mark from an earlier time limit once
the final end of the race is known. A `contacts_of` model that names someone it
did not create itself with `make_contact!` keeps
no such record: the same mismatch can still show up for a person reused this
way across several cuts.
