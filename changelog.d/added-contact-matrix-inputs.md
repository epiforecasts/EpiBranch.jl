`flatten_contacts` flattens a contact array stratified by several variables
(age by age by stratum by stratum) into the matrix and labels of composite
types, `mixing_rate` builds a `MixingProcess` rate from a contact matrix and
the group sizes, and `read_contacts` reads a contact matrix from a long-format
CSV file or a NetCDF file, through the new `CSV` and `NCDatasets` package
extensions. `contacts_from_table` builds the stratified contacts from a
long-format `DataFrame`.
