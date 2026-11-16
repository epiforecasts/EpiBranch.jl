module EpiBranchCSVExt

# Reads a long-format contact matrix from a delimited text file. The table is
# parsed here; turning it into composite types is `contacts_from_table`.

using EpiBranch: EpiBranch, CSVFormat, contacts_from_table
using CSV: CSV
using DataFrames: DataFrame

function EpiBranch.read_contacts(
        ::CSVFormat, path::AbstractString;
        delim = ',', strata, value, kwargs...
    )
    df = CSV.read(path, DataFrame; delim)
    return contacts_from_table(df; strata, value, kwargs...)
end

end
