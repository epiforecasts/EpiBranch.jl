module EpiBranchNCDatasetsExt

# Reads a stratified contact array from a NetCDF file. The array's axes are
# named in the file and are matched by name to the respondent and contact axes
# of each stratifying variable.

using EpiBranch: EpiBranch, NetCDFFormat, flatten_contacts
using NCDatasets: NCDatasets, NCDataset

function EpiBranch.read_contacts(
        ::NetCDFFormat, path::AbstractString;
        variable, strata, names = nothing, levels = nothing
    )
    pairs = [String(p.first) => String(p.second) for p in strata]
    k = length(pairs)
    k >= 1 || throw(ArgumentError("strata needs at least one pair of dimension names"))
    return NCDataset(path) do ds
        haskey(ds, String(variable)) || throw(
            ArgumentError("$(repr(path)) has no variable $(repr(String(variable)))")
        )
        var = ds[String(variable)]
        dims = collect(NCDatasets.dimnames(var))
        wanted = [first.(pairs); last.(pairs)]
        sort(dims) == sort(wanted) || throw(
            ArgumentError(
                "variable $(repr(String(variable))) has dimensions $(dims) but " *
                    "`strata` names $(wanted)"
            )
        )
        raw = Array(var)
        # The arrays `flatten_contacts` takes have a respondent then a contact
        # axis per variable; a NetCDF variable has its dimensions in any order.
        order = [findfirst(==(d), dims) for d in vec(permutedims(hcat(first.(pairs), last.(pairs))))]
        array = permutedims(raw, order)
        if levels === nothing
            levels = ntuple(k) do v
                r, c = pairs[v]
                _levels(ds, r, c, size(array, 2v - 1))
            end
        end
        flatten_contacts(
            array; names = something(names, ntuple(v -> Symbol(first(pairs[v])), k)),
            levels
        )
    end
end

# A coordinate variable named like the dimension gives the levels; a variable
# with no coordinate variable has the positions 1:n. A respondent and a contact
# axis carry the same levels, so the respondent's are used and the contact's
# checked against them when it has any.
function _levels(ds, respondent, contact, n)
    r = haskey(ds, respondent) ? collect(ds[respondent][:]) : nothing
    c = haskey(ds, contact) ? collect(ds[contact][:]) : nothing
    if r !== nothing && c !== nothing && r != c
        throw(
            ArgumentError(
                "coordinates of $(repr(respondent)) and $(repr(contact)) differ"
            )
        )
    end
    found = something(r, c, collect(1:n))
    return found
end

end
