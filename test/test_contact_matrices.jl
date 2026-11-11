using Distributions
using StableRNGs
using CSV
using CategoricalArrays
using DataFrames
using NCDatasets

@testset "Contact-matrix inputs" begin
    # Strata: age (2 levels) by setting (3 levels); every entry distinct.
    # Axes: respondent age, respondent setting, contact age, contact setting.
    A = reshape(Float64.(1:36), 2, 3, 2, 3)

    @testset "flatten_contacts" begin
        flat = flatten_contacts(A; names = (:age, :setting))
        @test flat isa StratifiedContacts
        @test size(flat.matrix) == (6, 6)
        @test flat.names == (:age, :setting)
        # First variable varies fastest down the types.
        @test flat.groups[1:3] == [(1, 1), (2, 1), (1, 2)]
        @test flat.labels[2] == "2:1"
        for (j, gj) in enumerate(flat.groups), (i, gi) in enumerate(flat.groups)
            @test flat.matrix[j, i] == A[gj[1], gj[2], gi[1], gi[2]]
        end

        # One variable is the matrix itself.
        M = [3.0 0.5; 1.0 2.0]
        one = flatten_contacts(M)
        @test one.matrix == M
        @test one.groups == [(1,), (2,)]

        named = flatten_contacts(
            A; names = (:age, :setting),
            levels = (["child", "adult"], ["home", "work", "school"])
        )
        @test named.labels[1] == "child:home"
        @test named.groups[4] == ("adult", "work")

        @test_throws ArgumentError flatten_contacts(rand(2, 2, 2))
        @test_throws ArgumentError flatten_contacts(rand(2, 3))
        @test_throws ArgumentError flatten_contacts(A; names = (:age,))
        @test_throws ArgumentError flatten_contacts(A; levels = ([1, 2], [1, 2]))
        @test_throws ArgumentError flatten_contacts(A; levels = ([1, 1], [1, 2, 3]))
    end

    @testset "flattened array feeds a multi-type BranchingProcess" begin
        flat = flatten_contacts(A; names = (:age, :setting))
        model = BranchingProcess(
            flat.matrix, 0.01, 2.0, R -> NegBin(R, 0.5), LogNormal(1.6, 0.5);
            type_labels = flat.labels
        )
        direct = BranchingProcess(
            flat.matrix, 0.01, 2.0, R -> NegBin(R, 0.5), LogNormal(1.6, 0.5)
        )
        @test model isa BranchingProcess
        @test direct isa BranchingProcess
    end

    @testset "mixing_rate" begin
        n = [1500, 500]
        C = [3.0 0.5; 2.0 1.0]
        rate = mixing_rate(C, n)
        # Survey layout: C[j, i] contacts of group i per group-j person, spread
        # over the n[i] people of group i.
        for j in 1:2, i in 1:2
            @test rate((j,), (i,)) ≈ C[j, i] / n[i]
            @test n[i] * rate((j,), (i,)) ≈ C[j, i]
        end
        @test_throws ArgumentError rate((3,), (1,))
        @test_throws ArgumentError mixing_rate(C, [1, 2, 3])
        @test_throws ArgumentError mixing_rate(C, [0, 5])
        @test_throws ArgumentError mixing_rate(-C, n)
        @test_throws ArgumentError mixing_rate(C, n; groups = [(1,), (1,)])

        # Named levels, several variables.
        flat = flatten_contacts(A; names = (:age, :setting))
        sizes = reshape(collect(100:100:600), 2, 3)
        r = mixing_rate(flat, sizes)
        @test r((2, 3), (1, 2)) ≈ A[2, 3, 1, 2] / sizes[1, 2]
    end

    @testset "mixing_rate gives the rate written by hand" begin
        n = [1500, 1500]
        M = [3.0 0.5; 0.5 0.5]
        band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)
        prog = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)]
        by_hand = (from, to) -> M[from[1], to[1]] / n[to[1]]
        make(rate) = ModelSpec(
            MixingProcess(; population_size = sum(n), mixing_by = (:age_band,), rate);
            progression = prog, attributes = band
        )
        a = simulate(make(by_hand); n_initial = 5, rng = StableRNG(3), max_time = 15.0)
        b = simulate(make(mixing_rate(M, n)); n_initial = 5, rng = StableRNG(3), max_time = 15.0)
        @test a.cumulative_cases == b.cumulative_cases
        @test a.cumulative_cases > 5

        # And the mean-field approximation reads the same matrix.
        mf = MixingMeanFieldProcess(make(mixing_rate(M, n)); removal_rate = 1.0)
        @test mf.contact_matrix ≈ [M[j, i] / n[i] for j in 1:2, i in 1:2]
    end

    @testset "contacts_from_table" begin
        df = DataFrame(
            part_age = repeat(["child", "adult"]; inner = 2, outer = 2),
            cont_age = repeat(["child", "adult"]; outer = 4),
            part_setting = repeat(["home", "work"]; inner = 4),
            cont_setting = repeat(["home", "work"]; inner = 4),
            contacts = Float64.(1:8)
        )
        # Rows have contact setting = respondent setting only; the rest default to 0.
        flat = contacts_from_table(
            df; strata = [:part_age => :cont_age, :part_setting => :cont_setting],
            value = :contacts, levels = (["child", "adult"], ["home", "work"])
        )
        @test flat.names == (:part_age, :part_setting)
        @test size(flat.matrix) == (4, 4)
        j = findfirst(==(("adult", "work")), flat.groups)
        i = findfirst(==(("child", "work")), flat.groups)
        @test flat.matrix[j, i] == 7.0
        @test flat.matrix[1, 4] == 0.0

        # Levels default to the distinct values in natural order.
        sorted = contacts_from_table(
            df; strata = ["part_age" => "cont_age", "part_setting" => "cont_setting"],
            value = "contacts"
        )
        @test sorted.groups[1] == ("adult", "home")

        bands = ["[15,Inf)", "[5,15)", "[0,5)"]
        natural = contacts_from_table(
            DataFrame(part = bands, cont = bands, n = [1.0, 2.0, 3.0]);
            strata = [:part => :cont], value = :n
        )
        @test natural.labels == ["[0,5)", "[5,15)", "[15,Inf)"]

        # A categorical column keeps its own level order.
        ordered = CategoricalArray(bands; levels = bands, ordered = true)
        cat = contacts_from_table(
            DataFrame(part = ordered, cont = ordered, n = [1.0, 2.0, 3.0]);
            strata = [:part => :cont], value = :n
        )
        @test cat.labels == bands

        # Composite labels join the levels with a colon.
        @test flat.labels[1] == "child:home"

        dup = vcat(df, df[1:1, :])
        @test_throws ArgumentError contacts_from_table(
            dup; strata = [:part_age => :cont_age, :part_setting => :cont_setting],
            value = :contacts
        )
        @test_throws ArgumentError contacts_from_table(
            df; strata = [:nope => :cont_age], value = :contacts
        )
        @test_throws ArgumentError contacts_from_table(
            df; strata = [:part_age => :cont_age], value = :contacts,
            levels = (["child"],)
        )
    end

    @testset "read_contacts without the reader" begin
        @test_throws ArgumentError read_contacts("contacts.xyz")
    end

    @testset "read_contacts from CSV" begin
        mktempdir() do dir
            path = joinpath(dir, "contacts.csv")
            CSV.write(
                path, DataFrame(
                    part_age = ["0-4", "0-4", "5+", "5+"],
                    cont_age = ["0-4", "5+", "0-4", "5+"],
                    contacts = [7.0, 2.0, 3.0, 5.0]
                )
            )
            flat = read_contacts(path; strata = ["part_age" => "cont_age"], value = "contacts")
            @test flat.matrix == [7.0 2.0; 3.0 5.0]
            @test flat.labels == ["0-4", "5+"]

            semi = joinpath(dir, "contacts.txt")
            CSV.write(semi, CSV.read(path, DataFrame); delim = ';')
            again = read_contacts(
                semi; delim = ';', strata = [:part_age => :cont_age], value = :contacts
            )
            @test again.matrix == flat.matrix
            @test read_contacts(
                semi; format = CSVFormat(), delim = ';',
                strata = [:part_age => :cont_age], value = :contacts
            ).matrix == flat.matrix
        end
    end

    @testset "read_contacts from NetCDF" begin
        mktempdir() do dir
            path = joinpath(dir, "contacts.nc")
            NCDataset(path, "c") do ds
                defDim(ds, "part_age", 2)
                defDim(ds, "cont_age", 2)
                defDim(ds, "part_setting", 3)
                defDim(ds, "cont_setting", 3)
                lev = defVar(ds, "part_age", Float64, ("part_age",))
                lev[:] = [10.0, 20.0]
                lev = defVar(ds, "cont_age", Float64, ("cont_age",))
                lev[:] = [10.0, 20.0]
                # Stored with the dimensions in an order unlike the one
                # `flatten_contacts` takes.
                v = defVar(
                    ds, "contacts", Float64,
                    ("cont_setting", "part_age", "part_setting", "cont_age")
                )
                v[:, :, :, :] = permutedims(A, (4, 1, 2, 3))
            end
            flat = read_contacts(
                path; variable = "contacts",
                strata = ["part_age" => "cont_age", "part_setting" => "cont_setting"]
            )
            expected = flatten_contacts(
                A; names = (:part_age, :part_setting),
                levels = ([10.0, 20.0], [1, 2, 3])
            )
            @test flat.matrix == expected.matrix
            @test flat.groups == expected.groups
            @test flat.names == (:part_age, :part_setting)

            @test_throws ArgumentError read_contacts(
                path; variable = "contacts", strata = ["part_age" => "cont_age"]
            )
            @test_throws ArgumentError read_contacts(
                path; variable = "missing", strata = ["part_age" => "cont_age"]
            )
        end
    end
end
