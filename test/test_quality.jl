using Aqua

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(EpiBranch;
        ambiguities=false,  # may have legitimate ambiguities with Distributions
        piracies=false,     # we extend Distributions.fit/loglikelihood
        deps_compat=(ignore=[:Dates, :Random],),
    )
end
