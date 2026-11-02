`transmission_timing(contact_interval; offspring, contacts, latent_period,
infectious_period)` builds the `Transition`s and the `Infectiousness` window a
case's natural history needs, from a plain generation time or from a latent
and an infectious period with a contact interval. `contacts` and `offspring`
both set how many people a case reaches, but only `offspring` keeps its mean
and dispersion exact once there is a `:recovered` to cut transmission short;
`contacts` is thinned by it instead, and `implied_R` gives the reproduction
number that thinning leaves.
