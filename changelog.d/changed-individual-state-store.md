Simulations run faster: about three times as fast with isolation and contact
tracing, and 1.3 to 1.6 times as fast without interventions. The gain grows
with the size of the outbreak. An `Individual`'s `state` is now an
`EpiBranch.IndividualState`. It is read and written as a dictionary, as
before. Code that requires it to be a `Dict{Symbol, Any}` specifically needs
updating.
