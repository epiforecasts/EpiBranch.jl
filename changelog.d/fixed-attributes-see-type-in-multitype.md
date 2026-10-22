In a multi-type `BranchingProcess`, an `attributes` function now sees a
case's actual type: for a new contact, the type its parent's offspring draw
gave it, and for an index case, the type just drawn for it. Until now every
case's type was set after `attributes` ran, so an attribute that should
depend on type (such as a number of contacts that differs between age
groups) saw every case, index or not, as type 1.
