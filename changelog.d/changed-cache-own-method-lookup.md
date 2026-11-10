Simulating many small outbreaks takes less time: whether an intervention
defines a hook itself is looked up once and remembered until a new method is
defined, instead of being searched for on every run. A hook defined after the
package is loaded is still found.
