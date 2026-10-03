A `RoutedNetwork` route's `kernel` now accepts a covariate callable, a
`ContextualKernel` and a per-edge vector, each resolved per pair exactly as
`NetworkProcess` resolves its edge kernel, in both the route's targets and the
continuation drawn after a blocked contact. Previously every such form other
than a shared distribution was passed unresolved to the race and raised a
`MethodError`, so covariate and contextual kernels were unusable on a route.
