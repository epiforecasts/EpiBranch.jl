#!/usr/bin/env julia
# Generate the EpiBranch.jl hex sticker.
#
# Run from the logo/ directory:
#   julia --project=. make_logo.jl
#
# Outputs logo.svg and logo.png in this directory.

using Luxor
using Distributions
using Random

# Julia brand colours
const JULIA_RED = "#CB3C33"
const JULIA_GREEN = "#389826"
const JULIA_PURPLE = "#9558B2"
const JULIA_BLUE = "#4063D8"
const BG_DARK = "#1A1A2E"
const RIM_COLOUR = "#E8E8F0"
const EDGE_COLOUR = "#9DA3B4"

# Pointy-top hex sticker. R hexSticker convention: height > width.
# Scale: width 1200 → height = 1200 * 2/√3 ≈ 1386.
const HEX_WIDTH = 1200
const HEX_HEIGHT = round(Int, HEX_WIDTH * 2 / sqrt(3))
const HEX_RADIUS = HEX_HEIGHT / 2  # circumradius

"Generate a branching process realisation. Returns Vector of (id, gen, parent)."
function generate_tree(; R0 = 1.6, k = 0.7, max_gens = 4, max_nodes = 22,
        rng = Random.default_rng())
    p = k / (k + R0)
    offspring = NegativeBinomial(k, p)
    nodes = [(id = 1, gen = 0, parent = 0)]
    current = [1]
    for g in 1:max_gens
        next = Int[]
        for parent_id in current
            n_off = rand(rng, offspring)
            for _ in 1:n_off
                length(nodes) >= max_nodes && break
                push!(nodes, (id = length(nodes) + 1, gen = g, parent = parent_id))
                push!(next, length(nodes))
            end
            length(nodes) >= max_nodes && break
        end
        isempty(next) && break
        current = next
    end
    return nodes
end

function subtree_size(nodes, root_id)
    s = 1
    for n in nodes
        n.parent == root_id && (s += subtree_size(nodes, n.id))
    end
    return s
end

"""
Top-down dendrogram layout. Leaves are placed evenly along a horizontal
axis; each internal node is centred above the mean of its children. The
root ends up at depth 0, descendants at `depth × gen_height` below.
"""
function layout_dendrogram(nodes, root_id; gen_height, leaf_step)
    positions = Dict{Int, Tuple{Float64, Float64}}()
    leaf_x = Ref(0.0)
    function assign!(id, depth)
        children = [n for n in nodes if n.parent == id]
        x = if isempty(children)
            cur = leaf_x[]
            leaf_x[] += leaf_step
            cur
        else
            child_xs = Float64[]
            for c in children
                assign!(c.id, depth + 1)
                push!(child_xs, positions[c.id][1])
            end
            sum(child_xs) / length(child_xs)
        end
        positions[id] = (x, depth * gen_height)
    end
    assign!(root_id, 0)
    # Centre horizontally on x = 0.
    xs = [v[1] for v in values(positions)]
    cx = (minimum(xs) + maximum(xs)) / 2
    for k in keys(positions)
        positions[k] = (positions[k][1] - cx, positions[k][2])
    end
    return positions
end

"Find a tree that fits aesthetically inside the hex."
function pick_tree(seed_range; target_nodes = (12, 22), target_gens = (3, 4))
    for seed in seed_range
        rng = MersenneTwister(seed)
        tree = generate_tree(rng = rng)
        n = length(tree)
        max_gen = maximum(node.gen for node in tree)
        if target_nodes[1] <= n <= target_nodes[2] &&
           target_gens[1] <= max_gen <= target_gens[2]
            return tree, seed
        end
    end
    error("No seed in $(seed_range) produced a tree matching targets")
end

"Generation-coloured palette cycling through Julia brand colours."
function node_colour(gen)
    gen == 0 && return "#FFFFFF"
    palette = [JULIA_RED, JULIA_GREEN, JULIA_PURPLE, JULIA_BLUE]
    return palette[mod1(gen, length(palette))]
end

function draw_tree(tree, positions; node_radius_root = 34, node_radius = 22,
        edge_width = 5.0)
    setline(edge_width)
    setlinecap("round")
    sethue(EDGE_COLOUR)
    for n in tree
        n.parent == 0 && continue
        p = positions[n.parent]
        c = positions[n.id]
        line(Point(p[1], p[2]), Point(c[1], c[2]), :stroke)
    end
    for n in tree
        x, y = positions[n.id]
        r = n.gen == 0 ? node_radius_root : node_radius
        sethue(node_colour(n.gen))
        circle(Point(x, y), r, :fill)
        sethue(BG_DARK)
        setline(2.5)
        circle(Point(x, y), r, :stroke)
    end
end

function draw_hex_frame(radius; border_width = 20)
    # Pointy-top hex: first vertex at the top.
    vertices = [Point(radius * cos(π / 2 + i * π / 3),
                    -radius * sin(π / 2 + i * π / 3)) for i in 0:5]
    sethue(BG_DARK)
    poly(vertices, :fill, close = true)
    sethue(RIM_COLOUR)
    setline(border_width)
    poly(vertices, :stroke, close = true)
end

function draw_title(radius; text = "EpiBranch.jl", font_size = 112)
    sethue(RIM_COLOUR)
    fontface("Helvetica-Bold")
    fontsize(font_size)
    y = radius * 0.48
    textcentred(text, Point(0, y))
end

function build_logo(path_svg::AbstractString, path_png::AbstractString)
    tree, seed = pick_tree(1:300)
    @info "Selected branching tree" seed n_nodes=length(tree) max_gen=maximum(n.gen
    for n in tree)

    # Lay out and then scale to fit a target bounding box inside the hex.
    raw = layout_dendrogram(tree, 1; gen_height = 1.0, leaf_step = 1.0)
    xs = [v[1] for v in values(raw)]
    ys = [v[2] for v in values(raw)]
    x_span = maximum(xs) - minimum(xs)
    y_span = maximum(ys) - minimum(ys)

    target_width = HEX_WIDTH * 0.72
    target_height = HEX_HEIGHT * 0.40
    sx = x_span > 0 ? target_width / x_span : 1.0
    sy = y_span > 0 ? target_height / y_span : 1.0

    # Root translated so its centre sits near y = -HEX_RADIUS * 0.46,
    # leaving the bottom band of the hex for the wordmark.
    root_y_screen = -HEX_RADIUS * 0.55
    drawables = Dict(k => (v[1] * sx, v[2] * sy + root_y_screen)
    for (k, v) in raw)

    function render()
        origin()
        draw_hex_frame(HEX_RADIUS)
        draw_tree(tree, drawables)
        draw_title(HEX_RADIUS)
    end

    Drawing(HEX_WIDTH, HEX_HEIGHT, path_svg)
    render()
    finish()

    Drawing(HEX_WIDTH, HEX_HEIGHT, path_png)
    render()
    finish()

    return path_svg, path_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    here = @__DIR__
    svg, png = build_logo(joinpath(here, "logo.svg"), joinpath(here, "logo.png"))
    @info "Wrote logo" svg png
end
