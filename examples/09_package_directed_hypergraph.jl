#09_package_directed_hypergraph.jl

#this file connects the crn parsing work to the actual package that evan mentioned: SimpleDirectedHypergraphs.jl

#in file 8, i built my own directed hypergraph-style structures using tuples and matrices
#here i am taking the same idea and converting the source/target matrices into a DirectedHypergraph object from the package.

using SimpleDirectedHypergraphs
using Statistics

raw_reactions = [
    "A + B -> C",
    "C -> D + E",
    "E + F -> G",
    "G -> H",
    "H + A -> I"
]

println("\nraw crn reactions:")

for reaction in raw_reactions
    println(reaction)
end

#small helper for removing extra spaces

function clean_species_name(name)
    return strip(name)
end

#parsing one reaction string into reactants and products

function parse_reaction(reaction_string)
    if !occursin("->", reaction_string)
        error("reaction is missing -> : $reaction_string")
    end

    left_side, right_side = split(reaction_string, "->")

    reactants = clean_species_name.(split(strip(left_side), "+"))
    products = clean_species_name.(split(strip(right_side), "+"))

    reactants = filter(x -> x != "", reactants)
    products = filter(x -> x != "", products)

    return (
        raw = reaction_string,
        reactants = reactants,
        products = products
    )
end

parsed_reactions = [parse_reaction(r) for r in raw_reactions]

println("\nparsed reactions:")

for (i, reaction) in enumerate(parsed_reactions)
    println(
        "r", i, ": ",
        join(reaction.reactants, " + "),
        " -> ",
        join(reaction.products, " + ")
    )
end

#collecting all species in the crn
#these will become the nodes of the directed hypergraph

species_set = Set{String}()

for reaction in parsed_reactions
    for species in reaction.reactants
        push!(species_set, species)
    end

    for species in reaction.products
        push!(species_set, species)
    end
end

nodes = sort(collect(species_set))

println("\nextracted species / nodes:")
println(nodes)

#mapping species names to integer ids
#the package constructor works with matrices, so integer indexing is useful

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
id_to_node = Dict(i => node for (i, node) in enumerate(nodes))

println("\nnode to id mapping:")
println(node_to_id)

println("\nid to node mapping:")
println(id_to_node)

#converting parsed reactions into source and target id lists

directed_hyperedges = []

for (i, reaction) in enumerate(parsed_reactions)
    source_ids = [node_to_id[x] for x in reaction.reactants]
    target_ids = [node_to_id[x] for x in reaction.products]

    push!(
        directed_hyperedges,
        (
            id = i,
            raw = reaction.raw,
            reactants = reaction.reactants,
            products = reaction.products,
            source_ids = source_ids,
            target_ids = target_ids
        )
    )
end

println("\nid-based directed hyperedges:")

for edge in directed_hyperedges
    println("\nr", edge.id)
    println("  source/reactants: ", edge.reactants, " -> ", edge.source_ids)
    println("  target/products:  ", edge.products, " -> ", edge.target_ids)
end

#building normal integer source and target matrices first
#rows = species
#columns = reactions

source_matrix = zeros(Int, length(nodes), length(directed_hyperedges))
target_matrix = zeros(Int, length(nodes), length(directed_hyperedges))

for edge in directed_hyperedges
    for node_id in edge.source_ids
        source_matrix[node_id, edge.id] = 1
    end

    for node_id in edge.target_ids
        target_matrix[node_id, edge.id] = 1
    end
end

println("\nsource / reactant matrix:")
println(source_matrix)

println("\ntarget / product matrix:")
println(target_matrix)

#the DirectedHypergraph constructor accepts matrices with either numbers or nothing
#i am converting 1 values to 1.0 and 0 values to nothing
#this keeps only actual membership entries in the package object

function matrix_to_package_format(M)
    converted = Matrix{Union{Nothing, Float64}}(nothing, size(M, 1), size(M, 2))

    for i in 1:size(M, 1)
        for j in 1:size(M, 2)
            if M[i, j] != 0
                converted[i, j] = Float64(M[i, j])
            end
        end
    end

    return converted
end

tail_matrix = matrix_to_package_format(source_matrix)
head_matrix = matrix_to_package_format(target_matrix)

println("\ntail matrix for DirectedHypergraph:")
println(tail_matrix)

println("\nhead matrix for DirectedHypergraph:")
println(head_matrix)

#creating the package-level directed hypergraph
#this is the main package connection in this file

dh = DirectedHypergraph(tail_matrix, head_matrix)

println("\nDirectedHypergraph object:")
println(dh)

println("\nobject type:")
println(typeof(dh))

#inspecting the object a little bit
#this is useful while learning the package structure

println("\nfields stored in the DirectedHypergraph object:")
println(fieldnames(typeof(dh)))

#signed incidence is still useful for checking and for later ml code

signed_incidence = target_matrix .- source_matrix

println("\nsigned incidence matrix:")
println(signed_incidence)

membership_matrix = abs.(signed_incidence)

println("\nunsigned membership matrix:")
println(membership_matrix)

#reconstructing reactions from the signed incidence matrix as a check

function reconstruct_reaction(H, reaction_index, id_to_node)
    reactants = String[]
    products = String[]

    for i in 1:size(H, 1)
        if H[i, reaction_index] == -1
            push!(reactants, id_to_node[i])
        elseif H[i, reaction_index] == 1
            push!(products, id_to_node[i])
        end
    end

    return reactants, products
end

println("\nreconstructed reactions from signed incidence matrix:")

for j in 1:size(signed_incidence, 2)
    reactants, products = reconstruct_reaction(signed_incidence, j, id_to_node)

    println(
        "r", j, ": ",
        join(reactants, " + "),
        " -> ",
        join(products, " + ")
    )
end

#trying package helper functions
#i am wrapping these in try/catch because i am still exploring the package api

println("\ntrying package helper: to_undirected")

try
    undirected_version = to_undirected(dh)
    println(undirected_version)
catch err
    println("to_undirected did not run here:")
    println(err)
end

println("\ntrying package helper: get_weakly_connected_components")

try
    components = get_weakly_connected_components(dh)
    println(components)
catch err
    println("get_weakly_connected_components did not run here:")
    println(err)
end

#simple structural features
#these are the same type of features used in earlier files

source_count = vec(sum(source_matrix, dims = 2))
target_count = vec(sum(target_matrix, dims = 2))
participation_count = vec(sum(membership_matrix, dims = 2))

node_features = hcat(source_count, target_count, participation_count)

println("\nnode feature names:")
println(["source_count", "target_count", "participation_count"])

println("\nnode feature matrix:")
println(node_features)

source_size = vec(sum(source_matrix, dims = 1))
target_size = vec(sum(target_matrix, dims = 1))
total_size = vec(sum(membership_matrix, dims = 1))

hyperedge_features = hcat(source_size, target_size, total_size)

println("\nhyperedge feature names:")
println(["source_size", "target_size", "total_size"])

println("\nhyperedge feature matrix:")
println(hyperedge_features)

#keeping everything together
#this object contains both my preprocessing output and the package object

package_crn_data = (
    raw_reactions = raw_reactions,
    parsed_reactions = parsed_reactions,
    nodes = nodes,
    node_to_id = node_to_id,
    id_to_node = id_to_node,
    directed_hyperedges = directed_hyperedges,
    source_matrix = source_matrix,
    target_matrix = target_matrix,
    tail_matrix = tail_matrix,
    head_matrix = head_matrix,
    directed_hypergraph = dh,
    signed_incidence = signed_incidence,
    membership_matrix = membership_matrix,
    node_features = node_features,
    hyperedge_features = hyperedge_features
)

println("\nprocessed package crn data keys:")
println(keys(package_crn_data))

println("\nsummary:")
println("number of reactions: ", length(raw_reactions))
println("number of species / nodes: ", length(nodes))
println("number of directed hyperedges: ", length(directed_hyperedges))
println("source matrix size: ", size(source_matrix))
println("target matrix size: ", size(target_matrix))
println("tail matrix size: ", size(tail_matrix))
println("head matrix size: ", size(head_matrix))
println("node feature matrix size: ", size(node_features))
println("hyperedge feature matrix size: ", size(hyperedge_features))
println("package object type: ", typeof(dh))