#basic transformations
#just trying to turn toy reaction strings into the pieces I might need later

using Statistics
using SimpleHypergraphs
using SimpleDirectedHypergraphs


#toy reactions to play around with

raw_reactions = [
    "A + B -> C + D",
    "C -> E",
    "D + E -> F",
    "F -> G + H",
    "H + A -> I"
]

println("\nRaw reaction strings:")
for r in raw_reactions
    println(r)
end


#parsing the reaction strings into reactants and products

function clean_species_name(x)
    return strip(x)
end

function parse_side(side_string)
    species = split(side_string, "+")
    return [clean_species_name(s) for s in species]
end

function parse_reaction_string(reaction_string, reaction_id)
    sides = split(reaction_string, "->")

    reactants = parse_side(sides[1])
    products = parse_side(sides[2])

    return (
        id = reaction_id,
        raw = reaction_string,
        reactants = reactants,
        products = products
    )
end

reactions = [
    parse_reaction_string(raw_reactions[i], i)
    for i in 1:length(raw_reactions)
]

println("\nParsed reactions:")
for r in reactions
    println("e", r.id, ": ", join(r.reactants, " + "), " -> ", join(r.products, " + "))
end


#get all unique species/nodes

function extract_nodes(reactions)
    node_set = Set{String}()

    for r in reactions
        for node in vcat(r.reactants, r.products)
            push!(node_set, node)
        end
    end

    return sort(collect(node_set))
end

nodes = extract_nodes(reactions)

println("\nExtracted node/species set:")
println(nodes)


#assigning integer ids to species

function build_node_mappings(nodes)
    node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
    id_to_node = Dict(i => node for (i, node) in enumerate(nodes))
    return node_to_id, id_to_node
end

node_to_id, id_to_node = build_node_mappings(nodes)

println("\nNode to ID mapping:")
println(node_to_id)

println("\nID to node mapping:")
println(id_to_node)


#converting each reaction to an id-based hyperedge

function reaction_to_id_hyperedge(reaction, node_to_id)
    reactant_ids = [node_to_id[x] for x in reaction.reactants]
    product_ids = [node_to_id[x] for x in reaction.products]

    return (
        id = reaction.id,
        reactants = reaction.reactants,
        products = reaction.products,
        reactant_ids = reactant_ids,
        product_ids = product_ids
    )
end

id_hyperedges = [
    reaction_to_id_hyperedge(r, node_to_id)
    for r in reactions
]

println("\nID-based hyperedge representation:")
for e in id_hyperedges
    println("\ne", e.id)
    println("Reactants: ", e.reactants, " => ", e.reactant_ids)
    println("Products:  ", e.products, " => ", e.product_ids)
end


#building a signed incidence matrix
#using -1 for reactants and +1 for products

function build_signed_incidence(nodes, id_hyperedges)
    H = zeros(Int, length(nodes), length(id_hyperedges))

    for (j, e) in enumerate(id_hyperedges)
        for id in e.reactant_ids
            H[id, j] = -1
        end

        for id in e.product_ids
            H[id, j] = 1
        end
    end

    return H
end

H_signed = build_signed_incidence(nodes, id_hyperedges)

println("\nSigned incidence matrix:")
println("Rows = nodes/species")
println("Columns = reactions/hyperedges")
println(H_signed)


#splitting the signed matrix into source, target, and membership versions

function source_matrix(H_signed)
    return Int.(H_signed .== -1)
end

function target_matrix(H_signed)
    return Int.(H_signed .== 1)
end

function unsigned_matrix(H_signed)
    return abs.(H_signed)
end

H_source = source_matrix(H_signed)
H_target = target_matrix(H_signed)
H_unsigned = unsigned_matrix(H_signed)

println("\nSource/reactant matrix:")
println(H_source)

println("\nTarget/product matrix:")
println(H_target)

println("\nUnsigned membership matrix:")
println(H_unsigned)


#another representation: node-hyperedge-role triples

function incidence_to_edge_list(H_signed)
    edge_list = []

    for j in 1:size(H_signed, 2)
        for i in 1:size(H_signed, 1)
            role = H_signed[i, j]

            if role != 0
                push!(
                    edge_list,
                    (
                        node_id = i,
                        hyperedge_id = j,
                        role = role
                    )
                )
            end
        end
    end

    return edge_list
end

edge_list = incidence_to_edge_list(H_signed)

println("\nEdge-list representation:")
println("(role = -1 source/reactant, role = +1 target/product)")
for item in edge_list
    println(item)
end


#quick reaction size checks

function hyperedge_sizes(id_hyperedges)
    source_sizes = [length(e.reactant_ids) for e in id_hyperedges]
    target_sizes = [length(e.product_ids) for e in id_hyperedges]
    total_sizes = source_sizes .+ target_sizes

    return source_sizes, target_sizes, total_sizes
end

source_sizes, target_sizes, total_sizes = hyperedge_sizes(id_hyperedges)

println("\nHyperedge size statistics:")
for i in 1:length(id_hyperedges)
    println(
        "e", i,
        " | source size = ", source_sizes[i],
        " | target size = ", target_sizes[i],
        " | total size = ", total_sizes[i]
    )
end

println("\nAverage source size: ", mean(source_sizes))
println("Average target size: ", mean(target_sizes))
println("Average total hyperedge size: ", mean(total_sizes))


#counting how often each node appears anywhere

function node_participation_counts(H_unsigned, nodes)
    counts = vec(sum(H_unsigned, dims = 2))

    result = Dict{String, Int}()

    for i in 1:length(nodes)
        result[nodes[i]] = counts[i]
    end

    return result
end

participation = node_participation_counts(H_unsigned, nodes)

println("\nNode participation counts:")
for node in nodes
    println(node, " appears in ", participation[node], " hyperedge(s)")
end


#source vs target participation counts

function source_target_degrees(H_source, H_target, nodes)
    source_counts = vec(sum(H_source, dims = 2))
    target_counts = vec(sum(H_target, dims = 2))

    source_degree = Dict{String, Int}()
    target_degree = Dict{String, Int}()

    for i in 1:length(nodes)
        source_degree[nodes[i]] = source_counts[i]
        target_degree[nodes[i]] = target_counts[i]
    end

    return source_degree, target_degree
end

source_degree, target_degree =
    source_target_degrees(H_source, H_target, nodes)

println("\nSource/target degree-style counts:")
for node in nodes
    println(
        node,
        " | source count = ", source_degree[node],
        " | target count = ", target_degree[node]
    )
end


#simple structural node features for now

node_feature_names = [
    "source_count",
    "target_count",
    "participation_count"
]

node_features = zeros(Float64, length(nodes), length(node_feature_names))

for i in 1:length(nodes)
    node = nodes[i]

    node_features[i, 1] = source_degree[node]
    node_features[i, 2] = target_degree[node]
    node_features[i, 3] = participation[node]
end

println("\nNode feature names:")
println(node_feature_names)

println("\nNode feature matrix:")
println(node_features)


#same idea for reaction/hyperedge features

hyperedge_feature_names = [
    "source_size",
    "target_size",
    "total_size"
]

hyperedge_features = zeros(Float64, length(id_hyperedges), length(hyperedge_feature_names))

for i in 1:length(id_hyperedges)
    hyperedge_features[i, 1] = source_sizes[i]
    hyperedge_features[i, 2] = target_sizes[i]
    hyperedge_features[i, 3] = total_sizes[i]
end

println("\nHyperedge feature names:")
println(hyperedge_feature_names)

println("\nHyperedge feature matrix:")
println(hyperedge_features)


#checking source, target, and active masks

source_masks = Dict{Int, Vector{Bool}}()
target_masks = Dict{Int, Vector{Bool}}()
active_masks = Dict{Int, Vector{Bool}}()

for e in id_hyperedges
    source_mask = falses(length(nodes))
    target_mask = falses(length(nodes))

    for id in e.reactant_ids
        source_mask[id] = true
    end

    for id in e.product_ids
        target_mask[id] = true
    end

    active_mask = source_mask .| target_mask

    source_masks[e.id] = source_mask
    target_masks[e.id] = target_mask
    active_masks[e.id] = active_mask
end

println("\nMasks from transformed representation:")
for e in id_hyperedges
    println("\ne", e.id)
    println("Source mask: ", source_masks[e.id])
    println("Target mask: ", target_masks[e.id])
    println("Active mask: ", active_masks[e.id])
end


#storing everything together so it is easier to inspect later

transformed_data = Dict(
    "nodes" => nodes,
    "node_to_id" => node_to_id,
    "id_to_node" => id_to_node,
    "raw_reactions" => raw_reactions,
    "parsed_reactions" => reactions,
    "id_hyperedges" => id_hyperedges,
    "signed_incidence" => H_signed,
    "source_matrix" => H_source,
    "target_matrix" => H_target,
    "unsigned_matrix" => H_unsigned,
    "edge_list" => edge_list,
    "node_features" => node_features,
    "hyperedge_features" => hyperedge_features,
    "source_masks" => source_masks,
    "target_masks" => target_masks,
    "active_masks" => active_masks
)

println("\nTransformed data keys:")
println(collect(keys(transformed_data)))


#quick summary

println("\nSummary:")
println("Raw reactions: ", length(raw_reactions))
println("Nodes/species: ", length(nodes))
println("Directed hyperedges/reactions: ", length(id_hyperedges))
println("Signed incidence size: ", size(H_signed))
println("Node feature matrix size: ", size(node_features))
println("Hyperedge feature matrix size: ", size(hyperedge_features))
println("Edge-list length: ", length(edge_list))