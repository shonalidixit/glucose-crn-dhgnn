#gather operations
#trying out different ways of collecting node information from reactions

using Statistics
using SimpleHypergraphs
using SimpleDirectedHypergraphs


#toy reaction network

nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
id_to_node = Dict(i => node for (i, node) in enumerate(nodes))

reactions = [
    (
        id = 1,
        reactants = ["A", "B"],
        products = ["C", "D"]
    ),
    (
        id = 2,
        reactants = ["C"],
        products = ["E"]
    ),
    (
        id = 3,
        reactants = ["D", "E"],
        products = ["F"]
    ),
    (
        id = 4,
        reactants = ["F"],
        products = ["G", "H"]
    )
]

println("\nToy reactions:")
for r in reactions
    println("e", r.id, ": ", join(r.reactants, " + "), " -> ", join(r.products, " + "))
end


#node features for testing

#rows = nodes
#columns = feature values

node_features = [
    1.0  0.2  0.5;
    0.8  0.1  0.4;
    0.3  0.9  0.7;
    0.4  0.6  0.8;
    0.9  0.2  0.1;
    0.5  0.5  0.5;
    0.7  0.3  0.6;
    0.2  0.8  0.9
]

println("\nNode feature matrix:")
println(node_features)

println("\nNode feature lookup:")
for i in 1:length(nodes)
    println(nodes[i], " => ", node_features[i, :])
end


#small helper functions

function node_ids(node_names, node_to_id)
    return [node_to_id[node] for node in node_names]
end

function gather_features(feature_matrix, ids)
    return feature_matrix[ids, :]
end

function gather_reactant_features(reaction, node_to_id, feature_matrix)
    ids = node_ids(reaction.reactants, node_to_id)
    return gather_features(feature_matrix, ids)
end

function gather_product_features(reaction, node_to_id, feature_matrix)
    ids = node_ids(reaction.products, node_to_id)
    return gather_features(feature_matrix, ids)
end

function gather_active_features(reaction, node_to_id, feature_matrix)
    active_nodes = vcat(reaction.reactants, reaction.products)
    ids = node_ids(active_nodes, node_to_id)
    return gather_features(feature_matrix, ids)
end


#gather features for each reaction

println("\nGathered features per reaction:")

for r in reactions
    reactant_features = gather_reactant_features(r, node_to_id, node_features)
    product_features = gather_product_features(r, node_to_id, node_features)
    active_features = gather_active_features(r, node_to_id, node_features)

    println("\ne", r.id, ": ", join(r.reactants, " + "), " -> ", join(r.products, " + "))

    println("Reactant/source features:")
    println(reactant_features)

    println("Product/target features:")
    println(product_features)

    println("All active node features:")
    println(active_features)
end


#some simple aggregation ideas

#mean, sum, max etc.

function mean_feature(features)
    return vec(mean(features, dims = 1))
end

function sum_feature(features)
    return vec(sum(features, dims = 1))
end

function max_feature(features)
    return vec(maximum(features, dims = 1))
end

println("\nAggregated hyperedge features:")

for r in reactions
    src = gather_reactant_features(r, node_to_id, node_features)
    tgt = gather_product_features(r, node_to_id, node_features)

    src_mean = mean_feature(src)
    tgt_mean = mean_feature(tgt)

    src_sum = sum_feature(src)
    tgt_sum = sum_feature(tgt)

    direction_difference = tgt_mean .- src_mean
    concatenated = vcat(src_mean, tgt_mean)

    println("\ne", r.id)
    println("Source mean: ", src_mean)
    println("Target mean: ", tgt_mean)
    println("Source sum:  ", src_sum)
    println("Target sum:  ", tgt_sum)
    println("Target - source mean: ", direction_difference)
    println("Concatenated source/target mean: ", concatenated)
end


#trying the same thing from the incidence matrix

function build_signed_incidence(nodes, reactions, node_to_id)
    H = zeros(Int, length(nodes), length(reactions))

    for (j, r) in enumerate(reactions)
        for reactant in r.reactants
            H[node_to_id[reactant], j] = -1
        end

        for product in r.products
            H[node_to_id[product], j] = 1
        end
    end

    return H
end

H = build_signed_incidence(nodes, reactions, node_to_id)

println("\nSigned incidence matrix:")
println(H)

println("\nGathering features using incidence columns:")

for j in 1:size(H, 2)
    source_ids = findall(H[:, j] .== -1)
    target_ids = findall(H[:, j] .== 1)
    active_ids = findall(H[:, j] .!= 0)

    source_features = gather_features(node_features, source_ids)
    target_features = gather_features(node_features, target_ids)
    active_features = gather_features(node_features, active_ids)

    println("\nHyperedge e", j)
    println("Source IDs: ", source_ids)
    println("Target IDs: ", target_ids)
    println("Active IDs: ", active_ids)

    println("Source features:")
    println(source_features)

    println("Target features:")
    println(target_features)

    println("Active features:")
    println(active_features)
end


#padding source/target ids so reactions can be processed together

max_source_size = maximum(length(r.reactants) for r in reactions)
max_target_size = maximum(length(r.products) for r in reactions)

source_id_batch = zeros(Int, length(reactions), max_source_size)
target_id_batch = zeros(Int, length(reactions), max_target_size)

source_mask = falses(length(reactions), max_source_size)
target_mask = falses(length(reactions), max_target_size)

for (i, r) in enumerate(reactions)
    src_ids = node_ids(r.reactants, node_to_id)
    tgt_ids = node_ids(r.products, node_to_id)

    for (j, id) in enumerate(src_ids)
        source_id_batch[i, j] = id
        source_mask[i, j] = true
    end

    for (j, id) in enumerate(tgt_ids)
        target_id_batch[i, j] = id
        target_mask[i, j] = true
    end
end

println("\nPadded source ID batch:")
println(source_id_batch)

println("\nSource mask:")
println(source_mask)

println("\nPadded target ID batch:")
println(target_id_batch)

println("\nTarget mask:")
println(target_mask)


#turn padded ids into padded feature batches

num_reactions = length(reactions)
num_features = size(node_features, 2)

source_feature_batch = zeros(Float64, num_reactions, max_source_size, num_features)
target_feature_batch = zeros(Float64, num_reactions, max_target_size, num_features)

for i in 1:num_reactions
    for j in 1:max_source_size
        if source_mask[i, j]
            node_id = source_id_batch[i, j]
            source_feature_batch[i, j, :] = node_features[node_id, :]
        end
    end

    for j in 1:max_target_size
        if target_mask[i, j]
            node_id = target_id_batch[i, j]
            target_feature_batch[i, j, :] = node_features[node_id, :]
        end
    end
end

println("\nSource feature batch:")
println(source_feature_batch)

println("\nTarget feature batch:")
println(target_feature_batch)


#masked aggregation on padded batches

function masked_mean_feature_batch(feature_batch, mask)
    batch_size = size(feature_batch, 1)
    num_features = size(feature_batch, 3)

    output = zeros(Float64, batch_size, num_features)

    for i in 1:batch_size
        valid_count = sum(mask[i, :])

        if valid_count > 0
            for f in 1:num_features
                total = 0.0

                for j in 1:size(feature_batch, 2)
                    if mask[i, j]
                        total += feature_batch[i, j, f]
                    end
                end

                output[i, f] = total / valid_count
            end
        end
    end

    return output
end

source_mean_batch = masked_mean_feature_batch(source_feature_batch, source_mask)
target_mean_batch = masked_mean_feature_batch(target_feature_batch, target_mask)

println("\nSource mean feature batch:")
println(source_mean_batch)

println("\nTarget mean feature batch:")
println(target_mean_batch)


#building a simple reaction/hyperedge representation

hyperedge_embeddings = zeros(Float64, num_reactions, num_features * 3)

for i in 1:num_reactions
    src_mean = source_mean_batch[i, :]
    tgt_mean = target_mean_batch[i, :]
    diff = tgt_mean .- src_mean

    hyperedge_embeddings[i, :] = vcat(src_mean, tgt_mean, diff)
end

println("\nToy hyperedge embeddings:")
println(hyperedge_embeddings)


#checking which reactions each node belongs to

node_to_hyperedges = Dict(node => Int[] for node in nodes)

for r in reactions
    involved_nodes = vcat(r.reactants, r.products)

    for node in involved_nodes
        push!(node_to_hyperedges[node], r.id)
    end
end

println("\nNode-to-hyperedge gather mapping:")

for node in nodes
    println(node, " participates in hyperedges ", node_to_hyperedges[node])
end


#gather reaction embeddings for each node

println("\nGather hyperedge embeddings per node:")

for node in nodes
    connected_edges = node_to_hyperedges[node]

    if isempty(connected_edges)
        println(node, " has no connected hyperedges")
    else
        gathered = hyperedge_embeddings[connected_edges, :]
        println("\nNode ", node)
        println("Connected hyperedges: ", connected_edges)
        println("Gathered hyperedge embeddings:")
        println(gathered)
    end
end


#very simple node update experiment

#take the average of connected reaction embeddings

node_updated_features = Dict{String, Vector{Float64}}()

for node in nodes
    connected_edges = node_to_hyperedges[node]

    if isempty(connected_edges)
        node_updated_features[node] = zeros(Float64, size(hyperedge_embeddings, 2))
    else
        gathered = hyperedge_embeddings[connected_edges, :]
        node_updated_features[node] = vec(mean(gathered, dims = 1))
    end
end

println("\nToy node updates from gathered hyperedge embeddings:")

for node in nodes
    println(node, " updated representation: ", node_updated_features[node])
end


#quick summary

println("\nSummary:")
println("Number of nodes: ", length(nodes))
println("Number of hyperedges/reactions: ", length(reactions))
println("Node feature size: ", size(node_features))
println("Source feature batch size: ", size(source_feature_batch))
println("Target feature batch size: ", size(target_feature_batch))
println("Hyperedge embedding size: ", size(hyperedge_embeddings))