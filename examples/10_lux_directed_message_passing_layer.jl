#10_lux_directed_message_passing_layer.jl
#this file is a first draft of a lux-based directed message passing layer
#the main flow is source nodes -> hyperedge embeddings -> target node updates
#i am keeping this version simple so the logic is easy to check before making it more complex

using Lux
using Random
using Statistics
using SimpleDirectedHypergraphs

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

#cleaning species names

function clean_species_name(name)
    return String(strip(name))
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

#collecting species as nodes

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

println("\nspecies / nodes:")
println(nodes)

#creating node ids

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
id_to_node = Dict(i => node for (i, node) in enumerate(nodes))

println("\nnode to id mapping:")
println(node_to_id)

#creating directed hyperedges from parsed reactions

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
    println("source ids: ", edge.source_ids)
    println("target ids: ", edge.target_ids)
end

#building source and target matrices

source_matrix = zeros(Float32, length(nodes), length(directed_hyperedges))
target_matrix = zeros(Float32, length(nodes), length(directed_hyperedges))

for edge in directed_hyperedges
    for node_id in edge.source_ids
        source_matrix[node_id, edge.id] = 1.0f0
    end

    for node_id in edge.target_ids
        target_matrix[node_id, edge.id] = 1.0f0
    end
end

println("\nsource matrix:")
println(source_matrix)

println("\ntarget matrix:")
println(target_matrix)

#creating a DirectedHypergraph package object too
#this keeps this file connected to SimpleDirectedHypergraphs.jl

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

dh = DirectedHypergraph(tail_matrix, head_matrix)

println("\nDirectedHypergraph object type:")
println(typeof(dh))

#building simple node features
#these are just structural features for the first message passing test

signed_incidence = target_matrix .- source_matrix
membership_matrix = abs.(signed_incidence)

source_count = vec(sum(source_matrix, dims = 2))
target_count = vec(sum(target_matrix, dims = 2))
participation_count = vec(sum(membership_matrix, dims = 2))

node_features = Float32.(hcat(source_count, target_count, participation_count))

println("\nnode feature names:")
println(["source_count", "target_count", "participation_count"])

println("\nnode features:")
println(node_features)

#normalising features before sending them through dense layers

function normalise_columns(X)
    X_norm = copy(X)

    for j in 1:size(X, 2)
        col_mean = mean(X[:, j])
        col_std = std(X[:, j])

        if col_std == 0
            X_norm[:, j] .= 0
        else
            X_norm[:, j] .= (X[:, j] .- col_mean) ./ col_std
        end
    end

    return Float32.(X_norm)
end

node_features_norm = normalise_columns(node_features)

println("\nnormalised node features:")
println(node_features_norm)

#taking mean features over selected source nodes

function masked_mean_node_features(node_features, mask_vector)
    selected = findall(mask_vector .> 0)

    if length(selected) == 0
        return zeros(Float32, size(node_features, 2))
    end

    return vec(mean(node_features[selected, :], dims = 1))
end

#building source aggregated features for every hyperedge

function source_to_hyperedge_features(node_features, source_matrix)
    n_edges = size(source_matrix, 2)
    feature_dim = size(node_features, 2)

    edge_features = zeros(Float32, n_edges, feature_dim)

    for e in 1:n_edges
        edge_features[e, :] .= masked_mean_node_features(node_features, source_matrix[:, e])
    end

    return edge_features
end

initial_hyperedge_features = source_to_hyperedge_features(node_features_norm, source_matrix)

println("\ninitial hyperedge features from source aggregation:")
println(initial_hyperedge_features)

#setting up lux layers
#source_transform turns source aggregated features into hyperedge embeddings
#target_transform turns hyperedge embeddings into messages for target nodes

in_dim = size(node_features_norm, 2)
hidden_dim = 4

source_transform = Dense(in_dim => hidden_dim, tanh)
target_transform = Dense(hidden_dim => hidden_dim, tanh)

rng = Random.default_rng()

ps_source, st_source = Lux.setup(rng, source_transform)
ps_target, st_target = Lux.setup(rng, target_transform)

println("\nlux source transform:")
println(source_transform)

println("\nlux target transform:")
println(target_transform)

println("\nsource transform parameters:")
println(ps_source)

println("\ntarget transform parameters:")
println(ps_target)

#running directed message passing
#this is the first draft of the layer behaviour

function directed_message_passing(
    node_features,
    source_matrix,
    target_matrix,
    source_transform,
    target_transform,
    ps_source,
    st_source,
    ps_target,
    st_target
)
    n_nodes = size(node_features, 1)
    n_edges = size(source_matrix, 2)

    edge_input = source_to_hyperedge_features(node_features, source_matrix)

    edge_hidden, new_st_source =
        source_transform(edge_input', ps_source, st_source)

    edge_hidden = edge_hidden'

    edge_message, new_st_target =
        target_transform(edge_hidden', ps_target, st_target)

    edge_message = edge_message'

    node_updates = zeros(Float32, n_nodes, size(edge_message, 2))
    node_update_counts = zeros(Float32, n_nodes)

    for e in 1:n_edges
        target_nodes = findall(target_matrix[:, e] .> 0)

        for node_id in target_nodes
            node_updates[node_id, :] .+= edge_message[e, :]
            node_update_counts[node_id] += 1.0f0
        end
    end

    for node_id in 1:n_nodes
        if node_update_counts[node_id] > 0
            node_updates[node_id, :] ./= node_update_counts[node_id]
        end
    end

    return (
        hyperedge_embeddings = edge_hidden,
        node_updates = node_updates,
        st_source = new_st_source,
        st_target = new_st_target
    )
end

message_passing_output = directed_message_passing(
    node_features_norm,
    source_matrix,
    target_matrix,
    source_transform,
    target_transform,
    ps_source,
    st_source,
    ps_target,
    st_target
)

hyperedge_embeddings = message_passing_output.hyperedge_embeddings
node_updates = message_passing_output.node_updates

println("\nhyperedge embeddings:")
println(hyperedge_embeddings)

println("\nnode updates from directed message passing:")
println(node_updates)

println("\nhyperedge embedding size:")
println(size(hyperedge_embeddings))

println("\nnode update size:")
println(size(node_updates))

#showing message passing by reaction

println("\nmessage passing by reaction:")

for edge in directed_hyperedges
    println("\nr", edge.id, ": ", edge.raw)
    println("source nodes: ", edge.reactants)
    println("target nodes: ", edge.products)
    println("hyperedge embedding: ", hyperedge_embeddings[edge.id, :])
end

#showing node updates by species

println("\nnode updates by species:")

for i in 1:length(nodes)
    println(nodes[i], " update: ", node_updates[i, :])
end

#adding a small prediction head
#this is only to check that hyperedge embeddings can feed into another lux layer

prediction_head = Dense(hidden_dim => 1)

ps_head, st_head = Lux.setup(rng, prediction_head)

scores, st_head_new = prediction_head(hyperedge_embeddings', ps_head, st_head)
scores = vec(scores)

println("\ntoy reaction scores from prediction head:")
println(scores)

#creating toy labels for checking the forward pass
#label is 1 if the reaction has more than one reactant

toy_labels = Float32[
    length(edge.source_ids) > 1 ? 1.0f0 : 0.0f0
    for edge in directed_hyperedges
]

println("\ntoy labels:")
println(toy_labels)

#turning scores into probabilities

function sigmoid(x)
    return 1.0f0 / (1.0f0 + exp(-x))
end

probabilities = sigmoid.(scores)

println("\ntoy probabilities:")
println(probabilities)

predictions = Int.(probabilities .>= 0.5f0)

println("\ntoy predictions:")
println(predictions)

println("\nactual toy labels:")
println(Int.(toy_labels))

accuracy = mean(predictions .== Int.(toy_labels))

println("\ntoy accuracy:")
println(accuracy)

#storing outputs from this draft layer

lux_message_passing_data = (
    raw_reactions = raw_reactions,
    parsed_reactions = parsed_reactions,
    nodes = nodes,
    node_to_id = node_to_id,
    id_to_node = id_to_node,
    directed_hyperedges = directed_hyperedges,
    directed_hypergraph = dh,
    source_matrix = source_matrix,
    target_matrix = target_matrix,
    signed_incidence = signed_incidence,
    membership_matrix = membership_matrix,
    node_features = node_features,
    node_features_norm = node_features_norm,
    initial_hyperedge_features = initial_hyperedge_features,
    hyperedge_embeddings = hyperedge_embeddings,
    node_updates = node_updates,
    toy_scores = scores,
    toy_probabilities = probabilities,
    toy_predictions = predictions,
    toy_labels = toy_labels
)

println("\nprocessed lux message passing data keys:")
println(keys(lux_message_passing_data))

println("\nsummary:")
println("number of nodes: ", length(nodes))
println("number of reactions / hyperedges: ", length(directed_hyperedges))
println("node feature matrix size: ", size(node_features))
println("normalised node feature matrix size: ", size(node_features_norm))
println("source matrix size: ", size(source_matrix))
println("target matrix size: ", size(target_matrix))
println("hyperedge embedding size: ", size(hyperedge_embeddings))
println("node update size: ", size(node_updates))
println("package directed hypergraph type: ", typeof(dh))