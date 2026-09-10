#11_bz_crn_case_study.jl
#this file applies the crn parser and lux message passing pipeline to a small bz-inspired crn case study
#the goal is to move from toy reactions to a more chemistry-style reaction network
#this is still a simplified case study, but it checks whether the full workflow works on a bz-style crn

using Lux
using Random
using Statistics
using SimpleDirectedHypergraphs

raw_reactions = [
    "BrO3 + Br -> HBrO2",
    "HBrO2 + Br -> HOBr",
    "BrO3 + HBrO2 -> BrO2 + HOBr",
    "BrO2 + Ce3 -> Ce4 + HBrO2",
    "Ce4 + MalonicAcid -> Ce3 + Br",
    "HBrO2 + HBrO2 -> BrO3 + HOBr",
    "HOBr + Br -> Br2",
    "Br2 + MalonicAcid -> BrMalonicAcid + Br"
]

println("\nbz-inspired crn reactions:")
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

#collecting all species as nodes

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

println("\nid to node mapping:")
println(id_to_node)

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
    println("source/reactants: ", edge.reactants, " -> ", edge.source_ids)
    println("target/products: ", edge.products, " -> ", edge.target_ids)
end

#building source and target matrices
#rows are species and columns are reactions

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

println("\nsource / reactant matrix:")
println(source_matrix)

println("\ntarget / product matrix:")
println(target_matrix)

#converting matrices into the format used by SimpleDirectedHypergraphs.jl

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

bz_dh = DirectedHypergraph(tail_matrix, head_matrix)

println("\nDirectedHypergraph object type:")
println(typeof(bz_dh))

#creating incidence and membership matrices

signed_incidence = target_matrix .- source_matrix
membership_matrix = abs.(signed_incidence)

println("\nsigned incidence matrix:")
println(signed_incidence)

println("\nunsigned membership matrix:")
println(membership_matrix)

#reconstructing reactions from the signed incidence matrix
#this is a check that the matrix representation still matches the reaction list

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

#trying a package helper function
#this checks that the package object can be used for graph-style analysis

println("\nweakly connected components:")

try
    components = get_weakly_connected_components(bz_dh)
    println(components)
catch err
    println("could not compute weakly connected components:")
    println(err)
end

#building structural node features
#these are simple features for the first message passing test

source_count = vec(sum(source_matrix, dims = 2))
target_count = vec(sum(target_matrix, dims = 2))
participation_count = vec(sum(membership_matrix, dims = 2))

node_features = Float32.(hcat(source_count, target_count, participation_count))

println("\nnode feature names:")
println(["source_count", "target_count", "participation_count"])

println("\nnode features:")
println(node_features)

#building hyperedge features

source_size = vec(sum(source_matrix, dims = 1))
target_size = vec(sum(target_matrix, dims = 1))
total_size = vec(sum(membership_matrix, dims = 1))

hyperedge_features = Float32.(hcat(source_size, target_size, total_size))

println("\nhyperedge feature names:")
println(["source_size", "target_size", "total_size"])

println("\nhyperedge features:")
println(hyperedge_features)

#normalising node features before using them in lux layers

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

#taking the mean of selected source node features

function masked_mean_node_features(node_features, mask_vector)
    selected = findall(mask_vector .> 0)

    if length(selected) == 0
        return zeros(Float32, size(node_features, 2))
    end

    return vec(mean(node_features[selected, :], dims = 1))
end

#creating source aggregated features for every reaction

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

#setting up a simple lux message passing draft
#source_transform creates reaction embeddings from source node information
#target_transform creates messages from reaction embeddings

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

#running directed message passing

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

#showing the message passing result reaction by reaction

println("\nmessage passing by reaction:")

for edge in directed_hyperedges
    println("\nr", edge.id, ": ", edge.raw)
    println("source nodes: ", edge.reactants)
    println("target nodes: ", edge.products)
    println("hyperedge embedding: ", hyperedge_embeddings[edge.id, :])
end

#showing the node updates species by species

println("\nnode updates by species:")

for i in 1:length(nodes)
    println(nodes[i], " update: ", node_updates[i, :])
end

#adding a tiny prediction head
#this is not the final learning task
#it only checks that reaction embeddings can feed into another lux layer

prediction_head = Dense(hidden_dim => 1)

ps_head, st_head = Lux.setup(rng, prediction_head)

scores, st_head_new = prediction_head(hyperedge_embeddings', ps_head, st_head)
scores = vec(scores)

println("\ntoy reaction scores:")
println(scores)

#creating simple toy labels for testing
#label is 1 if a reaction has more than one reactant

toy_labels = Float32[
    length(edge.source_ids) > 1 ? 1.0f0 : 0.0f0
    for edge in directed_hyperedges
]

println("\ntoy labels:")
println(toy_labels)

function sigmoid(x)
    return 1.0f0 / (1.0f0 + exp(-x))
end

probabilities = sigmoid.(scores)
predictions = Int.(probabilities .>= 0.5f0)

println("\ntoy probabilities:")
println(probabilities)

println("\ntoy predictions:")
println(predictions)

println("\nactual toy labels:")
println(Int.(toy_labels))

accuracy = mean(predictions .== Int.(toy_labels))

println("\ntoy accuracy:")
println(accuracy)

#storing the bz case study data

bz_case_study_data = (
    case_study_name = "bz-inspired crn",
    raw_reactions = raw_reactions,
    parsed_reactions = parsed_reactions,
    nodes = nodes,
    node_to_id = node_to_id,
    id_to_node = id_to_node,
    directed_hyperedges = directed_hyperedges,
    directed_hypergraph = bz_dh,
    source_matrix = source_matrix,
    target_matrix = target_matrix,
    signed_incidence = signed_incidence,
    membership_matrix = membership_matrix,
    node_features = node_features,
    hyperedge_features = hyperedge_features,
    node_features_norm = node_features_norm,
    initial_hyperedge_features = initial_hyperedge_features,
    hyperedge_embeddings = hyperedge_embeddings,
    node_updates = node_updates,
    toy_scores = scores,
    toy_probabilities = probabilities,
    toy_predictions = predictions,
    toy_labels = toy_labels
)

println("\nprocessed bz case study data keys:")
println(keys(bz_case_study_data))

println("\nsummary:")
println("case study: ", bz_case_study_data.case_study_name)
println("number of reactions: ", length(raw_reactions))
println("number of species / nodes: ", length(nodes))
println("number of directed hyperedges: ", length(directed_hyperedges))
println("source matrix size: ", size(source_matrix))
println("target matrix size: ", size(target_matrix))
println("signed incidence matrix size: ", size(signed_incidence))
println("node feature matrix size: ", size(node_features))
println("hyperedge feature matrix size: ", size(hyperedge_features))
println("hyperedge embedding size: ", size(hyperedge_embeddings))
println("node update size: ", size(node_updates))
println("package directed hypergraph type: ", typeof(bz_dh))