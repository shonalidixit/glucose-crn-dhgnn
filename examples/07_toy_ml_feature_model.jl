#toy ml feature model
#small experiment to see how reaction/hyperedge features could become model input

using Statistics
using LinearAlgebra
using SimpleHypergraphs
using SimpleDirectedHypergraphs


#toy reaction network

nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))

reactions = [
    (id = 1, reactants = ["A", "B"], products = ["C"]),
    (id = 2, reactants = ["C", "D"], products = ["E"]),
    (id = 3, reactants = ["E"], products = ["F", "G"]),
    (id = 4, reactants = ["G"], products = ["H"]),
    (id = 5, reactants = ["H", "A"], products = ["I"]),
    (id = 6, reactants = ["B"], products = ["D"]),
    (id = 7, reactants = ["D", "F"], products = ["G"]),
    (id = 8, reactants = ["I"], products = ["A", "E"])
]

println("\nToy reactions:")
for r in reactions
    println("e", r.id, ": ", join(r.reactants, " + "), " -> ", join(r.products, " + "))
end


#building incidence matrix first

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

H_source = Int.(H .== -1)
H_target = Int.(H .== 1)
H_membership = abs.(H)


#node features from structure

source_count = vec(sum(H_source, dims = 2))
target_count = vec(sum(H_target, dims = 2))
participation_count = vec(sum(H_membership, dims = 2))

node_features = hcat(source_count, target_count, participation_count)

println("\nNode features:")
println("columns = source_count, target_count, participation_count")
println(node_features)


#reaction/hyperedge features

source_size = vec(sum(H_source, dims = 1))
target_size = vec(sum(H_target, dims = 1))
total_size = vec(sum(H_membership, dims = 1))

hyperedge_features = hcat(source_size, target_size, total_size)

println("\nHyperedge features:")
println("columns = source_size, target_size, total_size")
println(hyperedge_features)


#toy label
#for now, label = 1 if the reaction has more than one reactant

labels = Int.(source_size .> 1)

println("\nToy labels:")
println(labels)


#normalising features a little bit

function normalize_columns(X)
    X_norm = zeros(Float64, size(X))

    for j in 1:size(X, 2)
        col = X[:, j]
        μ = mean(col)
        σ = std(col)

        if σ == 0
            X_norm[:, j] .= 0.0
        else
            X_norm[:, j] = (col .- μ) ./ σ
        end
    end

    return X_norm
end

X = normalize_columns(Float64.(hyperedge_features))
y = labels

println("\nNormalised hyperedge features:")
println(X)


#simple score-based classifier
#not a full ML model yet, just a first toy experiment

weights = [1.0, -0.5, 0.25]
bias = 0.0

function sigmoid(z)
    return 1.0 / (1.0 + exp(-z))
end

function predict_scores(X, weights, bias)
    scores = zeros(Float64, size(X, 1))

    for i in 1:size(X, 1)
        scores[i] = dot(X[i, :], weights) + bias
    end

    return scores
end

scores = predict_scores(X, weights, bias)
probabilities = [sigmoid(s) for s in scores]
predictions = Int.(probabilities .>= 0.5)

println("\nScores:")
println(scores)

println("\nProbabilities:")
println(probabilities)

println("\nPredictions:")
println(predictions)

println("\nActual labels:")
println(y)


#small accuracy check

function accuracy(y_true, y_pred)
    correct = 0

    for i in 1:length(y_true)
        if y_true[i] == y_pred[i]
            correct += 1
        end
    end

    return correct / length(y_true)
end

acc = accuracy(y, predictions)

println("\nToy accuracy:")
println(acc)


#trying a few different weight settings manually

weight_trials = [
    [1.0, -0.5, 0.25],
    [0.8, -0.3, 0.1],
    [1.5, -1.0, 0.2],
    [0.5, 0.0, 0.5]
]

println("\nTrying a few simple weight settings:")

for (trial_id, w) in enumerate(weight_trials)
    trial_scores = predict_scores(X, w, bias)
    trial_probs = [sigmoid(s) for s in trial_scores]
    trial_preds = Int.(trial_probs .>= 0.5)
    trial_acc = accuracy(y, trial_preds)

    println("\ntrial ", trial_id)
    println("weights: ", w)
    println("predictions: ", trial_preds)
    println("accuracy: ", trial_acc)
end


#small interpretation

println("\nInterpretation:")
println("This is only a tiny ML-style experiment.")
println("The main point is to test how hyperedge features can be created from a directed reaction network.")
println("A later version could replace this with Lux.jl layers or a proper HGNN model.")


#quick summary

println("\nSummary:")
println("Number of nodes: ", length(nodes))
println("Number of reactions/hyperedges: ", length(reactions))
println("Node feature matrix size: ", size(node_features))
println("Hyperedge feature matrix size: ", size(hyperedge_features))
println("Number of labels: ", length(labels))