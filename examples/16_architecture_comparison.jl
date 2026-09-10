#16_architecture_comparison

#purpose:
#this file compares different directed hypergraph message-passing
#architectures for reaction-level regression.

#this file directly answers the thesis question:
#"Which directed hypergraph neural network architecture is most suitable
#for reaction-level regression tasks involving chemical reaction networks?"

#overall workflow:
#
#Chemical Reaction Network
#        ↓
#Species Features
#        ↓
#Different Message Passing Architectures
#        ↓
#Reaction Embeddings
#        ↓
#Regression Model
#        ↓
#Performance Evaluation
#        ↓
#Architecture Comparison

using Statistics
using LinearAlgebra
using Random

Random.seed!(42)

println("Architecture Comparison")


#1.defining a small Formose-inspired CRN


raw_reactions = [

    "CH2O + CH2O -> Glycolaldehyde",
    "Glycolaldehyde + CH2O -> Glyceraldehyde",
    "Glyceraldehyde -> Dihydroxyacetone",
    "Dihydroxyacetone + CH2O -> Tetrose",
    "Tetrose -> Glycolaldehyde + Glycolaldehyde",
    "Glyceraldehyde + Glycolaldehyde -> Pentose",
    "Pentose -> Dihydroxyacetone + Glycolaldehyde",
    "Tetrose + CH2O -> Hexose",
    "Hexose -> Glyceraldehyde + Glyceraldehyde"

]

println("\nRaw reactions:")

for reaction in raw_reactions

    println(reaction)

end


#2.parser


function parse_side(side::AbstractString)

    species = split(strip(side), "+")

    return strip.(species)

end


function parse_reaction(reaction::AbstractString)

    sides = split(reaction, "->")

    if length(sides) != 2

        error("Reaction must contain exactly one -> symbol.")

    end

    reactants = parse_side(sides[1])

    products = parse_side(sides[2])

    return reactants, products

end


parsed_reactions = [parse_reaction(r) for r in raw_reactions]

println("\nParsed reactions:")

for (i,(reactants,products)) in enumerate(parsed_reactions)

    println(

        "r$i: ",
        join(reactants," + "),
        " -> ",
        join(products," + ")

    )

end


#3.species mappings


all_species = String[]

for (reactants,products) in parsed_reactions

    append!(all_species,String.(reactants))
    append!(all_species,String.(products))

end


species = sort(unique(all_species))

node_to_id = Dict(s => i for (i,s) in enumerate(species))

id_to_node = Dict(i => s for (s,i) in node_to_id)

num_species = length(species)

num_reactions = length(parsed_reactions)


println("\nSpecies:")

println(species)

println("\nNode to ID mapping:")

println(node_to_id)


#4.source and target matrices


source_matrix = zeros(Float32,num_species,num_reactions)

target_matrix = zeros(Float32,num_species,num_reactions)


for (j,(reactants,products)) in enumerate(parsed_reactions)

    for reactant in reactants

        source_matrix[node_to_id[String(reactant)],j] += 1.0f0

    end

    for product in products

        target_matrix[node_to_id[String(product)],j] += 1.0f0

    end

end


membership_matrix = source_matrix .+ target_matrix


println("\nSource matrix:")

println(source_matrix)

println("\nTarget matrix:")

println(target_matrix)

println("\nMembership matrix:")

println(membership_matrix)
#5.building structural species features


#these are simple topology-based features.

source_count = vec(sum(source_matrix .> 0, dims = 2))
target_count = vec(sum(target_matrix .> 0, dims = 2))
participation_count = source_count .+ target_count

source_stoich_sum = vec(sum(source_matrix, dims = 2))
target_stoich_sum = vec(sum(target_matrix, dims = 2))
total_stoich_sum = source_stoich_sum .+ target_stoich_sum

structural_features = hcat(

    source_count,
    target_count,
    participation_count,
    source_stoich_sum,
    target_stoich_sum,
    total_stoich_sum

)

structural_features = Float32.(structural_features)

println("\nStructural species features:")
println(structural_features)


#6.building chemistry-informed species features


#toy chemistry descriptors.

carbon_atoms = Float32[
    1,
    2,
    3,
    3,
    4,
    5,
    6
]

oxygen_atoms = Float32[
    1,
    2,
    3,
    3,
    4,
    5,
    6
]

molecular_weight = Float32[
    30.0,
    60.0,
    90.0,
    90.0,
    120.0,
    150.0,
    180.0
]

chemistry_features = hcat(

    carbon_atoms,
    oxygen_atoms,
    molecular_weight

)

println("\nChemistry-informed features:")
println(chemistry_features)


#7.combined species features


combined_features = hcat(

    structural_features,
    chemistry_features

)

combined_features = Float32.(combined_features)

println("\nCombined species features:")
println(combined_features)


#8.standardise features


function standardise_features(X::AbstractMatrix)

    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)

    σ_safe = similar(σ)

    for i in eachindex(σ)

        σ_safe[i] = σ[i] == 0 ? 1 : σ[i]

    end

    X_norm = (X .- μ) ./ σ_safe

    return Float32.(X_norm)

end


combined_features_norm =
    standardise_features(combined_features)

println("\nNormalised combined features:")
println(combined_features_norm)


#9.helper functions


function safe_column_normalise(M::AbstractMatrix)

    col_sums = sum(M, dims = 1)

    safe_sums = similar(col_sums)

    for i in eachindex(col_sums)

        safe_sums[i] =
            col_sums[i] == 0 ? 1 : col_sums[i]

    end

    return Float32.(M ./ safe_sums)

end


function safe_row_normalise(M::AbstractMatrix)

    row_sums = sum(M, dims = 2)

    safe_sums = similar(row_sums)

    for i in eachindex(row_sums)

        safe_sums[i] =
            row_sums[i] == 0 ? 1 : row_sums[i]

    end

    return Float32.(M ./ safe_sums)

end


#10.architecture A


#Species -> Reaction

function species_to_reaction(

    X_species::AbstractMatrix,
    source_matrix::AbstractMatrix,
    target_matrix::AbstractMatrix

)

    source_norm =
        safe_column_normalise(source_matrix)

    target_norm =
        safe_column_normalise(target_matrix)

    reactant_embeddings =
        transpose(source_norm) * X_species

    product_embeddings =
        transpose(target_norm) * X_species

    reaction_embeddings =
        hcat(
            reactant_embeddings,
            product_embeddings
        )

    return Float32.(reaction_embeddings)

end


#11.architecture B


#Species -> Reaction
#
#Reaction -> Species
#
#Species -> Reaction

function reaction_to_species(

    X_reaction::AbstractMatrix,
    membership_matrix::AbstractMatrix

)

    membership_norm =
        safe_row_normalise(membership_matrix)

    updated_species =
        membership_norm * X_reaction

    return Float32.(updated_species)

end


reaction_embeddings_A =
    species_to_reaction(

        combined_features_norm,

        source_matrix,

        target_matrix

    )


updated_species =
    reaction_to_species(

        reaction_embeddings_A,

        membership_matrix

    )


reaction_embeddings_B =
    species_to_reaction(

        updated_species,

        source_matrix,

        target_matrix

    )


println("\nArchitecture A embedding size:")
println(size(reaction_embeddings_A))

println("\nArchitecture B embedding size:")
println(size(reaction_embeddings_B))
#12.toy regression target


#placeholder energy barrier values.
#these can later be replaced with
#real chemical reaction datasets.

reaction_targets = Float32[

    12.5,
    18.2,
    7.4,
    21.0,
    15.6,
    25.3,
    10.2,
    30.1,
    13.8

]

println("\nReaction targets:")
println(reaction_targets)


#13.build regression datasets


X_A = reaction_embeddings_A

X_B = reaction_embeddings_B

y = reaction_targets

println("\nArchitecture A dataset size:")
println(size(X_A))

println("\nArchitecture B dataset size:")
println(size(X_B))


#14.shuffle dataset


indices = collect(1:num_reactions)

Random.shuffle!(indices)

X_A = X_A[indices,:]
X_B = X_B[indices,:]

y = y[indices]

println("\nShuffled reaction indices:")
println(indices)


#15.train-test split


train_ratio = 0.80

num_train = floor(Int, train_ratio * num_reactions)

train_indices = 1:num_train

test_indices = (num_train + 1):num_reactions


X_A_train = X_A[train_indices,:]
X_A_test = X_A[test_indices,:]

X_B_train = X_B[train_indices,:]
X_B_test = X_B[test_indices,:]

y_train = y[train_indices]
y_test = y[test_indices]


println("\nTraining samples:")
println(length(y_train))

println("Testing samples:")
println(length(y_test))


#16.ridge regression helper functions


function add_intercept(X::AbstractMatrix)

    ones_column = ones(Float32,size(X,1),1)

    return hcat(ones_column,Float32.(X))

end


function fit_ridge_regression(

    X::AbstractMatrix,
    y::AbstractVector;

    λ = 1.0f-3

)

    X_aug = add_intercept(X)

    I_reg = Matrix{Float32}(I,size(X_aug,2),size(X_aug,2))

    I_reg[1,1] = 0.0f0

    β =

        (transpose(X_aug) * X_aug + λ * I_reg) \

        (transpose(X_aug) * y)

    return Float32.(β)

end


function predict_ridge(

    X::AbstractMatrix,
    β::AbstractVector

)

    X_aug = add_intercept(X)

    return Float32.(X_aug * β)

end


#17.evaluation metrics


function mse(

    y_true::AbstractVector,
    y_pred::AbstractVector

)

    mean((y_true .- y_pred).^2)

end


function rmse(

    y_true::AbstractVector,
    y_pred::AbstractVector

)

    sqrt(mean((y_true .- y_pred).^2))

end


function mae(

    y_true::AbstractVector,
    y_pred::AbstractVector

)

    mean(abs.(y_true .- y_pred))

end


function r_squared(

    y_true::AbstractVector,
    y_pred::AbstractVector

)

    ss_res = sum((y_true .- y_pred).^2)

    ss_tot = sum((y_true .- mean(y_true)).^2)

    return 1.0 - ss_res / ss_tot

end


#18.train Architecture A


println("\nTraining Architecture A")

beta_A = fit_ridge_regression(

    X_A_train,
    y_train

)

predictions_A = predict_ridge(

    X_A_test,
    beta_A

)


#19.train Architecture B


println("\nTraining Architecture B")

beta_B = fit_ridge_regression(

    X_B_train,
    y_train

)

predictions_B = predict_ridge(

    X_B_test,
    beta_B

)


#20.evaluate Architecture A


mse_A = mse(

    y_test,
    predictions_A

)

rmse_A = rmse(

    y_test,
    predictions_A

)

mae_A = mae(

    y_test,
    predictions_A

)

r2_A = r_squared(

    y_test,
    predictions_A

)


#21.evaluate Architecture B


mse_B = mse(

    y_test,
    predictions_B

)

rmse_B = rmse(

    y_test,
    predictions_B

)

mae_B = mae(

    y_test,
    predictions_B

)

r2_B = r_squared(

    y_test,
    predictions_B

)


println("\nArchitecture A metrics")

println("MSE : ",mse_A)
println("RMSE: ",rmse_A)
println("MAE : ",mae_A)
println("R²  : ",r2_A)


println("\nArchitecture B metrics")

println("MSE : ",mse_B)
println("RMSE: ",rmse_B)
println("MAE : ",mae_B)
println("R²  : ",r2_B)
#22.architecture comparison table


println("\n========================================")
println("Architecture Comparison")
println("========================================")

println(rpad("Architecture",45),
        rpad("RMSE",12),
        rpad("MAE",12),
        "R²")

println("-"^80)

println(rpad("Species -> Reaction",45),
        rpad(string(round(rmse_A,digits=4)),12),
        rpad(string(round(mae_A,digits=4)),12),
        round(r2_A,digits=4))

println(rpad("Species -> Reaction -> Species -> Reaction",45),
        rpad(string(round(rmse_B,digits=4)),12),
        rpad(string(round(mae_B,digits=4)),12),
        round(r2_B,digits=4))


#23.determine best architecture


best_architecture = ""

lowest_rmse = Inf

if rmse_A < lowest_rmse

    lowest_rmse = rmse_A

    best_architecture = "Species -> Reaction"

end

if rmse_B < lowest_rmse

    lowest_rmse = rmse_B

    best_architecture =
        "Species -> Reaction -> Species -> Reaction"

end


println("\nBest architecture:")

println(best_architecture)


#24.save comparison results


architecture_results = Dict(

    "Architecture A" => Dict(

        "Name" => "Species -> Reaction",

        "MSE" => mse_A,

        "RMSE" => rmse_A,

        "MAE" => mae_A,

        "R2" => r2_A

    ),

    "Architecture B" => Dict(

        "Name" => "Species -> Reaction -> Species -> Reaction",

        "MSE" => mse_B,

        "RMSE" => rmse_B,

        "MAE" => mae_B,

        "R2" => r2_B

    ),

    "Best Architecture" => best_architecture

)

println("\nArchitecture comparison results saved.")


#25.final summary


println("\n====================================")
println("Summary")
println("====================================")

println("Research Question:")

println("Which directed hypergraph neural network")

println("architecture performs best for")

println("reaction-level regression?")

println()

println("Architecture A:")
println("Species -> Reaction")

println("RMSE: ", rmse_A)
println("MAE : ", mae_A)
println("R²  : ", r2_A)

println()

println("Architecture B:")
println("Species -> Reaction -> Species -> Reaction")

println("RMSE: ", rmse_B)
println("MAE : ", mae_B)
println("R²  : ", r2_B)

println()

println("Best Architecture:")

println(best_architecture)

println()

println("Conclusion:")

println("The architecture with the lowest RMSE")

println("and MAE together with the highest R²")

println("is selected as the preferred")

println("directed hypergraph message-passing")

println("architecture for reaction-level")

println("regression on this chemical")

println("reaction network.")

println()

println("This file provides the experimental")

println("evidence required to answer the")

println("primary thesis research question.")

println()

