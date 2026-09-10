#13 species feature engineering 

#purpose:
#build species-level feature engineering for chemical reaction networks represented as directed hypergraphs.

#this file supports the thesis question:
#which structural and chemistry-informed features improve reaction-level regression performance?

using Statistics
using LinearAlgebra

println("Species Feature Engineering")


#1. defining a small Formose-inspired CRN


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

println("Raw reactions:")
for r in raw_reactions
    println(r)
end

#2.basic reaction parser

#abstractString avoids errors with SubString values returned by split().

function parse_side(side::AbstractString)
    species = split(strip(side), "+")
    return strip.(species)
end

function parse_reaction(reaction::AbstractString)
    sides = split(reaction, "->")

    if length(sides) != 2
        error("Reaction must contain exactly one -> symbol: $reaction")
    end

    reactants = parse_side(sides[1])
    products = parse_side(sides[2])

    return reactants, products
end

parsed_reactions = [parse_reaction(r) for r in raw_reactions]

println("\nParsed reactions:")
for (i, (reactants, products)) in enumerate(parsed_reactions)
    println("r$i: ", join(reactants, " + "), " -> ", join(products, " + "))
end


#3.species / node mappings


all_species = String[]

for (reactants, products) in parsed_reactions
    append!(all_species, String.(reactants))
    append!(all_species, String.(products))
end

species = sort(unique(all_species))

node_to_id = Dict(s => i for (i, s) in enumerate(species))
id_to_node = Dict(i => s for (s, i) in node_to_id)

num_species = length(species)
num_reactions = length(parsed_reactions)

println("\nSpecies / nodes:")
println(species)

println("\nNode to ID mapping:")
println(node_to_id)


#4.build source and target matrices


#source_matrix[i, j] = number of times species i appears as a reactant in reaction j
#target_matrix[i, j] = number of times species i appears as a product in reaction j

#this preserves repeated species such as:
#CH2O + CH2O -> Glycolaldehyde

#here source_matrix[CH2O, r1] = 2.0

source_matrix = zeros(Float32, num_species, num_reactions)
target_matrix = zeros(Float32, num_species, num_reactions)

for (j, (reactants, products)) in enumerate(parsed_reactions)
    for r in reactants
        source_matrix[node_to_id[String(r)], j] += 1.0f0
    end

    for p in products
        target_matrix[node_to_id[String(p)], j] += 1.0f0
    end
end

println("\nSource / reactant matrix:")
println(source_matrix)

println("\nTarget / product matrix:")
println(target_matrix)


#5.structural species features


#these features describe the role of each species in the reaction network
#they are simple baseline features that can later be compared against chemistry-informed descriptors

source_count = vec(sum(source_matrix .> 0, dims = 2))
target_count = vec(sum(target_matrix .> 0, dims = 2))
participation_count = source_count .+ target_count

#stoichiometry-aware counts
source_stoich_sum = vec(sum(source_matrix, dims = 2))
target_stoich_sum = vec(sum(target_matrix, dims = 2))
total_stoich_sum = source_stoich_sum .+ target_stoich_sum

function neighbouring_species(species_id::Int, parsed_reactions, node_to_id)
    neighbours = Set{Int}()

    for (reactants, products) in parsed_reactions
        reaction_species = vcat(String.(reactants), String.(products))
        ids = [node_to_id[s] for s in reaction_species]

        if species_id in ids
            for id in ids
                if id != species_id
                    push!(neighbours, id)
                end
            end
        end
    end

    return length(neighbours)
end

neighbour_count = Float32[
    neighbouring_species(i, parsed_reactions, node_to_id)
    for i in 1:num_species
]

reaction_count = participation_count

structural_feature_names = [
    "source_count",
    "target_count",
    "participation_count",
    "source_stoich_sum",
    "target_stoich_sum",
    "total_stoich_sum",
    "neighbour_count",
    "reaction_count"
]

structural_features = hcat(
    source_count,
    target_count,
    participation_count,
    source_stoich_sum,
    target_stoich_sum,
    total_stoich_sum,
    neighbour_count,
    reaction_count
)

structural_features = Float32.(structural_features)

println("\nStructural feature names:")
println(structural_feature_names)

println("\nStructural species features:")
println(structural_features)


#6.chemistry-informed species features


#in a complete version, these could come from molecular descriptors, molecular fingerprints, or chemistry ML tools

#here they are manually added to demonstrate how chemical information can be attached to species nodes.

chemical_feature_names = [
    "approx_molecular_weight",
    "num_carbon",
    "num_hydrogen",
    "num_oxygen",
    "num_atoms",
    "formal_charge"
]

chemical_features_dict = Dict(
    "CH2O" => Float32[30.03, 1, 2, 1, 4, 0],
    "Glycolaldehyde" => Float32[60.05, 2, 4, 2, 8, 0],
    "Glyceraldehyde" => Float32[90.08, 3, 6, 3, 12, 0],
    "Dihydroxyacetone" => Float32[90.08, 3, 6, 3, 12, 0],
    "Tetrose" => Float32[120.10, 4, 8, 4, 16, 0],
    "Pentose" => Float32[150.13, 5, 10, 5, 20, 0],
    "Hexose" => Float32[180.16, 6, 12, 6, 24, 0]
)

chemical_features = zeros(Float32, num_species, length(chemical_feature_names))

for (i, s) in enumerate(species)
    if haskey(chemical_features_dict, s)
        chemical_features[i, :] .= chemical_features_dict[s]
    else
        @warn "No chemistry-informed features found for species $s. Using zeros."
    end
end

println("\nChemistry-informed feature names:")
println(chemical_feature_names)

println("\nChemistry-informed species features:")
println(chemical_features)


#7.combine feature sets


combined_feature_names = vcat(structural_feature_names, chemical_feature_names)
combined_features = hcat(structural_features, chemical_features)

println("\nCombined feature names:")
println(combined_feature_names)

println("\nCombined species features:")
println(combined_features)


#8.feature normalisation


#standardise each feature column:
#x_norm = (x - mean) / standard deviation

#if a column has zero variance, use standard deviation = 1 to avoid division by zero.

function standardise_features(X::AbstractMatrix)
    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)

    σ_safe = similar(σ)

    for i in eachindex(σ)
        σ_safe[i] = σ[i] == 0 ? 1 : σ[i]
    end

    X_norm = (X .- μ) ./ σ_safe

    return Float32.(X_norm), Float32.(μ), Float32.(σ_safe)
end

structural_features_norm, structural_mean, structural_std =
    standardise_features(structural_features)

chemical_features_norm, chemical_mean, chemical_std =
    standardise_features(chemical_features)

combined_features_norm, combined_mean, combined_std =
    standardise_features(combined_features)

println("\nNormalised structural features:")
println(structural_features_norm)

println("\nNormalised chemistry-informed features:")
println(chemical_features_norm)

println("\nNormalised combined features:")
println(combined_features_norm)


#9.store feature sets for later experiments

#later regression experiments can compare:
#1. structural features only
#2. chemistry-informed features only
#3. combined features

feature_sets = Dict(
    :structural => structural_features_norm,
    :chemical => chemical_features_norm,
    :combined => combined_features_norm
)

feature_names = Dict(
    :structural => structural_feature_names,
    :chemical => chemical_feature_names,
    :combined => combined_feature_names
)

println("\nAvailable feature sets:")
for key in sort(collect(keys(feature_sets)))
    println(key, " => size ", size(feature_sets[key]))
end


#10.toy reaction level regression targets

#these are placeholder values only.
#In the final project, they would be replaced by real:
#- energy barriers
#- reaction rates
#- reaction yields

toy_energy_barriers = Float32[
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

toy_reaction_rates = Float32[
    1.2,
    0.8,
    2.1,
    0.5,
    1.4,
    0.3,
    1.9,
    0.2,
    1.1
]

toy_reaction_yields = Float32[
    65.0,
    58.0,
    72.0,
    49.0,
    61.0,
    38.0,
    70.0,
    31.0,
    67.0
]

reaction_targets = Dict(
    :energy_barrier => toy_energy_barriers,
    :reaction_rate => toy_reaction_rates,
    :reaction_yield => toy_reaction_yields
)

println("\nToy reaction-level regression targets:")
for key in sort(collect(keys(reaction_targets)))
    println(key, " => ", reaction_targets[key])
end


#11.package processed output

#this makes it easier to reuse the data in later files

processed_feature_data = (
    raw_reactions = raw_reactions,
    parsed_reactions = parsed_reactions,
    species = species,
    node_to_id = node_to_id,
    id_to_node = id_to_node,
    source_matrix = source_matrix,
    target_matrix = target_matrix,
    structural_feature_names = structural_feature_names,
    chemical_feature_names = chemical_feature_names,
    combined_feature_names = combined_feature_names,
    structural_features = structural_features,
    chemical_features = chemical_features,
    combined_features = combined_features,
    structural_features_norm = structural_features_norm,
    chemical_features_norm = chemical_features_norm,
    combined_features_norm = combined_features_norm,
    feature_sets = feature_sets,
    feature_names = feature_names,
    reaction_targets = reaction_targets
)

println("\nProcessed feature data keys:")
println(keys(processed_feature_data))


#12.final summary



println("Summary")
println("Number of reactions: ", num_reactions)
println("Number of species: ", num_species)
println("Source matrix size: ", size(source_matrix))
println("Target matrix size: ", size(target_matrix))
println("Structural feature matrix size: ", size(structural_features))
println("Chemical feature matrix size: ", size(chemical_features))
println("Combined feature matrix size: ", size(combined_features))
println("Normalised combined feature matrix size: ", size(combined_features_norm))

println("\nThis file prepares species-level structural and chemistry-informed")
println("features for later reaction-level regression experiments.")