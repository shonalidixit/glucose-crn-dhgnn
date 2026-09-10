using CSV
using DataFrames
using RDKitMinimalLib


"""
    split_species(reaction_smiles)

Split one side of a reaction SMILES into its individual molecular species.

Disconnected molecular components in SMILES are separated by periods.
"""
split_species(reaction_smiles::AbstractString) = split(reaction_smiles, ".")


"""
    remove_atom_maps(mapped_smiles)

Remove atom-map indices from an atom-mapped SMILES string.

Atom-map numbers describe atom correspondence across a reaction rather than
molecular identity. The original mapped SMILES remain unchanged in the
dataset; mapping labels are removed only when constructing molecular
identities and features.
"""
function remove_atom_maps(mapped_smiles::AbstractString)
    return replace(mapped_smiles, r":\d+" => "")
end


"""
    canonicalise_species(mapped_smiles)

Convert an atom-mapped molecular SMILES string into an unmapped canonical
SMILES representation.

Canonical SMILES are used as molecular vertex identifiers so that the same
molecule is not represented as multiple vertices solely because of SMILES
ordering or atom-map labels.
"""
function canonicalise_species(mapped_smiles::AbstractString)
    unmapped_smiles = remove_atom_maps(mapped_smiles)
    molecule = RDKitMinimalLib.get_mol(unmapped_smiles)

    return RDKitMinimalLib.get_smiles(molecule)
end


"""
    canonicalise_reaction_side(reaction_smiles)

Split one side of a reaction into molecular components and canonicalise each
component independently.
"""
function canonicalise_reaction_side(reaction_smiles::AbstractString)
    species = split_species(reaction_smiles)

    return canonicalise_species.(species)
end


"""
    collect_molecular_species(reactions)

Extract and canonicalise all reactant and product species in the reaction
dataset.

Returns the canonical reactants and products for every reaction together
with the unique molecular species that will form the hypergraph vertices.
"""
function collect_molecular_species(reactions::DataFrame)
    reactants = canonicalise_reaction_side.(reactions.Rsmiles)
    products = canonicalise_reaction_side.(reactions.Psmiles)

    all_species = vcat(
        reduce(vcat, reactants),
        reduce(vcat, products),
    )

    unique_species = sort(unique(all_species))

    return reactants, products, unique_species
end


"""
    morgan_fingerprint(canonical_smiles; radius=2, nbits=1024)

Generate a fixed-length Morgan fingerprint for a molecular species.

A radius of 2 captures local atomic environments up to two bonds away and is
used here as a simple structural baseline for molecular vertex features.
"""
function morgan_fingerprint(
    canonical_smiles::AbstractString;
    radius::Int=2,
    nbits::Int=1024,
)
    molecule = RDKitMinimalLib.get_mol(canonical_smiles)

    options = Dict{String,Any}(
        "radius" => radius,
        "nBits" => nbits,
    )

    fingerprint = RDKitMinimalLib.get_morgan_fp(
        molecule,
        options,
    )

    return Float32[
        bit == '1' ? 1.0f0 : 0.0f0
        for bit in fingerprint
    ]
end


"""
    build_vertex_features(species; radius=2, nbits=1024)

Generate a Morgan fingerprint feature matrix for all molecular vertices.

Rows correspond to molecular species and columns correspond to fingerprint
features.
"""
function build_vertex_features(
    species;
    radius::Int=2,
    nbits::Int=1024,
)
    fingerprints = [
        morgan_fingerprint(
            molecule;
            radius=radius,
            nbits=nbits,
        )
        for molecule in species
    ]

    return reduce(vcat, permutedims.(fingerprints))
end


"""
    build_vertex_index(species)

Assign a unique integer vertex ID to each canonical molecular species.
"""
function build_vertex_index(species)
    return Dict(
        molecule => vertex_id
        for (vertex_id, molecule) in enumerate(species)
    )
end


data_path = joinpath(
    @__DIR__,
    "..",
    "data",
    "glucose",
    "glucose_network.csv",
)

reactions = CSV.read(data_path, DataFrame)

reactants, products, species =
    collect_molecular_species(reactions)

vertex_ids = build_vertex_index(species)

vertex_features = build_vertex_features(species)

println("Glucose CRN molecular feature preparation")
println("-----------------------------------------")
println("Reactions: ", nrow(reactions))
println("Unique molecular vertices: ", length(species))
println("Vertex feature matrix: ", size(vertex_features))
println("Morgan fingerprint radius: 2")
println("Morgan fingerprint length: 1024")
println("Maximum reactants per reaction: ", maximum(length.(reactants)))
println("Maximum products per reaction: ", maximum(length.(products)))