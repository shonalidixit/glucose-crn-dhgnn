using SparseArrays
using DataFrames

include("molecular_features.jl")


"""
    reaction_vertex_ids(reaction_species, vertex_ids)

Convert the canonical molecular species participating in each reaction
into their corresponding integer vertex IDs.
"""
function reaction_vertex_ids(
    reaction_species,
    vertex_ids::Dict,
)
    return [
        [vertex_ids[molecule] for molecule in reaction]
        for reaction in reaction_species
    ]
end


"""
    build_directed_incidence(
        reaction_sources,
        reaction_targets,
        n_vertices,
    )

Construct source and target incidence matrices for a directed hypergraph.

Rows correspond to molecular vertices and columns correspond to reactions.
The source incidence matrix records reactant participation, while the target
incidence matrix records product participation.
"""
function build_directed_incidence(
    reaction_sources,
    reaction_targets,
    n_vertices::Int,
)
    n_reactions = length(reaction_sources)

    source_rows = Int[]
    source_columns = Int[]

    target_rows = Int[]
    target_columns = Int[]

    for reaction_id in 1:n_reactions
        for vertex_id in reaction_sources[reaction_id]
            push!(source_rows, vertex_id)
            push!(source_columns, reaction_id)
        end

        for vertex_id in reaction_targets[reaction_id]
            push!(target_rows, vertex_id)
            push!(target_columns, reaction_id)
        end
    end

    source_incidence = sparse(
        source_rows,
        source_columns,
        ones(Float32, length(source_rows)),
        n_vertices,
        n_reactions,
    )

    target_incidence = sparse(
        target_rows,
        target_columns,
        ones(Float32, length(target_rows)),
        n_vertices,
        n_reactions,
    )

    return source_incidence, target_incidence
end


"""
    convert_numeric_column(values)

Convert reaction-property values to Float32.

Values that cannot be interpreted as numbers are represented as `missing`.
This is particularly important for DH because the original CSV column is
read as a string column.
"""
function convert_numeric_column(values)
    return [
        if value isa Number
            Float32(value)
        else
            parsed_value = tryparse(Float32, strip(string(value)))
            isnothing(parsed_value) ? missing : parsed_value
        end
        for value in values
    ]
end


"""
    prepare_reaction_properties(reactions)

Extract the energetic properties associated with each reaction.

DE, DG and DH are retained separately so that different reaction-level
regression targets can be investigated without changing the hypergraph
representation.
"""
function prepare_reaction_properties(reactions::DataFrame)
    return (
        DE=convert_numeric_column(reactions.DE),
        DG=convert_numeric_column(reactions.DG),
        DH=convert_numeric_column(reactions.DH),
    )
end


"""
    build_directed_crn(
        reactants,
        products,
        vertex_ids,
        reactions,
    )

Construct the directed hypergraph representation of the glucose reaction
network.

Molecular species are vertices and reactions are directed hyperedges.
Reactant molecules form the source side of each hyperedge and product
molecules form the target side.
"""
function build_directed_crn(
    reactants,
    products,
    vertex_ids,
    reactions::DataFrame,
)
    reaction_sources = reaction_vertex_ids(
        reactants,
        vertex_ids,
    )

    reaction_targets = reaction_vertex_ids(
        products,
        vertex_ids,
    )

    source_incidence, target_incidence =
        build_directed_incidence(
            reaction_sources,
            reaction_targets,
            length(vertex_ids),
        )

    reaction_properties =
        prepare_reaction_properties(reactions)

    return (
        reaction_sources=reaction_sources,
        reaction_targets=reaction_targets,
        source_incidence=source_incidence,
        target_incidence=target_incidence,
        reaction_properties=reaction_properties,
    )
end


directed_crn = build_directed_crn(
    reactants,
    products,
    vertex_ids,
    reactions,
)


println()
println("Directed glucose CRN")
println("--------------------")
println("Vertices: ", length(vertex_ids))
println("Directed hyperedges: ", nrow(reactions))
println("Source incidence matrix: ", size(directed_crn.source_incidence))
println("Target incidence matrix: ", size(directed_crn.target_incidence))

println(
    "Source incidences: ",
    nnz(directed_crn.source_incidence),
)

println(
    "Target incidences: ",
    nnz(directed_crn.target_incidence),
)

println(
    "Reactions with multiple reactants: ",
    count(length(source) > 1 for source in directed_crn.reaction_sources),
)

println(
    "Reactions with multiple products: ",
    count(length(target) > 1 for target in directed_crn.reaction_targets),
)

println(
    "Maximum source hyperedge size: ",
    maximum(length.(directed_crn.reaction_sources)),
)

println(
    "Maximum target hyperedge size: ",
    maximum(length.(directed_crn.reaction_targets)),
)

println(
    "Valid DE values: ",
    count(!ismissing, directed_crn.reaction_properties.DE),
)

println(
    "Valid DG values: ",
    count(!ismissing, directed_crn.reaction_properties.DG),
)

println(
    "Valid DH values: ",
    count(!ismissing, directed_crn.reaction_properties.DH),
)