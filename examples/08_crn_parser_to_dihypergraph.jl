#08_crn_parser_to_dihypergraph.jl

#this file is for converting CRN reaction strings into a directed hypergraph style data structure.

#the main idea is:
#species   = nodes
#reactions = directed hyperedges
#reactants = source/tail side
#products  = target/head side

#i am starting with a small toy CRN first, because it makes the parsing logic easier to check before applying the same workflow to CRNs from literature.


#1.small CRN example

raw_reactions = [
    "A + B -> C",
    "C -> D + E",
    "E + F -> G",
    "G -> H",
    "H + A -> I"
]

println("\nRaw CRN reactions:")

for reaction in raw_reactions
    println(reaction)
end

#2.cleaning helper

#this just removes extra spaces from species names
#for example, " A " becomes "A"

function clean_species_name(name)
    return strip(name)
end


#3.parsing one reaction


#this function takes a reaction string like:
#A + B -> C + D
#and converts it into:
#reactants = ["A", "B"]
#products  = ["C", "D"]

function parse_reaction(reaction_string)
    if !occursin("->", reaction_string)
        error("Reaction is missing -> : $reaction_string")
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


#4.parsing the full CRN

parsed_reactions = [parse_reaction(r) for r in raw_reactions]

println("\nParsed reactions:")

for (i, reaction) in enumerate(parsed_reactions)
    println(
        "r", i, ": ",
        join(reaction.reactants, " + "),
        " -> ",
        join(reaction.products, " + ")
    )
end


#5.extracting species / nodes

#here i collect every species that appears either as a reactant or product
#these become the nodes of the directed hypergraph

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

println("\nExtracted species / nodes:")
println(nodes)


#6.creating node ID mappings


#it is easier to use integer IDs for matrix construction and ML code
#so each species gets an integer ID

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
id_to_node = Dict(i => node for (i, node) in enumerate(nodes))

println("\nNode to ID mapping:")
println(node_to_id)

println("\nID to node mapping:")
println(id_to_node)


#7.converting reactions into directed hyperedges

#each reaction becomes one directed hyperedge
#the source side contains reactant IDs
#the target side contains product IDs

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

println("\nDirected hyperedge representation:")

for edge in directed_hyperedges
    println("\nr", edge.id)
    println("  raw reaction: ", edge.raw)
    println("  source/reactants: ", edge.reactants, " -> ", edge.source_ids)
    println("  target/products:  ", edge.products, " -> ", edge.target_ids)
end


#8.signed incidence matrix


#this matrix stores direction information

#-1 means the species is on the source/reactant side
#+1 means the species is on the target/product side
#0 means the species is not involved in that reaction

function build_signed_incidence(nodes, directed_hyperedges)
    H = zeros(Int, length(nodes), length(directed_hyperedges))

    for edge in directed_hyperedges
        for node_id in edge.source_ids
            H[node_id, edge.id] = -1
        end

        for node_id in edge.target_ids
            H[node_id, edge.id] = 1
        end
    end

    return H
end

signed_incidence = build_signed_incidence(nodes, directed_hyperedges)

println("\nSigned incidence matrix:")
println("Rows = species / nodes")
println("Columns = reactions / directed hyperedges")
println(signed_incidence)


#9.source and target incidence matrices


#these are split versions of the signed incidence matrix
#they make it easier to work separately with reactants and products

function build_source_matrix(nodes, directed_hyperedges)
    S = zeros(Int, length(nodes), length(directed_hyperedges))

    for edge in directed_hyperedges
        for node_id in edge.source_ids
            S[node_id, edge.id] = 1
        end
    end

    return S
end

function build_target_matrix(nodes, directed_hyperedges)
    T = zeros(Int, length(nodes), length(directed_hyperedges))

    for edge in directed_hyperedges
        for node_id in edge.target_ids
            T[node_id, edge.id] = 1
        end
    end

    return T
end

source_matrix = build_source_matrix(nodes, directed_hyperedges)
target_matrix = build_target_matrix(nodes, directed_hyperedges)

println("\nSource / reactant matrix:")
println(source_matrix)

println("\nTarget / product matrix:")
println(target_matrix)



#10.unsigned membership matrix


#this ignores direction and only checks whether a species participates in a reaction.

membership_matrix = abs.(signed_incidence)

println("\nUnsigned membership matrix:")
println(membership_matrix)


#11.reconstructing reactions from the matrix


#this is a sanity check
#if the incidence matrix is correct, the original reactions should be recoverable from it

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

println("\nReconstructed reactions from signed incidence matrix:")

for j in 1:size(signed_incidence, 2)
    reactants, products = reconstruct_reaction(signed_incidence, j, id_to_node)

    println(
        "r", j, ": ",
        join(reactants, " + "),
        " -> ",
        join(products, " + ")
    )
end


#12.edge-list representation


#this is another way of storing the same information
#it may be useful later for batching and indexing

#role = -1 means source/reactant
#role = +1 means target/product

edge_list = []

for edge in directed_hyperedges
    for node_id in edge.source_ids
        push!(
            edge_list,
            (
                node_id = node_id,
                hyperedge_id = edge.id,
                role = -1
            )
        )
    end

    for node_id in edge.target_ids
        push!(
            edge_list,
            (
                node_id = node_id,
                hyperedge_id = edge.id,
                role = 1
            )
        )
    end
end

println("\nEdge-list representation:")
println("(role = -1 source/reactant, role = +1 target/product)")

for item in edge_list
    println(item)
end


#13.simple node features

#these are basic structural features
#they are not chemistry-specific yet, but they are useful as a first ML input

source_count = vec(sum(source_matrix, dims = 2))
target_count = vec(sum(target_matrix, dims = 2))
participation_count = vec(sum(membership_matrix, dims = 2))

node_features = hcat(source_count, target_count, participation_count)

println("\nNode feature names:")
println(["source_count", "target_count", "participation_count"])

println("\nNode feature matrix:")
println(node_features)


#14.simple hyperedge features


#these describe the size of each reaction.
#again, this is a simple starting point before using richer information.

source_size = vec(sum(source_matrix, dims = 1))
target_size = vec(sum(target_matrix, dims = 1))
total_size = vec(sum(membership_matrix, dims = 1))

hyperedge_features = hcat(source_size, target_size, total_size)

println("\nHyperedge feature names:")
println(["source_size", "target_size", "total_size"])

println("\nHyperedge feature matrix:")
println(hyperedge_features)


#15.storing the processed CRN data together


#this keeps all the processed pieces in one object
#later this can be adapted to use package-specific DirectedHypergraph or HGNNHypergraph objects

crn_dihypergraph_data = (
    raw_reactions = raw_reactions,
    parsed_reactions = parsed_reactions,
    nodes = nodes,
    node_to_id = node_to_id,
    id_to_node = id_to_node,
    directed_hyperedges = directed_hyperedges,
    signed_incidence = signed_incidence,
    source_matrix = source_matrix,
    target_matrix = target_matrix,
    membership_matrix = membership_matrix,
    edge_list = edge_list,
    node_features = node_features,
    hyperedge_features = hyperedge_features
)

println("\nProcessed CRN directed hypergraph data keys:")
println(keys(crn_dihypergraph_data))


#16.printing the summary


println("\nSummary:")
println("Number of raw reactions: ", length(raw_reactions))
println("Number of species / nodes: ", length(nodes))
println("Number of directed hyperedges / reactions: ", length(directed_hyperedges))
println("Signed incidence matrix size: ", size(signed_incidence))
println("Source matrix size: ", size(source_matrix))
println("Target matrix size: ", size(target_matrix))
println("Membership matrix size: ", size(membership_matrix))
println("Node feature matrix size: ", size(node_features))
println("Hyperedge feature matrix size: ", size(hyperedge_features))
println("Edge-list length: ", length(edge_list))