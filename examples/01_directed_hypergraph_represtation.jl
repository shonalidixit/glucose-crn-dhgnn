#directed hypergraph representation
#starting with a small toy reaction network

#packages I am keeping in mind for the project:
#SimpleHypergraphs.jl
#SimpleDirectedHypergraphs.jl
#HyperGraphNeuralNetworks.jl


#toy species/nodes

nodes = ["A", "B", "C", "D", "E", "F"]

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
id_to_node = Dict(i => node for (i, node) in enumerate(nodes))

println("\nNodes:")
println(nodes)

println("\nNode to ID mapping:")
println(node_to_id)


#toy reactions as directed hyperedges

hyperedges = [
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
    )
]

println("\nDirected hyperedges / toy reactions:")
for edge in hyperedges
    println("e", edge.id, ": ",
        join(edge.reactants, " + "),
        " -> ",
        join(edge.products, " + "))
end


#just pulling out source and target sides

function source_set(edge)
    return edge.reactants
end

function target_set(edge)
    return edge.products
end

println("\nSource and target sets:")
for edge in hyperedges
    println("Hyperedge e", edge.id)
    println("  Source/reactants: ", source_set(edge))
    println("  Target/products: ", target_set(edge))
end


#signed incidence style matrix
#-1 for reactants, +1 for products, 0 otherwise

function build_directed_incidence(nodes, hyperedges, node_to_id)
    incidence = zeros(Int, length(nodes), length(hyperedges))

    for (j, edge) in enumerate(hyperedges)
        for r in edge.reactants
            incidence[node_to_id[r], j] = -1
        end

        for p in edge.products
            incidence[node_to_id[p], j] = 1
        end
    end

    return incidence
end

incidence = build_directed_incidence(nodes, hyperedges, node_to_id)

println("\nDirected incidence-style matrix:")
println("Rows = nodes: ", nodes)
println("Columns = hyperedges: e1, e2, e3")
println(incidence)


#separate reactant/product membership matrices

function build_source_target_incidence(nodes, hyperedges, node_to_id)
    source_incidence = zeros(Int, length(nodes), length(hyperedges))
    target_incidence = zeros(Int, length(nodes), length(hyperedges))

    for (j, edge) in enumerate(hyperedges)
        for r in edge.reactants
            source_incidence[node_to_id[r], j] = 1
        end

        for p in edge.products
            target_incidence[node_to_id[p], j] = 1
        end
    end

    return source_incidence, target_incidence
end

source_incidence, target_incidence =
    build_source_target_incidence(nodes, hyperedges, node_to_id)

println("\nSource/reactant incidence matrix:")
println(source_incidence)

println("\nTarget/product incidence matrix:")
println(target_incidence)


#basic size checks for each reaction

println("\nHyperedge statistics:")

for edge in hyperedges
    source_size = length(edge.reactants)
    target_size = length(edge.products)
    total_size = source_size + target_size

    println("Hyperedge e", edge.id)
    println("  Source size: ", source_size)
    println("  Target size: ", target_size)
    println("  Total size: ", total_size)
end


#how often each species appears

node_participation = Dict(node => 0 for node in nodes)

for edge in hyperedges
    involved_nodes = vcat(edge.reactants, edge.products)

    for node in involved_nodes
        node_participation[node] += 1
    end
end

println("\nNode participation counts:")
for node in nodes
    println(node, " participates in ", node_participation[node], " hyperedge(s)")
end


#source vs target counts for each node

out_degree = Dict(node => 0 for node in nodes)
in_degree = Dict(node => 0 for node in nodes)

for edge in hyperedges
    for r in edge.reactants
        out_degree[r] += 1
    end

    for p in edge.products
        in_degree[p] += 1
    end
end

println("\nDirected node degree-style counts:")
for node in nodes
    println(
        node,
        " | out/source count = ", out_degree[node],
        " | in/target count = ", in_degree[node]
    )
end


#same reactions but using ids instead of names

function convert_edges_to_ids(hyperedges, node_to_id)
    id_edges = []

    for edge in hyperedges
        push!(
            id_edges,
            (
                id = edge.id,
                reactant_ids = [node_to_id[r] for r in edge.reactants],
                product_ids = [node_to_id[p] for p in edge.products]
            )
        )
    end

    return id_edges
end

id_hyperedges = convert_edges_to_ids(hyperedges, node_to_id)

println("\nID-based directed hyperedge representation:")
for edge in id_hyperedges
    println("e", edge.id)
    println("  Reactant IDs: ", edge.reactant_ids)
    println("  Product IDs: ", edge.product_ids)
end


#quick summary

println("\nSummary:")
println("Number of nodes: ", length(nodes))
println("Number of directed hyperedges: ", length(hyperedges))
println("Incidence matrix size: ", size(incidence))