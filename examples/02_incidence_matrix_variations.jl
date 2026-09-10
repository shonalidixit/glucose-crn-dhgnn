#incidence matrix variations
#playing around with a few ways to store the same reaction network

#toy reaction network

nodes = ["A", "B", "C", "D", "E", "F", "G"]

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
id_to_node = Dict(i => node for (i, node) in enumerate(nodes))

reactions = [
    (
        id = 1,
        reactants = ["A", "B"],
        products = ["C"]
    ),
    (
        id = 2,
        reactants = ["C"],
        products = ["D", "E"]
    ),
    (
        id = 3,
        reactants = ["E", "F"],
        products = ["G"]
    )
]

println("\nToy reactions:")
for r in reactions
    println("e", r.id, ": ", join(r.reactants, " + "), " -> ", join(r.products, " + "))
end


#signed incidence matrix
#-1 for reactants, +1 for products

function signed_incidence_matrix(nodes, reactions, node_to_id)
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

H_signed = signed_incidence_matrix(nodes, reactions, node_to_id)

println("\nSigned incidence matrix:")
println("Rows = nodes: ", nodes)
println("Columns = reactions: e1, e2, e3")
println(H_signed)


#separate source and target matrices

function source_incidence_matrix(nodes, reactions, node_to_id)
    H_source = zeros(Int, length(nodes), length(reactions))

    for (j, r) in enumerate(reactions)
        for reactant in r.reactants
            H_source[node_to_id[reactant], j] = 1
        end
    end

    return H_source
end

function target_incidence_matrix(nodes, reactions, node_to_id)
    H_target = zeros(Int, length(nodes), length(reactions))

    for (j, r) in enumerate(reactions)
        for product in r.products
            H_target[node_to_id[product], j] = 1
        end
    end

    return H_target
end

H_source = source_incidence_matrix(nodes, reactions, node_to_id)
H_target = target_incidence_matrix(nodes, reactions, node_to_id)

println("\nSource/reactant incidence matrix:")
println(H_source)

println("\nTarget/product incidence matrix:")
println(H_target)


#unsigned version, ignoring direction for now

function unsigned_membership_matrix(H_signed)
    return abs.(H_signed)
end

H_unsigned = unsigned_membership_matrix(H_signed)

println("\nUnsigned hypergraph membership matrix:")
println(H_unsigned)


#checking if I can recover reactants/products from one matrix column

function extract_reactants_from_column(H_signed, edge_index, id_to_node)
    reactants = String[]

    for i in 1:size(H_signed, 1)
        if H_signed[i, edge_index] == -1
            push!(reactants, id_to_node[i])
        end
    end

    return reactants
end

function extract_products_from_column(H_signed, edge_index, id_to_node)
    products = String[]

    for i in 1:size(H_signed, 1)
        if H_signed[i, edge_index] == 1
            push!(products, id_to_node[i])
        end
    end

    return products
end

println("\nReconstruct reactions from signed incidence matrix:")
for j in 1:size(H_signed, 2)
    reactants = extract_reactants_from_column(H_signed, j, id_to_node)
    products = extract_products_from_column(H_signed, j, id_to_node)

    println("e", j, ": ", join(reactants, " + "), " -> ", join(products, " + "))
end


#edge list version of the same incidence information

function incidence_to_edge_list(H_signed)
    edge_list = []

    for j in 1:size(H_signed, 2)
        for i in 1:size(H_signed, 1)
            value = H_signed[i, j]

            if value != 0
                push!(
                    edge_list,
                    (
                        node_id = i,
                        hyperedge_id = j,
                        role = value
                    )
                )
            end
        end
    end

    return edge_list
end

edge_list = incidence_to_edge_list(H_signed)

println("\nEdge-list style representation:")
println("(role = -1 means source/reactant, role = +1 means target/product)")
for item in edge_list
    println(item)
end


#quick stats from the matrices

source_sizes = vec(sum(H_source, dims = 1))
target_sizes = vec(sum(H_target, dims = 1))
total_sizes = vec(sum(H_unsigned, dims = 1))

println("\nHyperedge sizes computed from matrices:")
for j in 1:length(reactions)
    println("e", j,
        " | source size = ", source_sizes[j],
        " | target size = ", target_sizes[j],
        " | total size = ", total_sizes[j])
end

node_membership_counts = vec(sum(H_unsigned, dims = 2))

println("\nNode membership counts computed from unsigned matrix:")
for i in 1:length(nodes)
    println(nodes[i], " appears in ", node_membership_counts[i], " hyperedge(s)")
end