#masking operations
#trying out masks on a small reaction network

using Statistics
using SimpleHypergraphs
using SimpleDirectedHypergraphs


#toy reactions

nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
id_to_node = Dict(i => node for (i, node) in enumerate(nodes))

reactions = [
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
    ),
    (
        id = 4,
        reactants = ["F"],
        products = ["G", "H"]
    )
]

println("\nToy directed hypergraph reactions:")
for r in reactions
    println("e", r.id, ": ",
        join(r.reactants, " + "),
        " -> ",
        join(r.products, " + "))
end


#small helper functions

function empty_bool_mask(n)
    return falses(n)
end

function mask_from_nodes(selected_nodes, node_to_id, total_nodes)
    mask = falses(total_nodes)

    for node in selected_nodes
        mask[node_to_id[node]] = true
    end

    return mask
end

function print_mask_with_nodes(mask, nodes, label)
    println("\n", label)
    println("Mask: ", mask)
    println("Selected nodes: ", nodes[mask])
end


#reactant/source masks

println("\nReactant/source masks per hyperedge:")

reactant_masks = Dict{Int, Vector{Bool}}()

for r in reactions
    mask = mask_from_nodes(r.reactants, node_to_id, length(nodes))
    reactant_masks[r.id] = mask

    print_mask_with_nodes(mask, nodes, "e$(r.id) reactant/source mask")
end


#product/target masks

println("\nProduct/target masks per hyperedge:")

product_masks = Dict{Int, Vector{Bool}}()

for r in reactions
    mask = mask_from_nodes(r.products, node_to_id, length(nodes))
    product_masks[r.id] = mask

    print_mask_with_nodes(mask, nodes, "e$(r.id) product/target mask")
end


#active nodes means reactants and products together

println("\nActive node masks per hyperedge:")

active_masks = Dict{Int, Vector{Bool}}()

for r in reactions
    involved_nodes = vcat(r.reactants, r.products)
    mask = mask_from_nodes(involved_nodes, node_to_id, length(nodes))
    active_masks[r.id] = mask

    print_mask_with_nodes(mask, nodes, "e$(r.id) active node mask")
end


#opposite of active nodes

println("\nInactive node masks per hyperedge:")

inactive_masks = Dict{Int, Vector{Bool}}()

for r in reactions
    inactive_mask = .!active_masks[r.id]
    inactive_masks[r.id] = inactive_mask

    print_mask_with_nodes(inactive_mask, nodes, "e$(r.id) inactive node mask")
end


#same masks but as integers

println("\nInteger masks for reactants/products:")

for r in reactions
    reactant_int_mask = Int.(reactant_masks[r.id])
    product_int_mask = Int.(product_masks[r.id])
    active_int_mask = Int.(active_masks[r.id])

    println("\ne", r.id)
    println("Reactant integer mask: ", reactant_int_mask)
    println("Product integer mask:  ", product_int_mask)
    println("Active integer mask:   ", active_int_mask)
end


#global masks across the whole reaction network

global_reactant_mask = falses(length(nodes))
global_product_mask = falses(length(nodes))

for r in reactions
    global_reactant_mask .|= reactant_masks[r.id]
    global_product_mask .|= product_masks[r.id]
end

print_mask_with_nodes(global_reactant_mask, nodes, "Global reactant/source mask")
print_mask_with_nodes(global_product_mask, nodes, "Global product/target mask")

global_active_mask = global_reactant_mask .| global_product_mask
print_mask_with_nodes(global_active_mask, nodes, "Global active node mask")


#building masks from an incidence matrix too

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

println("\nMasks derived from incidence matrix:")

for j in 1:size(H, 2)
    source_mask = H[:, j] .== -1
    target_mask = H[:, j] .== 1
    active_mask = H[:, j] .!= 0

    println("\nHyperedge e", j)
    println("Source mask from incidence: ", source_mask)
    println("Target mask from incidence: ", target_mask)
    println("Active mask from incidence: ", active_mask)
    println("Source nodes: ", nodes[source_mask])
    println("Target nodes: ", nodes[target_mask])
    println("Active nodes: ", nodes[active_mask])
end


#trying masks on toy node features

node_features = [
    1.0  0.2  0.5;
    0.8  0.1  0.4;
    0.3  0.9  0.7;
    0.4  0.6  0.8;
    0.9  0.2  0.1;
    0.5  0.5  0.5;
    0.7  0.3  0.6;
    0.2  0.8  0.9
]

println("\nNode feature matrix:")
println(node_features)

chosen_edge = reactions[1]

chosen_reactant_mask = reactant_masks[chosen_edge.id]
chosen_product_mask = product_masks[chosen_edge.id]
chosen_active_mask = active_masks[chosen_edge.id]

println("\nFeature masking for hyperedge e", chosen_edge.id)

reactant_features = node_features[chosen_reactant_mask, :]
product_features = node_features[chosen_product_mask, :]
active_features = node_features[chosen_active_mask, :]

println("Reactant features:")
println(reactant_features)

println("Product features:")
println(product_features)

println("Active node features:")
println(active_features)


#filtering reactions with a simple hyperedge mask

hyperedge_mask = falses(length(reactions))

for (i, r) in enumerate(reactions)
    if length(r.reactants) >= 2
        hyperedge_mask[i] = true
    end
end

println("\nHyperedge mask where source size >= 2:")
println(hyperedge_mask)

selected_reactions = reactions[hyperedge_mask]

println("Selected reactions:")
for r in selected_reactions
    println("e", r.id, ": ",
        join(r.reactants, " + "),
        " -> ",
        join(r.products, " + "))
end


#padding ids and keeping masks for the real entries

max_source_size = maximum(length(r.reactants) for r in reactions)
max_target_size = maximum(length(r.products) for r in reactions)

source_id_batch = zeros(Int, length(reactions), max_source_size)
source_padding_mask = falses(length(reactions), max_source_size)

target_id_batch = zeros(Int, length(reactions), max_target_size)
target_padding_mask = falses(length(reactions), max_target_size)

for (i, r) in enumerate(reactions)
    reactant_ids = [node_to_id[node] for node in r.reactants]
    product_ids = [node_to_id[node] for node in r.products]

    for (j, id) in enumerate(reactant_ids)
        source_id_batch[i, j] = id
        source_padding_mask[i, j] = true
    end

    for (j, id) in enumerate(product_ids)
        target_id_batch[i, j] = id
        target_padding_mask[i, j] = true
    end
end

println("\nPadded source/reactant ID batch:")
println(source_id_batch)

println("\nSource padding mask:")
println(source_padding_mask)

println("\nPadded target/product ID batch:")
println(target_id_batch)

println("\nTarget padding mask:")
println(target_padding_mask)


#masked aggregation example

println("\nMasked feature aggregation per hyperedge:")

for r in reactions
    src_mask = reactant_masks[r.id]
    tgt_mask = product_masks[r.id]

    src_features = node_features[src_mask, :]
    tgt_features = node_features[tgt_mask, :]

    src_mean = vec(mean(src_features, dims = 1))
    tgt_mean = vec(mean(tgt_features, dims = 1))

    println("\ne", r.id, ": ",
        join(r.reactants, " + "),
        " -> ",
        join(r.products, " + "))

    println("Source mean feature: ", src_mean)
    println("Target mean feature: ", tgt_mean)
end


#quick summary

println("\nSummary:")
println("Number of nodes: ", length(nodes))
println("Number of reactions/hyperedges: ", length(reactions))
println("Signed incidence size: ", size(H))
println("Maximum source size: ", max_source_size)
println("Maximum target size: ", max_target_size)