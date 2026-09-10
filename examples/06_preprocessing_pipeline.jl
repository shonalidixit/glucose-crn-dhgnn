#preprocessing pipeline
#trying to connect the small pieces from the earlier files:
#parsing reactions, ids, incidence matrix, masks, and feature prep

using Statistics
using SimpleHypergraphs
using SimpleDirectedHypergraphs


#toy style reactions
raw_reactions = [
    "A + B -> C",
    "C + D -> E",
    "E -> F + G",
    "G -> H",
    "H + A -> I"
]

println("\nraw reactions")
for r in raw_reactions
    println(r)
end

#small helper to split one side of a reaction
function parse_side(x)
    return [strip(s) for s in split(x, "+")]
end

function parse_reaction(r, id)
    sides = split(r, "->")

    reactants = parse_side(sides[1])
    products = parse_side(sides[2])

    return (
        id = id,
        raw = r,
        reactants = reactants,
        products = products
    )
end

reactions = [parse_reaction(raw_reactions[i], i) for i in 1:length(raw_reactions)]

println("\nparsed reactions")
for r in reactions
    println("r", r.id, ": ", join(r.reactants, " + "), " -> ", join(r.products, " + "))
end

#collectig all species/nodes
species_set = Set{String}()

for r in reactions
    for s in vcat(r.reactants, r.products)
        push!(species_set, s)
    end
end

nodes = sort(collect(species_set))

node_to_id = Dict(node => i for (i, node) in enumerate(nodes))
id_to_node = Dict(i => node for (i, node) in enumerate(nodes))

println("\nnodes/species")
println(nodes)

println("\nnode ids")
println(node_to_id)

#converting reaction names into ids
id_reactions = []

for r in reactions
    reactant_ids = [node_to_id[x] for x in r.reactants]
    product_ids = [node_to_id[x] for x in r.products]

    push!(
        id_reactions,
        (
            id = r.id,
            reactants = r.reactants,
            products = r.products,
            reactant_ids = reactant_ids,
            product_ids = product_ids
        )
    )
end

println("\nid-based reactions")
for r in id_reactions
    println("r", r.id)
    println("  reactants: ", r.reactants, " -> ", r.reactant_ids)
    println("  products:  ", r.products, " -> ", r.product_ids)
end

#build signed incidence matrix
#-1 = reactant/source side
#+1 = product/target side
#0 = not involved

H = zeros(Int, length(nodes), length(id_reactions))

for (j, r) in enumerate(id_reactions)
    for id in r.reactant_ids
        H[id, j] = -1
    end

    for id in r.product_ids
        H[id, j] = 1
    end
end

println("\nsigned incidence matrix")
println(H)

H_source = Int.(H .== -1)
H_target = Int.(H .== 1)
H_membership = abs.(H)

println("\nsource matrix")
println(H_source)

println("\ntarget matrix")
println(H_target)

println("\nunsigned membership matrix")
println(H_membership)

#masks from the incidence matrix
source_masks = Dict{Int, Vector{Bool}}()
target_masks = Dict{Int, Vector{Bool}}()
active_masks = Dict{Int, Vector{Bool}}()

for j in 1:size(H, 2)
    source_masks[j] = H[:, j] .== -1
    target_masks[j] = H[:, j] .== 1
    active_masks[j] = H[:, j] .!= 0
end

println("\nchecking masks")
for j in 1:length(id_reactions)
    println("\nr", j)
    println("source nodes: ", nodes[source_masks[j]])
    println("target nodes: ", nodes[target_masks[j]])
    println("active nodes: ", nodes[active_masks[j]])
end

#simple structural features for nodes
#this is not chemical descriptor data yet, just useful toy features

source_count = vec(sum(H_source, dims = 2))
target_count = vec(sum(H_target, dims = 2))
participation_count = vec(sum(H_membership, dims = 2))

node_features = hcat(source_count, target_count, participation_count)

println("\nnode features")
println("columns = source_count, target_count, participation_count")
println(node_features)

#hyperedge/reaction features
source_size = vec(sum(H_source, dims = 1))
target_size = vec(sum(H_target, dims = 1))
total_size = vec(sum(H_membership, dims = 1))

hyperedge_features = hcat(source_size, target_size, total_size)

println("\nhyperedge features")
println("columns = source_size, target_size, total_size")
println(hyperedge_features)

#edge list version
#useful because some ML/message passing code works with index lists

edge_list = []

for j in 1:size(H, 2)
    for i in 1:size(H, 1)
        if H[i, j] != 0
            push!(
                edge_list,
                (
                    node_id = i,
                    hyperedge_id = j,
                    role = H[i, j]
                )
            )
        end
    end
end

println("\nedge list")
for e in edge_list
    println(e)
end

#gather prep: for each reaction, storing source and target ids separately
source_id_lists = Dict{Int, Vector{Int}}()
target_id_lists = Dict{Int, Vector{Int}}()

for r in id_reactions
    source_id_lists[r.id] = r.reactant_ids
    target_id_lists[r.id] = r.product_ids
end

println("\nsource/target id lists")
for r in id_reactions
    println("r", r.id, " source ids = ", source_id_lists[r.id], 
            " target ids = ", target_id_lists[r.id])
end

#small batching prep
#since reactions have variable numbers of reactants/products,
#pad the ids and keep masks so padded zeros are ignored

max_source_len = maximum(length(r.reactant_ids) for r in id_reactions)
max_target_len = maximum(length(r.product_ids) for r in id_reactions)

source_id_batch = zeros(Int, length(id_reactions), max_source_len)
target_id_batch = zeros(Int, length(id_reactions), max_target_len)

source_batch_mask = falses(length(id_reactions), max_source_len)
target_batch_mask = falses(length(id_reactions), max_target_len)

for (i, r) in enumerate(id_reactions)
    for (j, id) in enumerate(r.reactant_ids)
        source_id_batch[i, j] = id
        source_batch_mask[i, j] = true
    end

    for (j, id) in enumerate(r.product_ids)
        target_id_batch[i, j] = id
        target_batch_mask[i, j] = true
    end
end

println("\npadded source id batch")
println(source_id_batch)

println("\nsource batch mask")
println(source_batch_mask)

println("\npadded target id batch")
println(target_id_batch)

println("\ntarget batch mask")
println(target_batch_mask)

#packaging the output like a preprocessing function might return
processed = Dict(
    "nodes" => nodes,
    "node_to_id" => node_to_id,
    "id_to_node" => id_to_node,
    "reactions" => id_reactions,
    "signed_incidence" => H,
    "source_matrix" => H_source,
    "target_matrix" => H_target,
    "membership_matrix" => H_membership,
    "node_features" => node_features,
    "hyperedge_features" => hyperedge_features,
    "edge_list" => edge_list,
    "source_masks" => source_masks,
    "target_masks" => target_masks,
    "active_masks" => active_masks,
    "source_id_batch" => source_id_batch,
    "target_id_batch" => target_id_batch,
    "source_batch_mask" => source_batch_mask,
    "target_batch_mask" => target_batch_mask
)

println("\nprocessed object keys")
println(collect(keys(processed)))

println("\nsummary")
println("number of nodes: ", length(nodes))
println("number of reactions/hyperedges: ", length(id_reactions))
println("incidence matrix size: ", size(H))
println("node feature matrix size: ", size(node_features))
println("hyperedge feature matrix size: ", size(hyperedge_features))

