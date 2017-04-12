include("Save.jl")
using Save
using HDF5

include("utils.jl")
include("reweight2.jl")
include("downsample.jl")
include("compute_regiongraph.jl")
include("compute_fullgraph.jl")
include("filter_samples.jl")

#TODO: Change this to read and write to a cloud filesystem
function do_prep(basename, patch_size = (318,318,33))
	mean_labels = Save.load(joinpath(basename,"mean_agg_tr.h5"))
	full_size = size(mean_labels)

	raw = Save.load(joinpath(basename,"raw.h5"))
	central_ranges = [Int(ceil(p/2+1)) : Int(floor(s-p/2-1)) for (p,s) in zip(patch_size, full_size)]
	valid_mask = zeros(Int8, full_size)
	valid_mask[central_ranges...] = 1

	samples = gen_samples(mean_labels, patch_size, N=400000, mask=valid_mask, M=30)
	Save.save(joinpath(basename,"samples.h5"), samples)

	affinities = Save.load(joinpath(basename,"aff.h5"))
	@time vertices,edges = compute_regiongraph(raw, mean_labels, affinities, threshold=0.3)
	Save.save(joinpath(basename,"vertices.h5"), vertices)
	Save.save(joinpath(basename,"edges.h5"), edges)
	full_edges = compute_fullgraph(Batched(), raw, resolution=Int[4,4,40], radius=130, downsample=Int[4,4,1])
	Save.save(joinpath(basename,"full_edges.h5"), full_edges)
	contact_edges = compute_contactgraph(raw)
	Save.save(joinpath(basename,"contact_edges.h5"), contact_edges)
end

basename = expanduser(ARGS[1])
do_prep(basename)