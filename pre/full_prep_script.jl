using Save
using HDF5

include("utils.jl")
include("reweight2.jl")
include("downsample.jl")
include("compute_regiongraph.jl")
include("filter_samples.jl")

function do_prep(basename, patch_size = (318,318,33))
	mean_labels = h5read(joinpath(basename,"mean_agg_tr.h5"),"/main")
	full_size = size(mean_labels)

	raw = h5read(joinpath(basename,"raw.h5"),"/main")
	central_ranges = [Int(ceil(p/2+1)) : Int(floor(s-p/2-1)) for (p,s) in zip(patch_size, full_size)]
	valid_mask = zeros(full_size)
	valid_mask[central_ranges...] = 1

	samples = gen_samples(mean_labels, patch_size = patch_size, N=400000, mask=valid_mask, M=30)
	Save.save(joinpath(basename,"samples.h5"), samples)
	Save.save(joinpath(basename,"filtered_samples.h5"), filter_samples(samples))

	affinities = h5read(joinpath(basename,"aff.h5"),"/main")
	@time vertices,edges = compute_regiongraph(raw, mean_labels, affinities, threshold=0.3)
	Save.save(joinpath(basename,"vertices.h5"), vertices)
	Save.save(joinpath(basename,"edges.h5"), edges)
end

basename = expanduser("~/mydatasets/golden")
