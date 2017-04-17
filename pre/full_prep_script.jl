include("Save.jl")
using Save
using HDF5

include("utils.jl")
include("reweight2.jl")
include("downsample.jl")
include("compute_regiongraph.jl")
include("compute_fullgraph.jl")
include("filter_samples.jl")
include("compute_proofreadgraph.jl")

#TODO: Change this to read and write to a cloud filesystem
function do_prep(basename; patch_size = (318,318,33), ground_truth=false, compute_full_edges=false)
	basename=expanduser(basename)

	mean_labels = Save.load(joinpath(basename,"mean_agg_tr.h5"))
	full_size = size(mean_labels)
	println(full_size)

	raw = Save.load(joinpath(basename,"raw.h5"))
	#the sample around a point x is [x-floor(patch_size/2): x-floor(patch_size/2)+patch_size]
	central_ranges = [Int(floor(p/2) + 1) : Int(s - p + floor(p/2) + 1) for (p,s) in zip(patch_size, full_size)]
	println(map(length,central_ranges))
	println(patch_size)
	mask = zeros(Int8, full_size)
	mask[central_ranges...] = 1
	samples = gen_samples(mean_labels, patch_size, N=400000, mask=mask, M=30)
	Save.save(joinpath(basename,"samples.h5"), flatten(samples))

	vertices = unique(raw)
	Save.save(joinpath(basename,"vertices.h5"), vertices)

	affinities = Save.load(joinpath(basename,"aff.h5"))
	@time mean_edges = compute_regiongraph(raw, mean_labels, affinities, threshold=0.3)
	Save.save(joinpath(basename,"mean_edges.h5"), mean_edges)

	if compute_full_edges
		full_edges = compute_fullgraph(Batched(), raw, resolution=Int[4,4,40], radius=130, downsample=Int[4,4,1])
		Save.save(joinpath(basename,"full_edges.h5"), full_edges)
	end

	contact_edges = compute_contactgraph(raw)
	Save.save(joinpath(basename,"contact_edges.h5"), contact_edges)
	
	if ground_truth
		valid = to_indicator(parse_valid_file(joinpath(basename,"valid.txt")))
		Save.save(joinpath(basename,"valid.h5"),valid)

		#=
		vertices = Save.load(joinpath(basename,"vertices.h5"))
		full_edges = Save.load(joinpath(basename,"full_edges.h5"))
		proofread = Save.load(joinpath(basename,"proofread.h5"))
		proofread_edges = compute_proofreadgraph(raw,proofread,vertices,full_edges)
		Save.save(joinpath(basename,"proofread_edges.h5"), proofread_edges)
		=#

		samples = Save.load(joinpath(basename,"samples.h5"))
		samples = Tuple{Int,Int,Int}[tuple(samples[:,i]...) for i in 1:size(samples,2)]
		valid_samples = filter(x->(valid[raw[x[3]+1,x[2]+1,x[1]+1]+1]==1), samples)
		Save.save(joinpath(basename,"valid_samples.h5"),flatten(valid_samples))

		padded_valid_samples = filter(samples_filter(full_size, Int[patch_size...] + 2*[20,20,0]),valid_samples)

		println(map(length,[minimum([x[i] for x in samples]) : maximum([x[i] for x in samples])for i in 1:3]))
		println(map(length,[minimum([x[i] for x in padded_valid_samples]) : maximum([x[i] for x in padded_valid_samples])for i in 1:3]))

		println(length(padded_valid_samples))
		Save.save(joinpath(basename,"padded_valid_samples.h5"), flatten(padded_valid_samples[1:250000]))
	end
end

basename = expanduser(ARGS[1])
@time do_prep(basename, ground_truth=false)
