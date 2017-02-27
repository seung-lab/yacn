using Save
using HDF5
using BigArrays
using BigArrays.AlignedBigArrays

include("utils.jl")
include("reweight2.jl")
include("downsample.jl")

xindices = [14977:17024,16513:18560,18049:20096]
yindices = [27265:29312,28801:30848,30337:32384]
zindices = [4003:4258]
prefix = expanduser("~/seungmount/Omni/TracerTasks/pinky/proofreading/")

function do_prep(basename, patch_size = (318,318,33))
	valid = to_indicator(parse_valid_file(joinpath(basename,"valid.txt")))
	Save.save(joinpath(basename,"valid.h5"),valid)
	mean_labels = h5read(joinpath(basename,"mean_agg_tr.h5"),"/main")
	full_size = size(mean_labels)

	raw = h5read(joinpath(basename,"raw.h5"),"/main")
	central_ranges = [Int(ceil(p/2+1)) : Int(floor(s-p/2-1)) for (p,s) in zip(patch_size, full_size)]
	valid_mask = zeros(full_size)
	valid_mask[central_ranges...] = map(i->valid[i+1], raw)[central_ranges...]

	samples = gen_samples(mean_labels, patch_size = patch_size, N=400000, mask=valid_mask, M=30)
	Save.save(joinpath(basename,"samples.h5"), samples)
end

function downsample(basename)
	basename_ds=joinpath(basename,"ds/")
	mkpath(basename_ds)

	function downsampler(f)
		function g(vol)
			v=Save.load(joinpath(basename, vol))
			vds = f(v)
			Save.save(joinpath(basename_ds,vol),vds)
		end
		return g
	end

	#map(downsampler(downsample_volume),["mean_agg_tr.h5", "proofread.h5", "raw.h5", "image.h5"])
	#map(downsampler(x->x),["valid.h5"])
	map(downsampler(downsample_samples),["samples.h5"])
end

registerFile = "/usr/people/jzung/titan01/datasets/pinky/4_aligned/registry.txt"
ba = AlignedBigArray(registerFile)
function fetch_image(ranges...)
	img = ba[ranges[1],ranges[2],3:258]
end

for rx in xindices, ry in yindices, rz in zindices
	basename=joinpath(prefix,"chunk_$(rx.start)-$(rx.stop)_$(ry.start)-$(ry.stop)_$(rz.start)-$(rz.stop).omni.files")
	println(basename)
	#img = fetch_image(rx,ry,rz)
	#Save.save(joinpath(basename,"image.h5"), img)
	do_prep(basename)
	downsample(basename)
end
