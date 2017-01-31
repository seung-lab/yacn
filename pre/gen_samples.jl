using Agglomeration
using Agglomerators
using SegmentationMetrics
using Process
using MergeTrees
using HDF5
using Save

include("reweight2.jl")

using Watershed

ranges=(Colon(),Colon(),Colon())

base_dir = expanduser("~/datasets/s1_block")

affinities=h5read("$(base_dir)/affinities.h5","/main",(ranges...,1:3))
machine_labels = Watershed.watershed(affinities, high=0.9999, low=0.1)
h5write("$(base_dir)/machine_labels.h5","/main",machine_labels)
machine_labels = h5read("$(base_dir)/machine_labels.h5","/main",ranges)

if false
	human_labels = h5read("$(base_dir)/human_labels.h5","/main",ranges)
	M=SegmentationMetrics.incidence_matrix(machine_labels, human_labels)
	g(x)=if x[1] > 0; x[2]; else 0; end
	relabels = Int[g(findmax(M[i,:])) for i in 1:size(M,1)]
	oracle_labels = map(i-> if i==0; 0; else relabels[i]; end, machine_labels)
	Save.save("$(base_dir)/oracle_labels.h5",oracle_labels)
	samples = gen_samples(oracle_labels)
	Save.save("$(base_dir)/oracle_samples.h5",samples)
else
	merge_tree = Process.forward2(affinities, machine_labels)
	#println(merge_tree)
	Save.save("$(base_dir)/maff_merge_tree.jls",merge_tree)
	for thresh in [0.1,0.2,0.6]
		mean_labels=MergeTrees.flatten(machine_labels,merge_tree,thresh)
		Save.save("$(base_dir)/mean$(thresh)_labels.h5",mean_labels)
		samples = gen_samples(mean_labels, patch_size=[314,314,48], N=500000)
		Save.save("$(base_dir)/mean$(thresh)_samples.h5",samples)
	end
end

