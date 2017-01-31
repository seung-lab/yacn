using HDF5
using Save

include("reweight2.jl")

ranges=(Colon(),Colon(),Colon())
#ranges=(1:800,1:800,1:100)

base_dir = expanduser("~/datasets/pinky_proofreading/ds")

mean_labels = h5read("$(base_dir)/mean_labels.h5","/main",ranges)
valid = h5read("$(base_dir)/valid.h5","/main",ranges)
println("loaded")
for i in eachindex(mean_labels, valid)
	if valid[i]!=2
		mean_labels[i]=0
	end
end
valid=0
gc()
println(size(mean_labels))

samples = gen_samples(mean_labels, patch_size=[314,314,48], N=500000)
function stars(mean_labels,samples)
	tmp=zeros(UInt32,size(mean_labels))
	for i in 1:size(samples,2)
		tmp[samples[1,i],samples[2,i],samples[3,i]]=1
	end
	return tmp
end
#Save.save("$(base_dir)/mean_samples_stars.h5",stars(mean_labels, samples))
Save.save("$(base_dir)/mean_samples.h5",samples)
