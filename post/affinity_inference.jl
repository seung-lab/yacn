include("forward.jl")
using HDF5, PyCall
using Base.Profile
using Base.Threads

const patch_size = Int[158,158,32]
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport adversarial
filename = chomp(readstring(`zenity --file-selection`))
adversarial.main_model[:restore](filename)

const boundary_vec=permutedims(adversarial.main_model[:get_boundary_vec](),[4,3,2,1])

function aff{T}(l1::Array{T,4},l2::Array{T,4})
	return reshape(exp(-0.5*sum((l1.-l2).^2,1)),tuple(patch_size...))
end

offsets1 = [(0,2,0,0),(0,0,2,0),(0,0,0,1)]
offsets2 = [(0,-1,0,0),(0,0,-1,0),(0,0,0,0)]
function grad{T}(l1::Array{T,4})
	boundary=1-aff(l1, boundary_vec)
	s=Any[0 for i in offsets1]
	@threads for i in 1:size(offsets1,1)
		o1=offsets1[i]
		o2=offsets2[i]
		s[i]=min(aff(circshift(l1,o1),circshift(l1,o2)),circshift(boundary,o1[2:4]),circshift(boundary,o2[2:4]))
	end
	return cat(4,s...)
end

set="test"
LOAD_DIR=expanduser("~/datasets/SNEMI3D/$(set)")
inpt = convert(Array{Float32,3},h5read("$(LOAD_DIR)/image.h5","/main"))#[1:300,1:300,1:40]
otpt = zeros(Float32, (size(inpt)...,3))
forward!(inpt,otpt,patch_size,x->grad(permutedims(adversarial.main_model[:compute_vector_labels](permutedims(x,[3,2,1])),[4,3,2,1])), step=0.33)

h5open("$(dirname(filename))/$(set)_diff.h5", "w") do file
	write(file, "/main", otpt)
end
