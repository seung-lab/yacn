include("forward.jl")
using HDF5, PyCall
using Base.Profile
using Base.Threads

const patch_size = Int[158,158,32]
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport discriminate
filename = chomp(readstring(`zenity --file-selection`))
filename_prefix = splitext(basename(filename))[1]
discriminate.main_model[:restore](filename)

set="test"
LOAD_DIR=expanduser("~/datasets/AC3/$(set)")
inpt = convert(Array{Float32,3},h5read("$(LOAD_DIR)/mean_labels.h5","/main"))#[1:300,1:300,1:40]
otpt = zeros(Float32, (size(inpt)...,1))
forward_max!(inpt,otpt,patch_size,x->(permutedims(adversarial.main_model[:compute_discrim_magnitude](permutedims(x,[3,2,1])),[3,2,1,4])), step=0.3)
otpt = reshape(otpt,size(otpt)[1:3])

h5open("$(dirname(filename))/$(set)_discrim_$(filename_prefix).h5", "w") do file
	write(file, "/main", otpt)
end
