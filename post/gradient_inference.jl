include("forward.jl")
using HDF5, PyCall
using Base.Profile
using Base.Threads

const patch_size = Int[158,158,32]
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport adversarial
filename = chomp(readstring(`zenity --file-selection`))
filename_prefix = splitext(basename(filename))[1]
adversarial.main_model[:restore](filename)

set="test"
LOAD_DIR=expanduser("~/datasets/SNEMI3D/$(set)")
inpt = convert(Array{Float32,3},h5read("$(LOAD_DIR)/image.h5","/main"))#[1:300,1:300,1:40]
otpt = zeros(Float32, (size(inpt)...,2))
forward!(inpt,otpt,patch_size,x->(permutedims(adversarial.main_model[:compute_gradients](permutedims(x,[3,2,1])),[3,2,1,4])), step=0.99)

h5open("$(dirname(filename))/$(set)_loss_grad_$(filename_prefix).h5", "w") do file
	write(file, "/main", otpt)
end
