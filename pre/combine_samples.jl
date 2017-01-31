using Save

function load_weights_samples(s)
	return (Save.load("$(s)_labels.h5"), Save.load("$(s)_samples.h5"))
end
function save_weights_samples(s,x)
	println(typeof(x[1]))
	println(typeof(x[2]))
	Save.save("$(s)_labels.h5",x[1])
	Save.save("$(s)_samples.h5",x[2])
end
function combine{T}(x::Tuple{Array{T,3},Array{Int,2}},y::Tuple{Array{T,3},Array{Int,2}})
	A1,s1 = x
	A2,s2 = y

	N=maximum(A1)
	A3 = cat(3,A1,map(x-> x==zero(T) ? zero(T) : T(N+x), A2))
	s3 = cat(2, s1, reshape(Int[0,0,size(A1,3)],(3,1)).+s2)
	return (A3,s3)
end
base_dir = expanduser("~/datasets/s1_block")
combined = foldr(combine, [load_weights_samples("$(base_dir)/mean$(thresh)") for thresh in [0.1,0.2,0.6]])
save_weights_samples("$(base_dir)/mean_combined",combined)

