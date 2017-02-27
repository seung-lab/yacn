using DataFrames

function parse_valid_file(valid_file)
	f=readtable(valid_file,skipstart=2,header=false,names=Symbol[:seg_id,:status])
	d=Dict{Int,Int}()
	d[0]=0
	for (i,j) in zip(f[:seg_id],f[:status])
		d[i]=j
	end
	return d
end

function to_indicator(d)
	N=maximum(keys(d))
	A=zeros(Int,N+1)
	for i in 0:N
		if haskey(d,i) && d[i]==2
			#Argh, 1-indexing
			A[i+1]=1
		end
	end
	return A
end

function flatten_samples{T<:Vector}(A::Vector{T})
	A_flat=fill(0,(3,length(A)))
	for i in 1:length(A)
		A_flat[:,i]=A[i]
	end
	return A_flat
end
