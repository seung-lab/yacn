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
