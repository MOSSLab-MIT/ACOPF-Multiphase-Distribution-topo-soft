using SparseArrays

function _jsonable(x)
    if x isa SparseMatrixCSC
        I, J, V = findnz(x)
        return Dict(
            "rows" => I,
            "cols" => J,
            "vals" => _jsonable(V),
            "shape" => [size(x)...],
        )
    elseif x isa Complex
        return Dict("re" => real(x), "im" => imag(x))
    elseif x isa AbstractDict
        out = Dict{String, Any}()
        for (k, v) in x
            key = k isa AbstractString ? String(k) : string(k)
            out[key] = _jsonable(v)
        end
        return out
    elseif x isa AbstractArray
        return map(_jsonable, x)
    elseif x isa Tuple
        return [_jsonable(v) for v in x]
    else
        return x
    end
end

function to_dict(data::OPFData)
    d = Dict{String, Any}()
    for field in fieldnames(OPFData)
        v = getfield(data, field)
        if isa(v, SparseMatrixCSC) 
            I, J, V = findnz(v)
            M, N = size(v)
            d[string(field)] = Dict("I" => I, "J" => J, "V" => V, "M" => M, "N" => N)
        elseif isa(v, AbstractArray)
            d[string(field)] = copy(v)
        else
            d[string(field)] = v
        end
    end
    return d
end


##
"""
Convert OPFData to a JSON-friendly dictionary with automatic field detection.
Handles scalars, vectors, strings, and sparse matrices automatically.
"""
function network_to_json_dict(network::OPFData)
    d = Dict{String, Any}()
    
    # Get all field names dynamically
    for field in fieldnames(OPFData)
        value = getfield(network, field)
        field_name = string(field)
        d[field_name] = _jsonable(value)
    end
    
    return d
end
