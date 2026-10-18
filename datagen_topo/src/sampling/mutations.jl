## Internal functions to set loads and DERs

function _set_loads!(data, pd, qd)
    L = data.n_net
    length(pd) == L || throw(DimensionMismatch())
    length(qd) == L || throw(DimensionMismatch())
    
    data.p_load .= pd
    data.q_load .= qd

    return nothing
end

function _set_DERs!(data, pder_avai)
    length(pder_avai) == data.n_der_unit || throw(DimensionMismatch())
    data.der_pavail_ref .= pder_avai
    return nothing
end 

function _set_obj_weights!(data, weights)
    length(weights) == length(data.obj_weights) || throw(DimensionMismatch())
    data.obj_weights .= weights
    return nothing
end
