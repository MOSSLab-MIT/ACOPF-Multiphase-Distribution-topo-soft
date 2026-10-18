function parse_bool(s::AbstractString)
    v = lowercase(strip(s))
    if v in ("1", "true", "yes", "y", "on")
        return true
    end
    if v in ("0", "false", "no", "n", "off")
        return false
    end
    error("Cannot parse Bool from '$s'. Expected one of: true/false, 1/0, yes/no.")
end

function parse_cli_tokens(args::Vector{String})
    positional = String[]
    flags = Dict{String, String}()
    i = 1
    while i <= length(args)
        token = args[i]
        if startswith(token, "--")
            body = token[3:end]
            if isempty(body)
                error("Invalid flag: '$token'")
            end
            if occursin("=", body)
                k, v = split(body, "="; limit=2)
                flags[k] = v
                i += 1
                continue
            end
            if i == length(args) || startswith(args[i + 1], "--")
                flags[body] = "true"
                i += 1
            else
                flags[body] = args[i + 1]
                i += 2
            end
        else
            push!(positional, token)
            i += 1
        end
    end
    return positional, flags
end
