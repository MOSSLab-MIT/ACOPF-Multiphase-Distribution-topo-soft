using Test

const ROOT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATAGEN_DIR = joinpath(ROOT_DIR, "datagen_topo")

include(joinpath(DATAGEN_DIR, "datagen_topo.jl"))

const CASE_PATHS = Dict(
    "IEEE13Nodeckt" => joinpath(DATAGEN_DIR, "grid_data", "13_bus", "IEEE13Nodeckt.dss"),
    "IEEE123Master" => joinpath(DATAGEN_DIR, "grid_data", "123_bus", "IEEE123Master.dss"),
)

@testset "branch_imax fallback uses pu units" begin
    saw_fallback_case = false

    for (case_name, case_path) in CASE_PATHS
        @testset "case: $case_name" begin
            network = FeederFlow.parse_file(case_path; regulator_model=:nonideal)
            i_base = network.base.Sbase / network.base.Vbase
            fallback_imax_pu = IMAX_DEFAULT_AMPS / i_base

            expected_branch_imax = Float64[]
            fallback_rows = Int[]

            for line in network.lines
                (line.is_switch || !line.is_closed) && continue

                admittance = FeederFlow.switch_line_admittance(network, line)
                admittance === nothing && continue

                imax_raw = max(Float64(line.normamps), Float64(line.emergamps))
                if imax_raw > 0.0
                    push!(expected_branch_imax, imax_raw)
                else
                    push!(expected_branch_imax, fallback_imax_pu)
                    push!(fallback_rows, length(expected_branch_imax))
                end
            end

            data = create_network_data(case_path)
            @test length(data.branch_imax) == length(expected_branch_imax)
            @test data.branch_imax ≈ expected_branch_imax atol=1e-12 rtol=1e-12

            if !isempty(fallback_rows)
                saw_fallback_case = true
                for idx in fallback_rows
                    @test data.branch_imax[idx] ≈ fallback_imax_pu atol=1e-12 rtol=1e-12
                end
            end
        end
    end

    # Fallback path is feeder-dependent; not all retained cases use IMAX_DEFAULT_AMPS.
    if !saw_fallback_case
        @warn "No case exercised IMAX_DEFAULT_AMPS fallback in this run"
    end
end
