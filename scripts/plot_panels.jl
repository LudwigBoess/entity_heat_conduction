println("loading packages")
using CairoMakie, MakieHelper
using Printf
using StatsBase
using EntityTools
using Roots
println("done")


function read_field_data(info, t, x_l, dim=2)

    data = read_field(info, ["N",
                            "B1", "B2", "B3", 
                            "E1", "E2", "E3", "T00", "Qx"], verbose=true; t)

    #Xc = [0.5 * (data["X1"][i] + data["X1"][i+1]) for i = 1:length(data["X1"])-1]

    # v_bulk = data["V1_1"].^2 .+ data["V2_1"].^2 .+ data["V3_1"].^2
    # γ_bar = @. data["T00_1"] / data["N_1"]
    # Γ = @. 1 / sqrt(1 - v_bulk)
    # A = 1.187
    # B = 1.251
    # C = 0.714
    # D = 0.936
    # γ_e(E_int) = (A + B*E_int) / ( C + D * E_int )
    # E(E_int, j) = (γ_bar[j] - Γ[j]) * Γ[j] / ( 1 + γ_e(E_int) * (Γ[j]^2 - 1) ) - E_int
    # T = similar(data["T00_1"])
    # for i in eachindex(T)
    #     _E(E_int) = E(E_int, i)
    #     T[i] = find_zero(_E, (-1000000.0, 1000000.0), verbose=false)
    # end

    # T .*= 2/3
    T = data["Qx"]


    println("maximum(T) = ", maximum(T))
    println("maximum(B) = ", maximum(data["B1"]), " ", maximum(data["B2"]), " ", maximum(data["B3"]))
    println("maximum(E) = ", maximum(data["E1"]), " ", maximum(data["E2"]), " ", maximum(data["E3"]))
    println("maximum(N) = ", maximum(data["N"]))
    if dim == 2
        return data["X1"], data["X1"], data["X2"], data["N"], data["E1"], data["E2"], data["E3"], data["B1"], data["B2"], data["B3"], T, data["t"], data["step"]
    elseif dim == 3
        return data["X1"], data["X1"], data["X2"], data["N"][:,:,128], 
                data["E1"][:,:,128], data["E2"][:,:,128], data["E3"][:,:,128], 
                data["B1"][:,:,128], data["B2"][:,:,128], data["B3"][:,:,128], 
                T[:,:,128], data["t"], data["step"]
    end
end

function round_to_next_100(x)
    return ceil(x / 100) * 100
end
γ = 5/3
r(M) = (γ + 1)/ ( γ - 1 + 2/M^2)   


function plot_shock_wide(sim_path, t, dim)


    println(t)
    flush(stdout)
    flush(stderr)

    scale_factor = 1

    #dx = 900.0
    Escale = Symlog10(1.e-9, linscale=0.5)
    Bscale = Symlog10(10.0)
    f = Figure(size=(1000, 600))

    e_lim = (-0.01, 0.01)
    B_lim = (-0.05, 0.05)
    y_l = (0.0, 25.0)
    x_l = (0.0, 125.0)
    x_factor = 0.0

    @info "reading data"
    info = EntityData(sim_path)
    #x_ticks = round_to_next_100(x_l[1]):10:round_to_next_100(x_l[2])

    X, Xc, Y, N, E1, E2, E3, B1, B2, B3, T, t, step = read_field_data(info, t, x_l, dim)

    println("maximum(B) = ", maximum(B1), " ", maximum(B2), " ", maximum(B3))
    println("maximum(E) = ", maximum(E1), " ", maximum(E2), " ", maximum(E3))
    sel = findall( x_l[1] .<= Xc .* scale_factor .<= x_l[2])
    #sel = 1:size(E1, 1)
    @info "density"
    # density
    rho_lim = (0.9, 2.1)
    Colorbar(f[1, 2], limits=rho_lim, colormap=:viridis, label=L"N", vertical=true, tickalign=1)
    ax = Axis(f[1, 1], ylabel=L"y \:\: \left[ c/\omega_{pe} \right]", aspect = DataAspect(),
            title = L"t = %$(round(t; digits=2)) \: \omega_e t")
    xlims!(ax, x_l )
    ylims!(ax, y_l )
    image!(ax, x_l, y_l, N, colormap=:viridis, colorrange=rho_lim)
    hidexdecorations!(ax, ticks = false)


    ax = Axis(f[2, 1], aspect=((x_l[2]-x_l[1])/sum(abs.(y_l))), ylabel="N", xlabel=L"x \:\: \left[ c/\omega_{pe} \right]")
    xlims!(ax, x_l )
    ylims!(ax, rho_lim)
    rho = mean(eachcol(N))
    x_mean = Xc .* scale_factor
    lines!(ax, x_mean, rho)

    @info "E"
    # electric field
    Colorbar(f[3:5, 2], limits=e_lim, colormap=:RdBu, label=L"E \:\: \left[ \sqrt{8 \pi n_b m_e c^2} \right]", tickalign=1)#, scale=Escale, ticks=[-0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1])
    ax = Axis(f[3, 1], ylabel=L"y \:\: \left[ c/\omega_{pe} \right]", 
                aspect = DataAspect())
    xlims!(ax, x_l )
    ylims!(ax, y_l )
    image!(ax, x_l, y_l, E1, colormap=:RdBu, colorrange=e_lim)#, colorscale=Escale)
    text!(ax, x_l[2]-x_factor, 0.2*y_l[2], text=L"E_x", color=:black)
    hidexdecorations!(ax, ticks = false)

    ax = Axis(f[4, 1], ylabel=L"y \:\: \left[ c/\omega_{pe} \right]", aspect = DataAspect())
    xlims!(ax, x_l )
    ylims!(ax, y_l )
    image!(ax, x_l, y_l, E2, colormap=:RdBu, colorrange=e_lim)#, colorscale=Escale)
    text!(ax, x_l[2]-x_factor, 0.2*y_l[2], text=L"E_y", color=:black)
    hidexdecorations!(ax, ticks = false)

    ax = Axis(f[5, 1], ylabel=L"y \:\: \left[ c/\omega_{pe} \right]", aspect = DataAspect())
    xlims!(ax, x_l )
    ylims!(ax, y_l )
    image!(ax, x_l, y_l, E3, colormap=:RdBu, colorrange=e_lim)#, colorscale=Escale)
    text!(ax, x_l[2]-x_factor, 0.2*y_l[2], text=L"E_z", color=:black)
    hidexdecorations!(ax, ticks = false)

    ax = Axis(f[6, 1], aspect=((x_l[2]-x_l[1])/sum(abs.(y_l))), xlabel=L"x \:\: \left[ c/\omega_{pe} \right]", ylabel=L"E")#,
                #yscale=Escale, yticks=[-0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1])
    xlims!(ax, x_l )
    ylims!(ax, e_lim)
    lines!(ax, x_mean, mean(eachcol(E1)), label=L"E_x")
    lines!(ax, x_mean, mean(eachcol(E2)), label=L"E_y")
    lines!(ax, x_mean, mean(eachcol(E3)), label=L"E_z")
    Legend(f[6, 2], ax, framevisible=false)

    @info "T_22"
    # obliquity
    theta_lim = (0.00, 0.01)
    Colorbar(f[1, 4], limits=theta_lim, colormap=:magma, label=L"q_\parallel", tickalign=1)
    ax = Axis(f[1, 3], ylabel=L"y \:\: \left[ c/\omega_{pe} \right]", xticks = 0:100:1000, aspect = DataAspect())
    xlims!(ax, x_l )
    ylims!(ax, y_l )
    image!(ax, x_l, y_l, T, colormap=:magma, colorrange=theta_lim)
    hidexdecorations!(ax, ticks = false)

    ax = Axis(f[2, 3], aspect=((x_l[2]-x_l[1])/sum(abs.(y_l))), xlabel=L"x \:\: \left[ c/\omega_{pe} \right]", ylabel=L"q_\parallel")
    xlims!(ax, x_l )
    ylims!(ax, theta_lim)
    lines!(ax, x_mean, mean(eachcol(T)))

    theta = deg2rad(0.0)
    phi = 0.0

    @info "B"
    # magnetic field
    #b_lim = (-5.0, 5.0)
    #b_lim = (-10.0, 10.0)
    b_lim = B_lim
    #b_lim = (-2.5, 2.5)

    B1 .-= cos(theta)
    B2 .-= sin(theta) * sin(phi)
    B3 .-= sin(theta) * cos(phi)

    Colorbar(f[3:5, 4], limits=b_lim, colormap=:PiYG, label=L"B - B_0 \:\: \left[ \sqrt{4 \pi n_b m_e c^2} \right]", tickalign=1)
    ax = Axis(f[3, 3], ylabel=L"y \:\: \left[ c/\omega_{pe} \right]", aspect = DataAspect())
    xlims!(ax, x_l )
    ylims!(ax, y_l )
    image!(ax, x_l, y_l, B1, colormap=:PiYG, colorrange=b_lim)
    text!(ax, x_l[2]-x_factor, 0.2*y_l[2], text=L"B_x", color=:black)
    hidexdecorations!(ax, ticks = false)

    ax = Axis(f[4, 3], ylabel=L"y \:\: \left[ c/\omega_{pe} \right]", aspect = DataAspect())
    xlims!(ax, x_l )
    ylims!(ax, y_l )
    image!(ax, x_l, y_l, B2, colormap=:PiYG, colorrange=b_lim)
    text!(ax, x_l[2]-x_factor, 0.2*y_l[2], text=L"B_y", color=:black)
    hidexdecorations!(ax, ticks = false)

    ax = Axis(f[5, 3], ylabel=L"y \:\: \left[ c/\omega_{pe} \right]", aspect = DataAspect())
    xlims!(ax, x_l )
    ylims!(ax, y_l )
    image!(ax, x_l, y_l, B3, colormap=:PiYG, colorrange=b_lim)
    text!(ax, x_l[2]-x_factor, 0.2*y_l[2], text=L"B_z", color=:black)
    hidexdecorations!(ax, ticks = false)

    ax = Axis(f[6, 3], aspect=((x_l[2]-x_l[1])/sum(abs.(y_l))), xlabel=L"x \:\: \left[ c/\omega_{pe} \right]", ylabel=L"B - B_0")
    xlims!(ax, x_l )
    ylims!(ax, b_lim)
    lines!(ax, x_mean, mean(eachcol(B1)), label=L"B_x")
    lines!(ax, x_mean, mean(eachcol(B2)), label=L"B_y")
    lines!(ax, x_mean, mean(eachcol(B3)), label=L"B_z")
    Legend(f[6, 4], ax, framevisible=false)


    @info "saving"
    save(sim_path * "Plots/fields_$(@sprintf("%08i", step)).png", f)

    @info "done"
    # de-allocate data (just for caution)
    X = Y = N = E1 = E2 = E3 = B1 = B2 = B3 = T = nothing
    GC.gc()
end

set_dark_theme!()

sim_path = "/lus/flare/projects/CRsInCosmoSims/PIC/Paper/heat_conduction/runs/2D/komarov/_conduction/"
info = EntityData(sim_path)

dim = 2
steps = 1000:10:1000
for step in steps
    plot_shock_wide(sim_path, step, dim)
end
