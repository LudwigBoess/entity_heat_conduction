using EntityTools
using CairoMakie, MakieHelper 
using Printf

function read_velocities(sim_path)

    info = EntityData(sim_path)
    d = read_particles(info, 1, ["U1", "U2"], t = 0.0)
    units = EntityUnits(info)

    vx = d["U1"] ./ EntityTools.vA(units)
    vy = d["U2"] ./ EntityTools.vA(units)

    return vx, vy
end

function get_phase_map(sim_path)

    vx, vy = read_velocities(sim_path)

    x_bins, y_bins, phase_map_count = phase_map(vx, vy, (-2.0, 2.0), (-2.0, 2.0), xbins=200, ybins=200)

    return x_bins, y_bins, phase_map_count
end

function plot_phase_map(sim_path, plot_name)

    x_bins, y_bins, phase_map_count = get_phase_map(sim_path)

    println("Max count in phase map: ", maximum(phase_map_count))
    clim = (1.0, 1.e6)
    fig = Figure()
    ax = Axis(fig[1, 1], title="Phase Map", xlabel=L"v_x / v_A", ylabel=L"v_y / v_A", aspect = 1)
    heatmap!(ax, x_bins, y_bins, phase_map_count', colormap=:viridis,
            colorscale=log10, colorrange=clim)
    #Colorbar(fig[1, 2], ax, label="Count")
    save(plot_name, fig)

end

sim_path = "/lus/flare/projects/CRsInCosmoSims/PIC/Paper/heat_conduction/runs/2D/RC/_conduction"
plot_name = "phase_map.png"
plot_phase_map(sim_path, plot_name)