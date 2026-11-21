// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/kinetics.hpp>
#include <kintera/kinetics/kinetics_formatter.hpp>
#include <kintera/thermo/relative_humidity.hpp>

// snap
#include <snap/mesh/meshblock.hpp>

#include "command_line.hpp"

using namespace snap;

int main(int argc, char **argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  // read parameters
  auto cli = CommandLine::ParseArguments(argc, argv);
  if (!cli) return 0;

  // input file
  auto infile = std::string(cli->input_filename);

  auto config = YAML::LoadFile(infile);
  auto Ps = config["problem"]["Ps"].as<double>(1.e5);
  auto Ts = config["problem"]["Ts"].as<double>(300.);
  auto Tmin = config["problem"]["Tmin"].as<double>(200.);
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  // initialize the block
  auto op_block = MeshBlockOptionsImpl::from_yaml(infile);
  auto block = MeshBlock(op_block);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available() && op_block->layout()->device() == "cuda") {
    std::cout << "Running on CUDA" << std::endl;
    int device_id = op_block->layout()->device_id();
    if (device_id < 0) device_id = op_block->layout()->local_rank();
    device = torch::Device(torch::kCUDA, device_id);
  }

  block->to(device);

  // useful modules
  auto phydro = block->phydro;
  auto pcoord = block->pcoord;
  auto peos = phydro->peos;
  auto m = block->named_modules()["hydro.eos.thermo"];
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);

  // dimensions and indices
  int nc3 = pcoord->x3v.size(0);
  int nc2 = pcoord->x2v.size(0);
  int nc1 = pcoord->x1v.size(0);
  int ny = thermo_y->options->species().size() - 1;
  int nvar = peos->nvar();

  // construct an adiabatic atmosphere
  kintera::ThermoX thermo_x(thermo_y->options);
  thermo_x->to(device);

  auto temp =
      Ts *
      torch::ones({nc3, nc2},
                  torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto pres =
      Ps *
      torch::ones({nc3, nc2},
                  torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto xfrac =
      torch::zeros({nc3, nc2, 1 + ny},
                   torch::TensorOptions().dtype(torch::kDouble).device(device));

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  // read in compositions
  for (int i = 1; i <= ny; ++i) {
    auto name = thermo_y->options->species()[i];
    auto xmixr = config["problem"]["x" + name].as<double>(0.);
    xfrac.select(2, i) = xmixr;
  }

  // dry air mole fraction
  xfrac.select(2, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  // adiabatic extrapolate half a grid to cell center
  int il = pcoord->il();
  int iu = pcoord->iu();
  auto dz = pcoord->dx1f[il].item<double>();
  thermo_x->extrapolate_dz(
      temp, pres, xfrac,
      kintera::ExtrapOptions().dz(dz / 2.).grav(grav).ds_dz(0.));

  int i = il;
  int nvapor = thermo_x->options->vapor_ids().size();
  int ncloud = thermo_x->options->cloud_ids().size();
  for (; i <= iu; ++i) {
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});

    w[IPR].select(2, i) = pres;
    w[IDN].select(2, i) = thermo_x->compute("V->D", {conc});

    auto result = thermo_x->compute("X->Y", {xfrac});
    w.narrow(0, ICY, ny).select(3, i) = thermo_x->compute("X->Y", {xfrac});

    if ((temp < Tmin).any().item<double>()) break;
    // step centre i -> i+1 by the centre spacing, not the cell width
    dz = pcoord->dx1v[i + 1].item<double>();
    thermo_x->extrapolate_dz(
        temp, pres, xfrac,
        kintera::ExtrapOptions().dz(dz).grav(grav).ds_dz(0.));
  }

  // isothermal extrapolation
  for (; i <= iu; ++i) {
    auto mu = (thermo_x->mu * xfrac).sum(-1);
    dz = pcoord->dx1v[i].item<double>();  // centre i-1 -> i
    pres *= exp(-grav * mu * dz / (kintera::constants::Rgas * temp));
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    w[IPR].select(2, i) = pres;
    w[IDN].select(2, i) = thermo_x->compute("V->D", {conc});
    w.narrow(0, ICY, ny).select(3, i) = thermo_x->compute("X->Y", {xfrac});
  }

  // add noise
  w[IVX] += 0.01 * torch::rand_like(w[IVX]);
  w[IVY] += 0.01 * torch::rand_like(w[IVY]);

  // initialize
  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  double current_time = block->initialize(vars, cli->restart_filename);

  // user output variables
  // (1) total precipitable mass fraction [kg/kg]
  block->user_output_callback = [&](Variables const &vars) {
    auto w = vars.at("hydro_w");
    Variables out;
    out["qtol"] = w.narrow(0, ICY, ny).sum(0);
    return out;
  };

  // the chemistry step runs inside the block's step (V3 S-K4)
  block->set_chemistry(*kintera::KineticsOptionsImpl::from_yaml(infile));

  // time loop
  if (cli->restart_filename == nullptr) {
    block->make_outputs(vars, current_time);
  }

  while (!block->pintg->stop(block->cycle, current_time)) {
    auto dt = block->max_time_step(vars);
    block->print_cycle_info(vars, current_time, dt);

    // evolve dynamics
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, dt, stage);
    }

    int err = block->check_redo(vars);
    if (err > 0) continue;  // redo this step with smaller dt
    if (err < 0) break;     // terminate simulation

    // make outputs
    ++block->cycle;
    current_time += dt;
    block->make_outputs(vars, current_time);
  }

  int status = block->finalize(vars, current_time);

  CommandLine::Destroy();
  return status;
}
