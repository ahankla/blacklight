// Blacklight geodesic integrator

// C++ headers
#include <cmath>     // abs, acos, cos, sqrt
#include <optional>  // optional
#include <string>    // string

// Library headers
#include <omp.h>  // omp_get_wtime

// Blacklight headers
#include "geodesic_integrator.hpp"
#include "../blacklight.hpp"                                 // enums
#include "../input_reader/input_reader.hpp"                  // InputReader
#include "../radiation_integrator/radiation_integrator.hpp"  // RadiationIntegrator
#include "../utils/exceptions.hpp"                           // BlacklightException

//--------------------------------------------------------------------------------------------------

// Geodesic integrator constructor
// Inputs:
//   p_input_reader: pointer to object containing input parameters
GeodesicIntegrator::GeodesicIntegrator(const InputReader *p_input_reader)
{
  // Copy general input data
  model_type = p_input_reader->model_type.value();

  // Validate opt-in sampling assumptions before tracing or reading checkpoints.
  if (p_input_reader->sample_spatial_reuse.value_or(false)
      and (model_type != ModelType::simulation
          or p_input_reader->simulation_format.value() != SimulationFormat::athena
          or not p_input_reader->simulation_interp.value_or(false)
          or p_input_reader->simulation_block_interp.value_or(false)
          or not p_input_reader->slow_light_on.value_or(false)
          or not p_input_reader->slow_interp.value_or(false)))
    throw BlacklightException("sample_spatial_reuse requires a fixed Athena grid, "
        "simulation_interp=true, simulation_block_interp=false, "
        "slow_light_on=true and slow_interp=true.");

  // Reject incompatible checkpoint layouts before Integrate can load them.
  if (p_input_reader->sample_spatial_reuse.value_or(false)
      and (p_input_reader->checkpoint_geodesic_load.value_or(false)
          or p_input_reader->checkpoint_geodesic_save.value_or(false)
          or p_input_reader->checkpoint_sample_load.value_or(false)
          or p_input_reader->checkpoint_sample_save.value_or(false)))
    throw BlacklightException("sample_spatial_reuse is incompatible with checkpoints.");

  // Optional storage-only optimization: omitted radial samples have zero scalar
  // emissivity/absorption. Vacuum polarization transport and path diagnostics do
  // not share this invariance. Checkpoints do not record the compacted layout.
  ray_sample_cut = p_input_reader->ray_sample_cut.value_or(false);
  if (ray_sample_cut)
  {
    if (model_type != ModelType::simulation
        or p_input_reader->image_polarization.value_or(false)
        or p_input_reader->render_num_images.value_or(0) != 0
        or not (p_input_reader->image_light.value_or(false)
            or p_input_reader->image_tau.value_or(false))
        or p_input_reader->image_time.value_or(false)
        or p_input_reader->image_length.value_or(false)
        or p_input_reader->image_lambda.value_or(false)
        or p_input_reader->image_emission.value_or(false)
        or p_input_reader->image_lambda_ave.value_or(false)
        or p_input_reader->image_emission_ave.value_or(false)
        or p_input_reader->image_tau_int.value_or(false)
        or p_input_reader->image_crossings.value_or(false))
      throw BlacklightException("ray_sample_cut requires simulation, unpolarized I_nu/tau "
          "only, no rendering or path diagnostics.");
    if (p_input_reader->checkpoint_geodesic_save.value_or(false)
        or p_input_reader->checkpoint_geodesic_load.value_or(false)
        or p_input_reader->checkpoint_sample_save.value_or(false)
        or p_input_reader->checkpoint_sample_load.value_or(false))
      throw BlacklightException("ray_sample_cut is incompatible with geodesic/sample checkpoints.");
    sample_cut_omit_in = p_input_reader->cut_omit_in.value_or(-1.0);
    sample_cut_omit_out = p_input_reader->cut_omit_out.value_or(-1.0);
  }

  // Copy checkpoint parameters
  checkpoint_geodesic_save = p_input_reader->checkpoint_geodesic_save.value();
  checkpoint_geodesic_load = p_input_reader->checkpoint_geodesic_load.value();
  if (checkpoint_geodesic_save and checkpoint_geodesic_load)
    throw BlacklightException("Cannot both save and load a geodesic checkpoint.");
  if (checkpoint_geodesic_save or checkpoint_geodesic_load)
    checkpoint_geodesic_file = p_input_reader->checkpoint_geodesic_file.value();

  // Copy camera parameters
  camera_type = p_input_reader->camera_type.value();
  camera_r = p_input_reader->camera_r.value();
  camera_th = p_input_reader->camera_th.value();
  camera_ph = p_input_reader->camera_ph.value();
  camera_urn = p_input_reader->camera_urn.value();
  camera_uthn = p_input_reader->camera_uthn.value();
  camera_uphn = p_input_reader->camera_uphn.value();
  camera_k_r = p_input_reader->camera_k_r.value();
  camera_k_th = p_input_reader->camera_k_th.value();
  camera_k_ph = p_input_reader->camera_k_ph.value();
  camera_rotation = p_input_reader->camera_rotation.value();
  camera_width = p_input_reader->camera_width.value();
  camera_resolution = p_input_reader->camera_resolution.value();
  if (camera_resolution <= 0)
    throw BlacklightException("Must have positive camera_resolution.");
  camera_pole = p_input_reader->camera_pole.value();

  // Copy ray-tracing parameters
  ray_flat = p_input_reader->ray_flat.value();
  ray_terminate = p_input_reader->ray_terminate.value();
  if (ray_terminate != RayTerminate::photon)
    ray_factor = p_input_reader->ray_factor.value();
  ray_integrator = p_input_reader->ray_integrator.value();
  ray_step = p_input_reader->ray_step.value();
  ray_max_steps = p_input_reader->ray_max_steps.value();
  if (ray_max_steps <= 0)
    throw BlacklightException("Must have positive ray_max_steps.");
  if (ray_integrator == RayIntegrator::dp)
  {
    ray_max_retries = p_input_reader->ray_max_retries.value();
    if (ray_max_retries <= 0)
      throw BlacklightException("Must have nonnegative ray_max_retries.");
    ray_tol_abs = p_input_reader->ray_tol_abs.value();
    ray_tol_rel = p_input_reader->ray_tol_rel.value();
  }

  // Copy image parameters
  image_num_frequencies = p_input_reader->image_num_frequencies.value();
  if (image_num_frequencies == 1)
  {
    image_frequency = p_input_reader->image_frequency.value();
    if (image_frequency <= 0.0)
      throw BlacklightException("Must choose positive image_frequency.");
  }
  else if (image_num_frequencies > 1)
  {
    image_frequency_start = p_input_reader->image_frequency_start.value();
    if (image_frequency_start <= 0.0)
      throw BlacklightException("Must choose positive image_frequency_start.");
    image_frequency_end = p_input_reader->image_frequency_end.value();
    if (image_frequency_end <= 0.0)
      throw BlacklightException("Must choose positive image_frequency_end.");
    image_frequency_spacing = p_input_reader->image_frequency_spacing.value();
  }
  else
    throw BlacklightException("Must have positive image_num_frequencies.");
  image_normalization = p_input_reader->image_normalization.value();

  // Copy adaptive parameters
  adaptive_max_level = p_input_reader->adaptive_max_level.value();
  if (adaptive_max_level > 0)
  {
    adaptive_block_size = p_input_reader->adaptive_block_size.value();
    if (adaptive_block_size <= 0)
      throw BlacklightException("Must have positive adaptive_block_size.");
    if (camera_resolution % adaptive_block_size != 0)
      throw BlacklightException("Must have adaptive_block_size divide camera_resolution.");
  }

  // Set and calculate geometry data
  if (model_type == ModelType::simulation)
  {
    bh_m = 1.0;
    bh_a = p_input_reader->simulation_a.value();
  }
  else if (model_type == ModelType::formula)
  {
    bh_m = 1.0;
    bh_a = p_input_reader->formula_spin.value();
  }
  r_horizon = bh_m + std::sqrt(bh_m * bh_m - bh_a * bh_a);
  if (ray_terminate == RayTerminate::photon)
    r_terminate = 2.0 * bh_m * (1.0 + std::cos(2.0 / 3.0 * std::acos(-std::abs(bh_a) / bh_m)));
  else if (ray_terminate == RayTerminate::multiplicative)
    r_terminate = r_horizon * ray_factor;
  else if (ray_terminate == RayTerminate::additive)
    r_terminate = r_horizon + ray_factor;

  // Calculate number of pixels
  camera_num_pix = camera_resolution * camera_resolution;

  // Allocate space for camera data
  camera_loc = new Array<int>[adaptive_max_level+1];
  camera_pos = new Array<double>[adaptive_max_level+1];
  camera_dir = new Array<double>[adaptive_max_level+1];

  // Allocate space for image data
  momentum_factors = new Array<double>[adaptive_max_level+1];

  // Allocate space for geodesic data
  geodesic_num_steps = new int[adaptive_max_level+1];
  layout_reused = new bool[adaptive_max_level+1]();
  sample_flags = new Array<bool>[adaptive_max_level+1];
  sample_num = new Array<int>[adaptive_max_level+1];
  sample_pos = new Array<double>[adaptive_max_level+1];
  sample_dir = new Array<double>[adaptive_max_level+1];
  sample_len = new Array<double>[adaptive_max_level+1];

  // Prepare bookkeeping for adaptive refinement
  if (adaptive_max_level > 0)
  {
    linear_root_blocks = camera_resolution / adaptive_block_size;
    block_num_pix = adaptive_block_size * adaptive_block_size;
    camera_loc[0].Allocate(linear_root_blocks * linear_root_blocks, 2);
    for (int block_v = 0, block = 0; block_v < linear_root_blocks; block_v++)
      for (int block_u = 0; block_u < linear_root_blocks; block_u++, block++)
      {
        camera_loc[0](block,0) = block_v;
        camera_loc[0](block,1) = block_u;
      }
  }
}

//--------------------------------------------------------------------------------------------------

// Geodesic integrator destructor
GeodesicIntegrator::~GeodesicIntegrator()
{
  for (int level = 0; level <= adaptive_max_level; level++)
  {
    camera_loc[level].Deallocate();
    camera_pos[level].Deallocate();
    camera_dir[level].Deallocate();
    momentum_factors[level].Deallocate();
    sample_flags[level].Deallocate();
    sample_num[level].Deallocate();
    sample_pos[level].Deallocate();
    sample_dir[level].Deallocate();
    sample_len[level].Deallocate();
  }
  delete[] camera_loc;
  delete[] camera_pos;
  delete[] camera_dir;
  delete[] momentum_factors;
  delete[] geodesic_num_steps;
  delete[] layout_reused;
  delete[] sample_flags;
  delete[] sample_num;
  delete[] sample_pos;
  delete[] sample_dir;
  delete[] sample_len;
}

//--------------------------------------------------------------------------------------------------

// Top-level function for integrating geodesics
// Inputs: (none)
// Outputs:
//   returned value: execution time in seconds
double GeodesicIntegrator::Integrate()
{
  // Prepare timer
  double time_start = omp_get_wtime();

  // Reset adaptive level counter
  adaptive_level = 0;

  // Load data from checkpoint
  if (checkpoint_geodesic_load)
    LoadGeodesics();

  // Calculate geodesics
  if (not checkpoint_geodesic_load)
  {
    InitializeCamera();
    if (ray_integrator == RayIntegrator::dp)
      IntegrateGeodesicsDP();
    else if (ray_integrator == RayIntegrator::rk4)
      IntegrateGeodesicsRK4();
    else if (ray_integrator == RayIntegrator::rk2)
      IntegrateGeodesicsRK2();
    ReverseGeodesics();
  }

  // Root rays are fixed for subsequent radiation frames of this object.
  layout_reused[0] = sample_pos[0].allocated and sample_dir[0].allocated
      and sample_len[0].allocated and sample_num[0].allocated;

  // Save data to checkpoint
  if (checkpoint_geodesic_save)
    SaveGeodesics();

  // Calculate elapsed time
  return omp_get_wtime() - time_start;
}

//--------------------------------------------------------------------------------------------------

// Function for augmenting rays at new refinement level
// Inputs:
//   p_radiation_integrator: pointer to object containing processed image
// Outputs:
//   returned value: execution time in seconds
// Notes:
//   Acquires values from RadiationIntegrator that were not available at construction.
double GeodesicIntegrator::AddGeodesics(const RadiationIntegrator *p_radiation_integrator)
{
  // Prepare timer
  double time_start = omp_get_wtime();

  // Radiation resets its level after every frame; this object retains geodesics.
  // Follow the requested level rather than incrementing across frame boundaries.
  adaptive_level = p_radiation_integrator->adaptive_level;
  if (adaptive_level < 1 or adaptive_level > adaptive_max_level)
    throw BlacklightException("Invalid requested adaptive geodesic level.");

  layout_reused[adaptive_level] = false;

  // Acquire refinement data
  block_counts = p_radiation_integrator->block_counts;
  refinement_flags = p_radiation_integrator->refinement_flags;

  // Camera and metric parameters are fixed for this object's lifetime. Reuse only
  // a complete cache with exactly the child ordering AugmentCamera would produce;
  // equal block counts alone do not imply equal rays. Deeper caches remain valid
  // independently: their absolute locations will be checked when requested.
  bool reuse = camera_loc[adaptive_level].allocated
      and camera_loc[adaptive_level].n2 == block_counts[adaptive_level]
      and camera_pos[adaptive_level].allocated and camera_dir[adaptive_level].allocated
      and momentum_factors[adaptive_level].allocated
      and sample_flags[adaptive_level].allocated and sample_num[adaptive_level].allocated
      and sample_pos[adaptive_level].allocated and sample_dir[adaptive_level].allocated
      and sample_len[adaptive_level].allocated;
  if (reuse)
  {
    int block = 0;
    for (int parent = 0; parent < block_counts[adaptive_level-1] and reuse; parent++)
      if (refinement_flags[adaptive_level-1](parent))
      {
        int y = 2 * camera_loc[adaptive_level-1](parent,0);
        int x = 2 * camera_loc[adaptive_level-1](parent,1);
        for (int dy = 0; dy < 2 and reuse; dy++)
          for (int dx = 0; dx < 2 and reuse; dx++, block++)
            if (block >= block_counts[adaptive_level]
                or camera_loc[adaptive_level](block,0) != y + dy
                or camera_loc[adaptive_level](block,1) != x + dx)
              reuse = false;
      }
    reuse = reuse and block == block_counts[adaptive_level];
  }
  if (reuse)
  {
    layout_reused[adaptive_level] = true;
    return omp_get_wtime() - time_start;
  }

  // Array::Allocate cannot replace an existing allocation. Discard this level's
  // old camera and reversed samples before rebuilding a changed layout. Radiation
  // sampling and coefficients are still recomputed by RadiationIntegrator each frame.
  camera_loc[adaptive_level].Deallocate();
  camera_pos[adaptive_level].Deallocate();
  camera_dir[adaptive_level].Deallocate();
  momentum_factors[adaptive_level].Deallocate();
  sample_flags[adaptive_level].Deallocate();
  sample_num[adaptive_level].Deallocate();
  sample_pos[adaptive_level].Deallocate();
  sample_dir[adaptive_level].Deallocate();
  sample_len[adaptive_level].Deallocate();

  // Calculate geodesics
  AugmentCamera();
  if (ray_integrator == RayIntegrator::dp)
    IntegrateGeodesicsDP();
  else if (ray_integrator == RayIntegrator::rk4)
    IntegrateGeodesicsRK4();
  else if (ray_integrator == RayIntegrator::rk2)
    IntegrateGeodesicsRK2();
  ReverseGeodesics();

  // Calculate elapsed time
  return omp_get_wtime() - time_start;
}
