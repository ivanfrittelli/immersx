// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/numerics/data_out.h>

#include <immersx/core/constraint.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/linear_adapter.h>
#include <immersx/io/utils.h>
#include <immersx/physics/poisson.h>
#include <immersx/physics/poisson_residual.h>
#include <immersx/mg/coupled_chebyshev_smoother.h>
#include <immersx/mg/coupled_transfer.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <immersx/mg/1d/poisson.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace
{
  class CoupledPoissonApplicationParameters : public dealii::ParameterAcceptor
  {
  public:
    CoupledPoissonApplicationParameters()
      : ParameterAcceptor("/Coupled Poisson/")
    {
      add_parameter("Output directory", output_directory);
      add_parameter("Multiplier output name", multiplier_output_name);
    }

    std::string output_directory       = ".";
    std::string multiplier_output_name = "multiplier";
  };

  void
  run_coupled_poisson(const std::string &parameter_file)
  {
    using namespace ImmersX;

    CoupledPoissonApplicationParameters application_parameters;
    PoissonParameters<2>                bulk_parameters;
    PoissonParameters<1, 2> embedded_parameters("/Embedded Poisson/");
    LinearSolverParameters  adapter_parameters;
    adapter_parameters.preconditioner = LinearPreconditioner::augmented_lagrangian;
    initialize_parameters(parameter_file);

    bulk_parameters.output_directory = application_parameters.output_directory;
    embedded_parameters.output_directory =
      application_parameters.output_directory;

    PoissonSolver<2>    bulk_problem(bulk_parameters);
    PoissonSolver<1, 2> embedded_problem(embedded_parameters);

    const auto initialize_problem = [](auto &problem) {
      problem.make_grid();
      problem.setup_fe();
      problem.setup_system();
      problem.assemble_system();
    };
    initialize_problem(bulk_problem);
    initialize_problem(embedded_problem);

    using FieldVector  = ImmersXLA::MPI::Vector;
    using GlobalVector = ImmersXLA::MPI::BlockVector;
    using Adapter      = LinearAdapter<FieldVector, GlobalVector>;

    Adapter    adapter(adapter_parameters, MPI_COMM_WORLD);
    const auto bulk     = adapter.add(bulk_problem, "bulk");
    const auto embedded = adapter.add(embedded_problem, "embedded");

    const auto bulk_view = fe_space(bulk_problem.dof_handler(),
                                    StaticMappingQ1<2>::mapping,
                                    bulk_problem.constraints(),
                                    bulk_problem.locally_relevant_dofs());
    const auto embedded_view =
      fe_space(embedded_problem.dof_handler(),
               StaticMappingQ1<1, 2>::mapping,
               embedded_problem.constraints(),
               embedded_problem.locally_relevant_dofs());

    // The multiplier has its own FE/DoFHandler, while sharing the embedded
    // geometry with the line problem.  This deliberately exercises the
    // paired-DoFHandler weak-term path rather than particle search.
    FE_DGQ<1, 2>     multiplier_fe(0);
    DoFHandler<1, 2> multiplier_dh(embedded_problem.triangulation());
    multiplier_dh.distribute_dofs(multiplier_fe);
    const auto multiplier_owned = multiplier_dh.locally_owned_dofs();
    const auto multiplier_relevant =
      DoFTools::extract_locally_relevant_dofs(multiplier_dh);
    AffineConstraints<double> multiplier_constraints;
    multiplier_constraints.reinit(multiplier_owned, multiplier_relevant);
    multiplier_constraints.close();

    const auto multiplier_view = fe_space(multiplier_dh,
                                          StaticMappingQ1<1, 2>::mapping,
                                          multiplier_constraints,
                                          multiplier_relevant);
    const auto bulk_field =
      bulk_view.field(bulk.fields().solution, "bulk_solution");
    const auto embedded_field =
      embedded_view.field(embedded.fields().solution, "embedded_solution");
    const auto multiplier = multiplier_view.field("lambda");
    const auto constraint =
      make_constraint(weak_term(value(bulk_field), test(multiplier)) -
                      weak_term(value(embedded_field), test(multiplier)));
    const auto coupling = adapter.add(constraint, "continuity");

    //      const auto operator_view = composition_.jacobian(0., state, nullptr, 0.);

    auto state = adapter.make_state();
    //Usual way 
    //adapter.solve(state);

    //Constrained smoother as preconditioner
    /*
    auto rhs = adapter.make_state();
    adapter.evaluate_residual(state, rhs);
    rhs *= -1;

    auto system_matrix = adapter.block_matrix(state);

    //Constrained preconditioner
    TrilinosWrappers::PreconditionChebyshev::AdditionalData smoother_data_bg;
    smoother_data_bg.eigenvalue_ratio     = 15.;
    smoother_data_bg.degree              = 5;
    CoupledChebyshevSmoother preconditioner;
    preconditioner.initialize(system_matrix, smoother_data_bg);
    */

    //My way
    cout<< "Starting coupled solver \n";
    
    using VectorType      = ImmersX::LA::MPI::Vector;
    using Payload = dealii::internal::LinearOperatorImplementation::EmptyPayload;

    auto rhs = adapter.make_state();
    adapter.evaluate_residual(state, rhs);
    rhs *= -1;

    auto system_matrix = adapter.block_matrix(state);


    //I 3 sparsity pattern
    MGLevelObject<TrilinosWrappers::SparsityPattern> bg_mg_sparsity_patterns;
    MGLevelObject<TrilinosWrappers::SparsityPattern> coupling_mg_sparsity_patterns;
    MGLevelObject<TrilinosWrappers::SparsityPattern> fg_mg_sparsity_patterns;

    //Le cose particolari per il problema bg
    MGLevelObject<TrilinosWrappers::SparsityPattern> bg_mg_interface_sparsity_patterns;
    MGLevelObject<ImmersX::LA::MPI::BlockSparseMatrix> bg_mg_interface_matrices;
    MGConstrainedDoFs                   bg_mg_constrained_dofs;

    //Le matrici (a blocchi)
    MGLevelObject<ImmersX::LA::MPI::BlockSparseMatrix> mg_matrices;

    MGLevelObject<ImmersX::LA::MPI::BlockVector> mg_solutions;
    MGLevelObject<ImmersX::LA::MPI::BlockVector> mg_rhs;

    std::vector<ImmersX::LA::MPI::SparseMatrix> mg_coupling_matrix;
    std::vector<ImmersX::LA::MPI::SparseMatrix> mg_coupling_matrix_transpose;

    GraphPoissonParameters par;
    GraphPoisson<ImmersX::LA::MPI::SparseMatrix> immersed_problem(par);

    immersed_problem.make_grid();
    immersed_problem.setup_system();

    LinearOperator<VectorType, VectorType, Payload> A = 
      linear_operator<VectorType, VectorType, Payload>(system_matrix.block(0,0));


    auto A_2 = linear_operator<VectorType, VectorType, Payload>(immersed_problem.mg_levels[0] -> system_matrix);
    auto M_2 = linear_operator<VectorType, VectorType, Payload>(immersed_problem.mg_levels[0] -> mass_matrix);

    auto Zero = M_2 * 0.0;

    const auto Bt =
      linear_operator<VectorType, VectorType, Payload>(system_matrix.block(1,2));
    const auto B = transpose_operator<VectorType, VectorType, Payload>(Bt);

    //Transfer
    CoupledTransfer coupled_transfer;
    const auto & dh = bulk_problem.dof_handler();
    coupled_transfer.build(dh, embedded_problem.dof_handler());
    const auto & tria = bulk_problem.triangulation();

    for (unsigned int level = 1; level < tria.n_levels(); level++) {
      mg_matrices[level].block(0,2).copy_from(mg_coupling_matrix[level-1]);
      mg_matrices[level].block(2,0).copy_from(mg_coupling_matrix_transpose[level-1]);

      mg_matrices[level].block(1,1).copy_from(immersed_problem.mg_levels[0] -> system_matrix);

      mg_matrices[level].block(2,1).copy_from(immersed_problem.mg_levels[0] -> mass_matrix);
      mg_matrices[level].block(1,2).copy_from(immersed_problem.mg_levels[0] -> mass_matrix);

      mg_matrices[level].block(2,2).reinit(immersed_problem.mg_levels[0] -> mass_matrix);
      mg_matrices[level].block(2,2) = 0;

      mg_matrices[level].block(0,1).reinit(mg_matrices[level].block(0,2));
      mg_matrices[level].block(1,0).reinit(mg_matrices[level].block(2,0));
      mg_matrices[level].block(0,1) = 0;
      mg_matrices[level].block(1,0) = 0;
    }

    //Smoothers
    TrilinosWrappers::PreconditionChebyshev::AdditionalData smoother_data_bg;
    smoother_data_bg.eigenvalue_ratio     = 15.;
    smoother_data_bg.degree              = 5;

    MGSmootherPrecondition<ImmersX::LA::MPI::BlockSparseMatrix, CoupledChebyshevSmoother, ImmersX::LA::MPI::BlockVector> mg_smoother;
    mg_smoother.initialize(mg_matrices, smoother_data_bg);
    mg_smoother.set_steps(2);
    mg_smoother.set_symmetric(true);

    MGLevelObject<CoupledChebyshevSmoother> smoothers;
    smoothers.resize(1, tria.n_levels()-1);

    for (unsigned int level = 1; level < tria.n_levels(); level++) {
      smoothers[level].initialize(mg_matrices[level], smoother_data_bg);
    }

    mg::Matrix<ImmersX::LA::MPI::BlockVector> mg_matrix(mg_matrices);
    mg::Matrix<ImmersX::LA::MPI::BlockVector> mg_interface_up(bg_mg_interface_matrices);
    mg::Matrix<ImmersX::LA::MPI::BlockVector> mg_interface_down(bg_mg_interface_matrices);

    //Coarse grid solver
    PreconditionIdentity coarse_solver_preconditioner;
    SolverControl coarse_solver_control(1000, 1e-10);
    SolverGMRES<ImmersX::LA::MPI::BlockVector> coarse_solver(coarse_solver_control);
    MGCoarseGridIterativeSolver<ImmersX::LA::MPI::BlockVector, SolverGMRES<ImmersX::LA::MPI::BlockVector>, ImmersX::LA::MPI::BlockSparseMatrix, PreconditionIdentity> 
      coarse_grid_solver(coarse_solver, mg_matrices[1], coarse_solver_preconditioner);

    //Oggetto multigrid
    Multigrid<ImmersX::LA::MPI::BlockVector> mg(
      mg_matrix, coarse_grid_solver, coupled_transfer, mg_smoother, mg_smoother, 1, tria.n_levels()-1);
    mg.set_edge_matrices(mg_interface_down, mg_interface_up);  

    auto Zero_1_2 = Bt*0.0;
    auto Zero_2_1 = B*0.0;
    auto my_AA   = block_operator<3, 3, ImmersX::LA::MPI::BlockVector>(
      {{ {{A, Zero_1_2, Bt}}, {{Zero_2_1, A_2, M_2}} ,{{B, M_2, Zero}} }}); 

    typedef decltype(my_AA) my_block_operator;

    MGLevelObject<std::shared_ptr<my_block_operator>> mg_matrices_operators;
    mg_matrices_operators.resize(1, tria.n_levels()-1);

    std::vector<decltype(M_2)> mg_A;
    std::vector<decltype(M_2)> mg_A_2;
    std::vector<decltype(M_2)> mg_Bt;
    std::vector<decltype(M_2)> mg_B;
    std::vector<decltype(M_2)> mg_zero_1_2;
    std::vector<decltype(M_2)> mg_zero_2_1;
    std::vector<decltype(M_2)> mg_zero_3_3;

    mg_A.resize(tria.n_levels());
    mg_A_2.resize(tria.n_levels());
    mg_B.resize(tria.n_levels());
    mg_Bt.resize(tria.n_levels());

    mg_zero_1_2.resize(tria.n_levels());
    mg_zero_2_1.resize(tria.n_levels());
    mg_zero_3_3.resize(tria.n_levels());          

    for (unsigned int level = 1; level < tria.n_levels(); level++) {

      mg_A[level-1] = (linear_operator<VectorType, VectorType, Payload>(mg_matrices[level].block(0,0)));
      mg_A_2[level-1] = linear_operator<VectorType, VectorType, Payload>(mg_matrices[level].block(1,1));
      mg_Bt[level-1] = (linear_operator<VectorType, VectorType, Payload>(mg_matrices[level].block(0,2)));
      mg_B[level-1] = (transpose_operator<VectorType, VectorType, Payload>(mg_matrices[level].block(2,0)));

      mg_zero_1_2[level-1] = ( 0.0 * mg_Bt[level-1] );
      mg_zero_2_1[level-1] = ( 0.0 * mg_B[level-1] );
      mg_zero_3_3[level-1] = ( 0.0 * M_2);

      mg_matrices_operators[level] = std::make_shared<my_block_operator>(block_operator<3, 3, ImmersX::LA::MPI::BlockVector>(
        {{
          {{mg_A[level-1], mg_zero_1_2[level-1], mg_Bt[level-1]}}, 
          {{mg_zero_2_1[level-1], mg_A_2[level-1], M_2}} ,
          {{mg_B[level-1], M_2, mg_zero_3_3[level-1]}}
        }})
      );


    }

    ImmersX::LA::MPI::BlockVector my_solution_block;
    ImmersX::LA::MPI::BlockVector my_system_rhs_block;
    my_AA.reinit_domain_vector(my_solution_block, false);
    my_AA.reinit_range_vector(my_system_rhs_block, false);

    my_system_rhs_block.block(0) = rhs.block(0);
    my_system_rhs_block.block(2) = rhs.block(1);

    const unsigned int pre_smooth_steps = 5;
    [[maybe_unused]] const unsigned int post_smooth_steps = 5;

    mg_solutions[tria.n_levels()-1].reinit(my_solution_block, false);
    mg_rhs[tria.n_levels()-1].reinit(my_system_rhs_block, false);

    ImmersX::LA::MPI::BlockVector smoother_residual;

    cout << "CHECK operator\n";


    //Pre smooth
    for (unsigned int level = tria.n_levels()-1; level > 1; level--) {     

      mg_matrices_operators[level] -> reinit_range_vector(smoother_residual, false);

      for(unsigned int i = 0; i < pre_smooth_steps; i++) {
        cout<<"Calcolo residuo\n";

        ImmersX::LA::MPI::BlockVector tmp1, tmp2;

        mg_matrices_operators[level] -> reinit_domain_vector(tmp1, false);
        mg_matrices_operators[level] -> reinit_range_vector(tmp2, false);
        
        mg_matrices_operators[level] -> vmult(tmp2, tmp1);
        //residual *= -1;
        //residual += mg_rhs[level];

        //std::cout<<"Il residuo ha norma: " << residual.l2_norm() << "\n";

        cout<<"Smooth\n";
        //smoothers[level].vmult(residual, residual);
        cout<<"Aggiornamento\n";
        //mg_solutions[level] += 0.25*residual;

        //mg_smoother.smooth(level, mg_solutions[level], mg_rhs[level]);
      }

      mg_rhs[level-1] = 0;
      mg_solutions[level-1] = 0;

      cout<<"Restrizione\n";
      //coupled_transfer.restrict_and_add(level, mg_rhs[level-1], mg_solutions[level]);
    }

    //Post smooth
    for (unsigned int level = 1; level < tria.n_levels()-1; level++) {
      
    }


    /*
    SolverControl            my_solver_control(1000, 1e-9);
    SolverGMRES<ImmersX::LA::MPI::BlockVector> solver_gmres(my_solver_control);

    CoupledMgPreconditioner<ImmersX::LA::MPI::BlockVector> mg_coupled_preconditioner(mg);

    solver_fgmres.solve(my_AA,
                        my_solution_block,
                        my_system_rhs_block,
                        mg_coupled_preconditioner);
    */

    cout<< "Ended coupled solver \n";

    SolverControl control(1000, 1e-12);
    dealii::SolverGMRES<GlobalVector> solver(control);
    //solver.solve(system_matrix, state, rhs, preconditioner);

    cout << "GMRES converged in "
            << control.last_step() << " iteration(s), residual "
            << control.last_value() << "." << std::endl;
    
    bulk_problem.set_solution(adapter.field(state, bulk.fields().solution));
    embedded_problem.set_solution(
      adapter.field(state, embedded.fields().solution));

    bulk_problem.output_results();
    embedded_problem.output_results();

    std::filesystem::create_directories(
      application_parameters.output_directory);
    dealii::DataOut<1, 2> multiplier_output;
    multiplier_output.attach_dof_handler(multiplier_dh);
    multiplier_output.add_data_vector(
      adapter.field(state, coupling.fields().multiplier),
      "lagrange_multiplier",
      dealii::DataOut<1, 2>::type_dof_data);
    multiplier_output.build_patches();
    const auto rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto multiplier_filename =
      std::filesystem::path(application_parameters.output_directory) /
      (application_parameters.multiplier_output_name + "-0." +
       std::to_string(rank) + ".vtu");
    std::ofstream multiplier_file(multiplier_filename);
    multiplier_output.write_vtu(multiplier_file);
    if (rank == 0)
      {
        std::ofstream multiplier_pvd(
          std::filesystem::path(application_parameters.output_directory) /
          (application_parameters.multiplier_output_name + ".pvd"));
        dealii::DataOutBase::write_pvd_record(
          multiplier_pvd,
          {{0., application_parameters.multiplier_output_name + "-0.0.vtu"}});
      }

    auto residual = adapter.make_state();
    adapter.evaluate_residual(state, residual);
    const double residual_norm = residual.l2_norm();
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "coupled_residual = " << residual_norm << '\n';

    AssertThrow(std::isfinite(residual_norm) && residual_norm < 1.e-7,
                dealii::ExcMessage(
                  "The coupled Poisson residual is too large."));
  }
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);
      const std::string parameter_file = argc > 1 ? argv[1] : "parameters.prm";
      const auto dimensions = ImmersX::get_dimension_parameters(parameter_file);

      if (dimensions.dimension != 2 || dimensions.space_dimension != 2 ||
          dimensions.reduced_dimension != 1)
        ImmersX::throw_unsupported_dimension_combination(dimensions);

      run_coupled_poisson(parameter_file);
    }
  catch (const std::exception &exception)
    {
      std::cerr << exception.what() << std::endl;
      return 1;
    }

  return 0;
}
