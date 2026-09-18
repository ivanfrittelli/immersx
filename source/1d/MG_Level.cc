#include <immersx/mg/1d/MG_Level.h>

#include <Kokkos_Pair.hpp>
#include <cstdlib>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <iostream>

template <typename MatrixType>
void
MG_Level<MatrixType>::make_grid() {
  triangulation = graph.create_graph_triangulation(); 
  dof_handler.distribute_dofs(fe);

  owned_dofs.resize(2);
  owned_dofs[0] = dof_handler.locally_owned_dofs();
  relevant_dofs.resize(2);
  relevant_dofs[0] = DoFTools::extract_locally_relevant_dofs(dof_handler);
}

/*
void
MG_Level<MatrixType>::setup_boundary_conditions(const std::set<int> & dirichlet_cells, const std::set<int> & neumann_cells) {
  for (int i = 0; i < 2; i++)
    for (const auto &cell : triangulation.active_cell_iterators()) {
        if (cell-> face(i) -> at_boundary()) {
          if (dirichlet_cells.count(cell->active_cell_index()) > 0) {
            cell -> face(i) -> set_boundary_id(1);
          }
          else if(neumann_cells.count(cell->active_cell_index()) > 0) {
            cell -> face(i) -> set_boundary_id(2);
          }
          else {
            cell -> face(i) -> set_boundary_id(0);
          }
        }
      }
}
*/

template <typename MatrixType>
void
MG_Level<MatrixType>::setup_system(const Function<3> & boundary_conditions) {
  std::cout << "   Number of degrees of freedom in level " << id << " : " << dof_handler.n_dofs()
            << std::endl;

  constraints.clear();

  //VectorTools::interpolate_boundary_values(dof_handler,
  //                                      1,
  //                                      boundary_conditions,
  //                                      constraints);

  //Nodi staccati
  //VectorTools::interpolate_boundary_values(dof_handler,
  //                                      0,
  //                                      Functions::ZeroFunction<3>(),
  //                                      constraints);
      
  //std::cout << "   Number of not dirichlet constrained degrees of freedom in level " << id << " : " << dof_handler.n_dofs() - dof_handler.n_boundary_dofs({0})
  //          << std::endl;

  
  //std::cout<<"Dof vincolati:"<<"\n";
  //std::cout<<dof_handler.n_boundary_dofs({0})<<"\n";
  //std::cout<<dof_handler.n_boundary_dofs({1})<<"\n";
  //std::cout<<dof_handler.n_boundary_dofs({2})<<"\n";

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  constraints.close();

  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  system_matrix.clear();
  DynamicSparsityPattern dsp(relevant_dofs[0]);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                              owned_dofs[0],
                                              mpi_communicator,
                                              relevant_dofs[0]);
  system_matrix.reinit(owned_dofs[0],
                          owned_dofs[0],
                          dsp,
                          mpi_communicator);

  mass_matrix.reinit(owned_dofs[0],
                          owned_dofs[0],
                          dsp,
                          mpi_communicator);
}

template <typename MatrixType>
void
MG_Level<MatrixType>::assemble_system(const FunctionParser<3> & reaction_term, const TimeInfo & time_info, const Vector<double> & radii) {
  
  system_matrix = 0;
  mass_matrix = 0;
  
  QGauss<1>     quadrature_formula(fe.degree + 1);

  FEValues<1,3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    
  FullMatrix<double> mass_cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices()) {

        double reaction = reaction_term.value(fe_values.quadrature_point(q_index));

        for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices()) {

            mass_cell_matrix(i,j) += fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index);

            if (time_info.is_time_dependent) {
              cell_matrix(i, j) += 
                radii[graph.small_to_big_cell_map[cell -> active_cell_index()]]*
                time_info.dt * 
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index));

              cell_matrix(i,j) += reaction * time_info.dt  *
                (fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index));

              cell_matrix(i,j) += fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index);
              
            } else {
              cell_matrix(i, j) +=
                std::pow(radii[graph.small_to_big_cell_map[cell -> active_cell_index()]], 4) * 3.1415 *
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index));           // dx

              cell_matrix(i,j) += reaction *
                (fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index));
            }
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
          cell_matrix, local_dof_indices, system_matrix);

      constraints.distribute_local_to_global(
          mass_cell_matrix, local_dof_indices, mass_matrix);
    }
  
  system_matrix.compress(VectorOperation::add);
  mass_matrix.compress(VectorOperation::add);
}

template <typename MatrixType>
void 
MG_Level<MatrixType>::assemble_system_and_rhs(const FunctionParser<3> & reaction_term,
  const FunctionParser<3> & rhs_function,
  Vector<double> & rhs, 
  const FunctionParser<3> & neumann_function,
  const Vector<double> & old_solution,
  const TimeInfo & time_info,
  const Vector<double> & radii) 
{
  rhs = 0;
  system_matrix = 0;

  QGauss<1>     quadrature_formula(fe.degree + 1);
  QGauss<0> face_quadrature_formula(fe.degree + 1);

  FEValues<1,3> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<1,3> fe_face_values(fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);


  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  Vector<double>     cell_rhs(dofs_per_cell);
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> old_solution_values(fe_values.n_quadrature_points);
  std::vector<Tensor<1,3>> old_solution_gradients(fe_values.n_quadrature_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    if(time_info.is_time_dependent) {
      fe_values.get_function_values(old_solution, old_solution_values);
      fe_values.get_function_gradients(old_solution, old_solution_gradients);
    }

    cell_matrix = 0;
    cell_rhs    = 0;

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      const auto &x_q = fe_values.quadrature_point(q_index);
      const double reaction = reaction_term.value(x_q);
      
      for (const unsigned int i : fe_values.dof_indices())
      {
          if(time_info.is_time_dependent) {      
            for (const unsigned int j : fe_values.dof_indices()) {
              cell_matrix(i, j) += 
                radii[graph.small_to_big_cell_map[cell -> active_cell_index()]]*
                time_info.dt * 
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index));

              cell_matrix(i,j) += reaction * time_info.dt  *
                (fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index));

              cell_matrix(i,j) += fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index);
            }

            cell_rhs(i) += time_info.dt * (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                          rhs_function.value(x_q) *       // f(x_q)
                          fe_values.JxW(q_index));            // dx

            /*
            for (const auto &f : cell->face_indices())
              if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 2)
                {
                  fe_face_values.reinit(cell, f);

                  for (const unsigned int q_index :
                      fe_face_values.quadrature_point_indices()) {
                        const auto neumann_function_value = neumann_function.value(fe_face_values.quadrature_point(q_index));

                        for (const unsigned int i : fe_face_values.dof_indices())
                          cell_rhs(i) += time_info.dt *
                            fe_face_values.shape_value(i, q_index) * // phi_i(x_q)
                            neumann_function_value * // g(x_q)
                            fe_face_values.JxW(q_index);                 // ds
                      }
                    
                }
              */

            cell_rhs(i) += fe_values.shape_value(i, q_index)
              * old_solution_values[q_index]
              * fe_values.JxW(q_index);
          }
          else{
            cell_rhs(i) += //(i == inlet_dof ? 1:0)*
              (fe_values.shape_value(i, q_index) * // phi_i(x_q)
              rhs_function.value(x_q) *       // f(x_q)
              fe_values.JxW(q_index));            // dx

            for (const unsigned int j : fe_values.dof_indices()) {
              cell_matrix(i, j) +=
                std::pow(radii[graph.small_to_big_cell_map[cell -> active_cell_index()]], 4) * 3.1415 *
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index));           // dx

              cell_matrix(i,j) += reaction *
                (fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                fe_values.JxW(q_index));
              }

            /*
            for (const auto &f : cell->face_indices())
              if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 2)
                {
                  fe_face_values.reinit(cell, f);

                  for (const unsigned int q_index :
                      fe_face_values.quadrature_point_indices()) {
                        const auto neumann_function_value = neumann_function.value(fe_face_values.quadrature_point(q_index));

                        for (const unsigned int i : fe_face_values.dof_indices())
                          cell_rhs(i) += fe_face_values.shape_value(i, q_index) * // phi_i(x_q)
                            neumann_function_value * // g(x_q)
                            fe_face_values.JxW(q_index);                 // ds
                      }
                    
                }
            */
          }
      }
    }
        
    cell->get_dof_indices(local_dof_indices);

    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, rhs);      
  }
}

template <typename MatrixType>
std::map<int, int>
MG_Level<MatrixType>::get_dof_to_vertex_map() {
  int current_cell = 0;

  std::map<int, int> dof_to_vertex_map;
  for(const auto & cell : dof_handler.active_cell_iterators()) {

    for (int i = 0; i < 2; i++) {

      int dof = cell -> vertex_dof_index(i, 0);
      int vertex = graph.cells[current_cell].vertices[i];

      if (dof_to_vertex_map.count(dof) == 0)
        dof_to_vertex_map[dof] = vertex;
    }

    current_cell++;
  }

  return dof_to_vertex_map;
}

template <typename MatrixType>
std::map<int, int>
MG_Level<MatrixType>::get_vertex_to_dof_map() {
  int current_cell = 0;

  std::map<int, int> vertex_to_dof_map;
  for(const auto & cell : dof_handler.active_cell_iterators()) {
    for (int i = 0; i < 2; i++) {
      int dof = cell -> vertex_dof_index(i, 0);
      int vertex = graph.cells[current_cell].vertices[i];

      if (vertex_to_dof_map.count(vertex) == 0)
        vertex_to_dof_map[vertex] = dof;
    }
    current_cell++;
  }
  return vertex_to_dof_map;

}

template class MG_Level<TrilinosWrappers::SparseMatrix>;