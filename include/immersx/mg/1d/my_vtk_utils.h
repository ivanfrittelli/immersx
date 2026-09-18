// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the reduced_lagrange_multipliers application, based on
// the deal.II library.
//
// The reduced_lagrange_multipliers application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the reduced_lagrange_multipliers distribution.
//
// ---------------------------------------------------------------------
#pragma once

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <map>
#include <unordered_map>

#include <immersx/mg/1d/graph.h>


#  include <deal.II/grid/tria.h>

#  include <deal.II/lac/la_parallel_vector.h>
#  include <deal.II/lac/vector.h>

using namespace dealii;

namespace myVTKUtils
/**
 * @namespace VTKUtils
 * @brief Utility functions for interfacing between VTK mesh/data files and deal.II data structures.
 *
 * This namespace provides a collection of functions to read VTK mesh files and
 * associated data fields, and to map them into deal.II Triangulation,
 * DoFHandler, and vector objects. The utilities support reading mesh geometry,
 * cell and vertex data, mapping VTK fields to deal.II finite elements, and
 * transferring data between serial and distributed representations.
 *
 * Main functionalities include:
 * - Reading VTK mesh files and populating deal.II Triangulation objects.
 * - Reading cell and vertex data arrays from VTK files into deal.II vectors.
 * - Mapping VTK data fields to suitable deal.II FiniteElement objects.
 * - Translating VTK data arrays into deal.II vectors associated with DoFHandler
 * objects.
 * - Transferring data between serial and distributed deal.II vectors.
 * - Mapping distributed vertex indices to serial indices for parallel
 * computations.
 *
 * These utilities are intended to facilitate the import of VTK-based mesh and
 * data into deal.II-based finite element workflows, supporting both serial and
 * parallel computations.
 */
{
  void read_vtk_graph(const std::string            &vtk_filename,
           Graph & my_graph, bool only_one);
  

  /**
   * @brief Read cell data (scalar or vector) from a VTK file and store it in
   * the output vector.
   *
   * This function reads the specified cell data array (scalar or vector) from
   * the given VTK file and stores it in the provided output vector. For vector
   * data, all components are stored in row-major order (cell0_comp0,
   * cell0_comp1, ..., cell1_comp0, ...).
   *
   * @param vtk_filename The name of the input VTK file.
   * @param cell_data_name The name of the cell data array to read.
   * @param output_vector The vector to store the cell data values.
   */
  void
  my_read_cell_data(const std::string &vtk_filename,
                 const std::string &cell_data_name,
                 Vector<double>    &output_vector);

  void 
  read_cell_label_graph(const std::string &vtk_filename,
                        Graph & my_graph,
                        const std::string &cell_data_name = "Label");
  
} // namespace VTKUtils
