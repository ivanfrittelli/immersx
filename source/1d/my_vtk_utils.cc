#include <immersx/mg/1d/my_vtk_utils.h>

#include <deal.II/base/exceptions.h>
#include <string>
#include <vector>

#  include <deal.II/distributed/fully_distributed_tria.h>

#  include <deal.II/dofs/dof_renumbering.h>

#  include <deal.II/fe/fe.h>
#  include <deal.II/fe/fe_dgq.h>
#  include <deal.II/fe/fe_nothing.h>
#  include <deal.II/fe/fe_q.h>
#  include <deal.II/fe/fe_system.h>

#  include <deal.II/grid/grid_in.h>
#  include <deal.II/grid/grid_out.h>
#  include <deal.II/grid/grid_tools.h>
#  include <deal.II/grid/tria.h>
#  include <deal.II/grid/tria_accessor.h>
#  include <deal.II/grid/tria_description.h>
#  include <deal.II/grid/tria_iterator.h>

#  include <vtkCellData.h>
#  include <vtkCleanUnstructuredGrid.h>
#  include <vtkDataArray.h>
#  include <vtkPointData.h>
#  include <vtkSmartPointer.h>
#  include <vtkUnstructuredGrid.h>
#  include <vtkUnstructuredGridReader.h>


#  include <stdexcept>

namespace myVTKUtils
{
  void
  read_vtk_graph(const std::string            &vtk_filename,
           Graph & my_graph, bool only_one)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));

    // Read points
    vtkPoints                   *vtk_points = grid->GetPoints();
    const vtkIdType              n_points   = only_one ? grid->GetCell(0)->GetNumberOfPoints() : vtk_points->GetNumberOfPoints();

    for (vtkIdType i = 0; i < n_points; ++i)
      {
        std::array<double, 3> coords = {{0, 0, 0}};
        vtk_points->GetPoint(i, coords.data());
        my_graph.add_point(coords[0], coords[1], coords[2]);
      }

    // Read cells
    const vtkIdType            n_cells = only_one ? 1 : grid->GetNumberOfCells();
                              std::ofstream out("AA.txt");

    for (vtkIdType i = 0; i < n_cells; ++i)
      {
        vtkCell *cell = grid->GetCell(i);
        if constexpr (1 == 1)
          {
            /*
            if (cell->GetCellType() != VTK_LINE)
              AssertThrow(false,
                          ExcMessage(
                            "Unsupported cell type in 1D VTK file: only "
                            "VTK_LINE is supported."));*/
                            /*
            AssertThrow(cell->GetNumberOfPoints() == 2,
                        ExcMessage(
                          "Only line cells with 2 points are supported."));
                          */
            out << "Cella: " << i << "\n";
            int starting_cell = -1, ending_cell = -1;
            for (unsigned int j = 0; j < cell->GetNumberOfPoints()-1; ++j) {
              my_graph.add_cell(cell->GetPointId(j), cell->GetPointId(j+1));

              my_graph.small_to_big_cell_map[my_graph.cells.size()-1] = my_graph.big_cells.size();

              if(j == 0) starting_cell = my_graph.cells.size() -1;
              if(j == cell->GetNumberOfPoints()-2) ending_cell = my_graph.cells.size() -1;

              out << cell->GetPointId(j) << " " << cell->GetPointId(j+1) << "\n";
            }
            AssertThrow(starting_cell >= 0 && ending_cell >= 0, ExcMessage("Cell with 1 vertex"));
            
            //if (cell->GetPointId(0) == cell->GetPointId(cell -> GetNumberOfPoints() -1)) std::cout<<cell->GetPointId(0) << " " << cell->GetNumberOfPoints() <<"\n";

            my_graph.add_big_cell(
              cell->GetPointId(0), cell->GetPointId(cell -> GetNumberOfPoints() -1),
              starting_cell, ending_cell,
              cell->GetNumberOfPoints() - 1);

            out << "\n\n";
          }
        else
          {
            AssertThrow(false, ExcMessage("Unsupported dimension."));
          }
      }

  }

  void 
  read_cell_label_graph(const std::string &vtk_filename,
                        Graph & my_graph,
                        const std::string &cell_data_name)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));
    vtkDataArray *data_array =
      grid->GetCellData()->GetArray(cell_data_name.c_str());
    AssertThrow(data_array,
                ExcMessage("Cell data array '" + cell_data_name +
                          "' not found in VTK file: " + vtk_filename));
    vtkIdType n_tuples     = data_array->GetNumberOfTuples();

    for (vtkIdType i = 0; i < n_tuples; ++i) {
        if(data_array->GetComponent(i, 0) == 1) {
          my_graph.dirichlet_big_cells.push_back(i);
        }
        else if (data_array->GetComponent(i, 0) == 2) {
          my_graph.neumann_big_cells.push_back(i);
        }
    }
  }

  void
  my_read_cell_data(const std::string &vtk_filename,
                 const std::string &cell_data_name,
                 Vector<double>    &output_vector)
  {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_filename.c_str());
    reader->Update();
    vtkUnstructuredGrid *grid = reader->GetOutput();
    AssertThrow(grid, ExcMessage("Failed to read VTK file: " + vtk_filename));
    vtkDataArray *data_array =
      grid->GetCellData()->GetArray(cell_data_name.c_str());
    AssertThrow(data_array,
                ExcMessage("Cell data array '" + cell_data_name +
                           "' not found in VTK file: " + vtk_filename));
    vtkIdType n_tuples     = data_array->GetNumberOfTuples();
    int       n_components = data_array->GetNumberOfComponents();
    output_vector.reinit(n_tuples * n_components);
    for (vtkIdType i = 0; i < n_tuples; ++i)
      for (int j = 0; j < n_components; ++j)
        output_vector[i * n_components + j] = data_array->GetComponent(i, j);
  }
} // namespace VTKUtils