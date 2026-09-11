// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <gtest/gtest.h>
#include <immersx/core/contributor.h>
#include <immersx/core/fe_space.h>
#include <immersx/core/state.h>
#include <immersx/core/weak_term.h>

using namespace dealii;
using namespace ImmersX;

namespace
{
  struct ScalarSpace
  {
    ScalarSpace()
    {
      GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(1);
      dof_handler.distribute_dofs(finite_element);
      constraints.close();
    }

    Triangulation<2>          triangulation;
    FE_Q<2>                   finite_element{1};
    DoFHandler<2>             dof_handler{triangulation};
    AffineConstraints<double> constraints;
  };

  using FieldVector = dealii::Vector<double>;
  using MatrixType  = dealii::SparseMatrix<double>;
  using ModelType   = ImmersX::SemiDiscreteModel<FieldVector, MatrixType>;
} // namespace

TEST(FoundationalApplications, MixedBiharmonicUsesCrossEquationWeakTerms)
{
  ScalarSpace space;
  StateLayout layout;
  const auto  V =
    fe_space(space.dof_handler, StaticMappingQ1<2>::mapping, space.constraints);
  const auto u = V.field(layout, "u");
  const auto w = V.field(layout, "w");

  ModelType                                             model;
  ImmersX::SemidiscreteBuilder<FieldVector, MatrixType> builder(layout, model);

  // For w = Delta u, the mixed equations use ordinary terms such as
  // grad(u) : grad(test(w)) and w * test(w). The second equation uses
  // grad(w) : grad(test(u)). There is no multiplier relation here.
  weak_term(gradient(u), gradient(test(w))).add(builder);
  weak_term(value(w), test(w)).add(builder);
  weak_term(gradient(w), gradient(test(u))).add(builder);

  ImmersX::StateView<FieldVector>               state_view(layout, 0.);
  const ImmersX::EvaluationContext<FieldVector> context(0., state_view);

  const auto u_from_w =
    model.state_matrix_operator(u.field_id(), w.field_id(), context);
  const auto w_from_u =
    model.state_matrix_operator(w.field_id(), u.field_id(), context);
  const auto w_from_w =
    model.state_matrix_operator(w.field_id(), w.field_id(), context);

  ASSERT_TRUE(u_from_w.has_value());
  ASSERT_TRUE(w_from_u.has_value());
  ASSERT_TRUE(w_from_w.has_value());
  EXPECT_FALSE(model.has_derivative_terms());
  EXPECT_TRUE(model.saddle_points().empty());
  EXPECT_GT(u_from_w->matrix()->n_nonzero_elements(), 0u);
  EXPECT_GT(w_from_u->matrix()->n_nonzero_elements(), 0u);
  EXPECT_GT(w_from_w->matrix()->n_nonzero_elements(), 0u);
}

TEST(FoundationalApplications, IncompressibleElasticityUsesSaddlePointWeakTerms)
{
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  FESystem<2>               fe(FE_Q<2>(1), 3);
  DoFHandler<2>             dof_handler(tria);
  AffineConstraints<double> constraints;
  dof_handler.distribute_dofs(fe);
  constraints.close();

  StateLayout layout;
  const auto  V =
    fe_space(dof_handler, StaticMappingQ1<2>::mapping, constraints);
  const auto displacement =
    V.field(layout, "displacement", FEValuesExtractors::Vector(0));
  const auto pressure =
    V.field(layout, "pressure", FEValuesExtractors::Scalar(2));

  ModelType                                             model;
  ImmersX::SemidiscreteBuilder<FieldVector, MatrixType> builder(layout, model);

  // A(u) + B^T p = f and B u = 0 are ordinary weak terms. The explicit
  // saddle-point metadata describes the pressure role; it does not create a
  // Constraint or a second hierarchy.
  weak_term(symmetric_gradient(displacement),
            symmetric_gradient(test(displacement)))
    .add(builder);
  weak_term(value(pressure), divergence(test(displacement))).add(builder);
  weak_term(divergence(displacement), test(pressure)).add(builder);
  builder.saddle_point(pressure.field_id(), {displacement.field_id()});

  ImmersX::StateView<FieldVector>               state_view(layout, 0.);
  const ImmersX::EvaluationContext<FieldVector> context(0., state_view);
  const auto                                    displacement_from_displacement =
    model.state_matrix_operator(displacement.field_id(),
                                displacement.field_id(),
                                context);
  const auto displacement_from_pressure =
    model.state_matrix_operator(displacement.field_id(),
                                pressure.field_id(),
                                context);
  const auto pressure_from_displacement =
    model.state_matrix_operator(pressure.field_id(),
                                displacement.field_id(),
                                context);

  ASSERT_TRUE(displacement_from_displacement.has_value());
  ASSERT_TRUE(displacement_from_pressure.has_value());
  ASSERT_TRUE(pressure_from_displacement.has_value());
  ASSERT_EQ(model.saddle_points().size(), 1u);
  EXPECT_EQ(model.saddle_points().front().multiplier, pressure.field_id());
  EXPECT_GT(displacement_from_displacement->matrix()->n_nonzero_elements(), 0u);
  EXPECT_GT(displacement_from_pressure->matrix()->n_nonzero_elements(), 0u);
  EXPECT_GT(pressure_from_displacement->matrix()->n_nonzero_elements(), 0u);
}
