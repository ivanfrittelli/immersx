// ---------------------------------------------------------------------
//
// Copyright (C) 2024 by Luca Heltai
//
// This file is part of the ImmersX application, based on
// the deal.II library.
//
// The ImmersX application is free software; you can use
// it, redistribute it, and/or modify it under the terms of the Apache-2.0
// License WITH LLVM-exception as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later version. The
// full text of the license can be found in the file LICENSE.md at the top level
// of the ImmersX distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <gtest/gtest.h>
#include <immersx/coupling/inclusions.h>

#include <array>

using namespace ImmersX;
#include <immersx/coupling/reference_cross_section.h> // Add include for ReferenceCrossSection
#include <immersx/coupling/tensor_product_lift.h>

using namespace dealii;

TEST(ReferenceCrossSection, CheckBasisOrthogonality) // NOLINT
{
  const int          dim          = 2;
  const int          spacedim     = 2;
  const int          n_components = 1;
  const unsigned int degree       = 3;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Or "hyper_cube"
  par.refinement_level = 2;            // Use a reasonable refinement level
  // Leaving par.selected_coefficients empty means all basis functions are
  // selected

  // Create the reference inclusion object
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  // Get the computed basis functions and the mass matrix
  const auto &basis       = ref_inclusion.get_basis_functions();
  const auto &mass_matrix = ref_inclusion.get_mass_matrix();

  const unsigned int n_basis_functions = basis.size();

  // Verify the number of basis functions matches the polynomial space size
  PolynomialsP<spacedim> polynomials(degree);
  ASSERT_EQ(n_basis_functions, polynomials.n() * n_components);

  // Check for M-orthonormality: basis[i]^T * M * basis[j] == delta_ij
  Vector<double> tmp(mass_matrix.m()); // Temporary vector for M * basis[j]

  for (unsigned int i = 0; i < n_basis_functions; ++i)
    {
      ASSERT_GT(basis[i].l2_norm(), 1e-15)
        << "Basis function " << i << " is zero.";
      for (unsigned int j = 0; j < n_basis_functions; ++j)
        {
          const double dot_product =
            mass_matrix.matrix_scalar_product(basis[i], basis[j]);

          if (i == j)
            {
              // Diagonal elements should be close to |D| for orthogonal basis
              ASSERT_NEAR(dot_product, ref_inclusion.measure(), 1e-10)
                << "Basis functions " << i << " and " << j
                << " are not M-orthonormal (diagonal check).";
            }
          else
            {
              // Off-diagonal elements should be close to 0 for orthogonal basis
              ASSERT_NEAR(dot_product, 0.0, 1e-10)
                << "Basis functions " << i << " and " << j
                << " are not M-orthogonal (off-diagonal check).";
            }
        }
    }
}


TEST(ReferenceCrossSection, CheckDiskQuadrature) // NOLINT
{
  const int          dim          = 1;
  const int          spacedim     = 2;
  const int          n_components = 1;
  const unsigned int degree       = 3;

  // Set up parameters for the reference inclusion
  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_ball"; // Use a circular domain
  par.refinement_level = 5;            // Use a reasonable refinement level

  // Create the reference inclusion object
  ReferenceCrossSection<dim, spacedim, n_components> ref_inclusion(par);

  // Get the global quadrature
  const auto &quadrature = ref_inclusion.get_global_quadrature();

  // Compute the sum of all quadrature weights, which should equal
  // the measure of the domain (2*pi for a unit circle, with r=1)
  double sum = 0.0;
  for (const auto &weight : quadrature.get_weights())
    {
      sum += weight;
    }

  // The area of a unit circle is 2*pi
  ASSERT_NEAR(sum, 2 * numbers::PI, 1e-3)
    << "Integral of unit function over unit circle should equal 2*pi";

  // Test that a constant function integrates as expected
  double const_integral = 0.0;
  double const_value    = 2.5; // Some arbitrary constant

  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const_integral += const_value * quadrature.weight(q);
    }

  ASSERT_NEAR(const_integral, const_value * 2 * numbers::PI, 2e-3)
    << "Constant function integral incorrect";

  // Test that x^2 + y^2 integrates to expected value over unit circle
  // For x^2 + y^2 over unit circle, the exact result is pi/2
  double r_squared_integral = 0.0;

  for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const Point<spacedim> &p = quadrature.point(q);
      r_squared_integral += p.square() * quadrature.weight(q);
    }

  ASSERT_NEAR(r_squared_integral, 2 * numbers::PI, 1e-3)
    << "Integral of r^2 over unit circle should equal 2 pi";
}


TEST(ReferenceFrame, RotationInvariants3D) // NOLINT
{
  const std::array<Tensor<1, 3>, 4> tangents = {{Tensor<1, 3>({0., 0., 1.}),
                                                 Tensor<1, 3>({1., 0., 0.}),
                                                 Tensor<1, 3>({0., 0., -1.}),
                                                 Tensor<1, 3>({1., 2., 2.})}};

  for (const auto &tangent : tangents)
    {
      const auto rotation     = detail::reference_to_physical_rotation(tangent);
      const auto unit_tangent = tangent / tangent.norm();

      Tensor<1, 3> reference_vertical;
      reference_vertical[2] = 1.;
      const auto mapped     = rotation * reference_vertical;
      for (unsigned int d = 0; d < 3; ++d)
        EXPECT_NEAR(mapped[d], unit_tangent[d], 1.e-12);

      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          {
            double entry = 0.;
            for (unsigned int k = 0; k < 3; ++k)
              entry += rotation[k][i] * rotation[k][j];
            EXPECT_NEAR(entry, i == j ? 1. : 0., 1.e-12);
          }

      const double determinant =
        rotation[0][0] *
          (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1]) -
        rotation[0][1] *
          (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0]) +
        rotation[0][2] *
          (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
      EXPECT_NEAR(determinant, 1., 1.e-12);
    }
}


TEST(ReferenceFrame, RotationInvariants2D) // NOLINT
{
  const std::array<Tensor<1, 2>, 3> tangents = {{Tensor<1, 2>({0., 1.}),
                                                 Tensor<1, 2>({1., 0.}),
                                                 Tensor<1, 2>({0., -1.})}};

  for (const auto &tangent : tangents)
    {
      const auto rotation     = detail::reference_to_physical_rotation(tangent);
      const auto unit_tangent = tangent / tangent.norm();

      Tensor<1, 2> reference_vertical;
      reference_vertical[1] = 1.;
      const auto mapped     = rotation * reference_vertical;
      for (unsigned int d = 0; d < 2; ++d)
        EXPECT_NEAR(mapped[d], unit_tangent[d], 1.e-12);

      for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
          {
            double entry = 0.;
            for (unsigned int k = 0; k < 2; ++k)
              entry += rotation[k][i] * rotation[k][j];
            EXPECT_NEAR(entry, i == j ? 1. : 0., 1.e-12);
          }

      const double determinant =
        rotation[0][0] * rotation[1][1] - rotation[0][1] * rotation[1][0];
      EXPECT_NEAR(determinant, 1., 1.e-12);
    }
}


TEST(ReferenceCrossSection, RotationAwareVectorModes) // NOLINT
{
  ParameterAcceptor::clear();
  ReferenceCrossSectionParameters<1, 3, 3> parameters(
    "/Rotation-aware cross section/");
  parameters.inclusion_degree      = 0;
  parameters.refinement_level      = 1;
  parameters.selected_coefficients = {2};
  ReferenceCrossSection<1, 3, 3> section(parameters);

  const Point<3>                  origin(1., 2., 3.);
  const double                    scale    = 0.37;
  const std::vector<Tensor<1, 3>> tangents = {Tensor<1, 3>({0., 0., 1.}),
                                              Tensor<1, 3>({1., 0., 0.}),
                                              Tensor<1, 3>({0., 0., -1.}),
                                              Tensor<1, 3>({1., 2., 2.})};

  for (const auto &tangent : tangents)
    {
      const auto rotation = detail::reference_to_physical_rotation(tangent);
      const auto transformed =
        section.get_transformed_quadrature(origin, tangent, scale);
      for (unsigned int q = 0; q < section.n_quadrature_points(); ++q)
        {
          const auto expected_point =
            origin +
            rotation * (scale * section.get_global_quadrature().point(q));
          for (unsigned int d = 0; d < 3; ++d)
            EXPECT_NEAR(transformed.point(q)[d], expected_point[d], 1.e-12);

          Tensor<1, 3> reference_value;
          for (unsigned int component = 0; component < 3; ++component)
            reference_value[component] = section.shape_value(0, q, component);
          const auto expected_mode = rotation * reference_value;
          const auto mode_values =
            section.get_transformed_mode_values(q, tangent);
          ASSERT_EQ(mode_values.size(), 3u);
          for (unsigned int component = 0; component < 3; ++component)
            EXPECT_NEAR(mode_values[component],
                        expected_mode[component],
                        1.e-12);
        }
    }
}


TEST(TensorProductLift, RotationAwareVectorModes) // NOLINT
{
  ParameterAcceptor::clear();
  TensorProductLift<1, 2, 3, 3> lift("/Rotation-aware lift/");
  lift.section.inclusion_degree      = 0;
  lift.section.refinement_level      = 1;
  lift.section.selected_coefficients = {2};
  TensorProductLiftSupport<1, 2, 3, 3> support(lift.parameters());

  const std::array<Tensor<1, 3>, 4> tangents = {{Tensor<1, 3>({0., 0., 1.}),
                                                 Tensor<1, 3>({1., 0., 0.}),
                                                 Tensor<1, 3>({0., 0., -1.}),
                                                 Tensor<1, 3>({1., 2., 2.})}};

  for (const auto &tangent : tangents)
    {
      const auto points = support.transform(Point<3>(), tangent, 1., 0.25, 0);
      ASSERT_FALSE(points.empty());
      const auto expected = tangent / tangent.norm();
      for (const auto &point : points)
        {
          ASSERT_EQ(point.mode_values.size(), 3u);
          // Selected mode 2 is the reference e_z Cartesian vector.
          for (unsigned int component = 0; component < 3; ++component)
            EXPECT_NEAR(point.mode_values[component],
                        expected[component],
                        1.e-12);
        }
    }
}


TEST(ReferenceCrossSection, ScalarModesRemainUnchangedUnderRotation)
{
  ParameterAcceptor::clear();
  ReferenceCrossSectionParameters<2, 3, 1> parameters(
    "/Scalar rotation cross section/");
  parameters.inclusion_degree      = 0;
  parameters.refinement_level      = 1;
  parameters.selected_coefficients = {0};
  ReferenceCrossSection<2, 3, 1> section(parameters);

  for (const auto &tangent : {Tensor<1, 3>({0., 0., 1.}),
                              Tensor<1, 3>({1., 0., 0.}),
                              Tensor<1, 3>({1., 2., 2.})})
    {
      const auto values = section.get_transformed_mode_values(0, tangent);
      ASSERT_EQ(values.size(), 1u);
      EXPECT_NEAR(values[0], section.shape_value(0, 0, 0), 1.e-12);
    }
}


TEST(ReferenceCrossSection, CheckHyperSphereQuadrature) // NOLINT
{
  const int          dim          = 1;
  const int          spacedim     = 2;
  const int          n_components = 1;
  const unsigned int degree       = 0;

  ReferenceCrossSectionParameters<dim, spacedim, n_components> par;
  par.inclusion_degree = degree;
  par.inclusion_type   = "hyper_sphere";
  par.refinement_level = 5;

