//                                MFEM Example 5
//
// Compile with: make exStokes
//
// Sample runs:  exStokes -m ../data/inline-tri.mesh
//               exStokes -m ../data/iniline-quad.mesh
//               exStokes -m ../data/iniline-tet.mesh
//               exStokes -m ../data/iniline-hex.mesh
//
// Description:  This example code solves a lid driven 2D/3D mixed Stokes cavity test
//               corresponding to the saddle point system
//                                 -k*Delta u + grad p = f
//                                 - div u      = 0
//               on the unit square with essential boundary conditions:
//               and (u,v) = (1,0) on y=1, (u,v)=0 on other boundaries.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_bdr(const Vector & x, Vector & u);
void fFun(const Vector & x, Vector & f);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-tri.mesh";
   int order = 1;
   bool visualization = 1;
   double viscosity=1.;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(10000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }

   // 4. Define a finite element space on the mesh. Here we use the
   //    P_{k}/P_{k-1} finite elements where k is the defined order.
   FiniteElementCollection *vel_fecoll(new H1_FECollection(order+1, dim));
   FiniteElementCollection *pres_fecoll(new H1_FECollection(order, dim));

   FiniteElementSpace *Vh_space = new FiniteElementSpace(mesh, vel_fecoll, dim);
   FiniteElementSpace *Qh_space = new FiniteElementSpace(mesh, pres_fecoll, 1);

   // 5. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = Vh_space->GetVSize();
   block_offsets[2] = Qh_space->GetVSize();
   block_offsets.PartialSum();

   std::cout << "***********************************************************\n";
   std::cout << "dim(Vh) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(Qh) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(Vh+Qh) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // Mark essential (Dirichlet) boundary degrees of freedom
   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   Array<int> ess_dof;
   Vh_space->GetEssentialVDofs(ess_bdr, ess_dof);

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   VectorFunctionCoefficient fcoeff(dim, fFun);
   VectorFunctionCoefficient bdr_ucoeff(dim, uFun_bdr);


   // 7. Allocate memory (x, rhs) for the analytical solution and the
   //    right hand side.  Define the GridFunction u,p for the finite
   //    element solution and linear form fform or the right hand
   //    side. The data allocated by x and rhs are passed as a
   //    reference to the grid functions (u,p) and the linear form
   //    form fform.
   BlockVector x(block_offsets), rhs(block_offsets);

   GridFunction x_bdr;
   x_bdr.MakeRef(Vh_space, x.GetBlock(0), 0);
   x_bdr.ProjectCoefficient(bdr_ucoeff);

   // LinearForm *fform(new LinearForm);
   // fform->Update(Vh_space, rhs.GetBlock(0), 0);
   // fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   // fform->Assemble();

   // 8. Assemble the finite element matrices for the Stokes operator
   //
   //                            S = [ A  B^T ]
   //                                [ B   eps*M  ]
   //     where:
   //
   //     A = \int_\Omega k \nabla u_h \cdot \nabla v_h dx,  u_h, v_h \in V_h
   //     B   = -\int_\Omega \div u_h q_h dx,   u_h \in Q_h, q_h \in Q_h
   //     M   = \int_\Omega \epsilon p_h q_h dx,   p_h, q_h \in Q_h,
   //     \epsilon = small pressure penalization coefficient (for uniqueness)
   BilinearForm *aVarf(new BilinearForm(Vh_space));
   MixedBilinearForm *bVarf(new MixedBilinearForm(Vh_space, Qh_space));
   BilinearForm *mVarf(new BilinearForm(Qh_space));

   ConstantCoefficient viscosity_coeff(viscosity);
   aVarf->AddDomainIntegrator(new VectorDiffusionIntegrator(viscosity_coeff));
   aVarf->Assemble();
   aVarf->EliminateEssentialBC(ess_bdr, x_bdr, rhs.GetBlock(0));
   aVarf->Finalize();
   SparseMatrix &A(aVarf->SpMat());

   ConstantCoefficient minus_one(-1.0);
   bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator(minus_one));
   bVarf->Assemble();
   bVarf->EliminateTrialDofs(ess_bdr, x, rhs.GetBlock(1));
   bVarf->Finalize();
   SparseMatrix & B(bVarf->SpMat());
   SparseMatrix *BT = Transpose(B);

   ConstantCoefficient pressure_epsilon(1e-12);
   mVarf->AddDomainIntegrator(new MassIntegrator(pressure_epsilon));
   mVarf->Assemble();
   mVarf->Finalize();
   SparseMatrix &M(mVarf->SpMat());

   BlockMatrix stokesMatrix(block_offsets);
   stokesMatrix.SetBlock(0,0, &A);
   stokesMatrix.SetBlock(0,1, BT);
   stokesMatrix.SetBlock(1,0, &B);
   stokesMatrix.SetBlock(1,1, &M);

   // 9. Construct the operators for preconditioner
   //
   //                 P = [ diag(A)         0         ]
   //                     [  0       B diag(A)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement
   SparseMatrix *MinvBt = Transpose(B);
   Vector Ad(A.Height());
   A.GetDiag(Ad);
   for (int i = 0; i < Ad.Size(); i++)
   {
      MinvBt->ScaleRow(i, 1./Ad(i));
   }
   SparseMatrix *S = Mult(B, *MinvBt);

   Solver *invA, *invS;
   invA = new DSmoother(A);
#ifndef MFEM_USE_SUITESPARSE
   invS = new GSSmoother(*S);
#else
   invS = new UMFPackSolver(*S);
#endif

   invA->iterative_mode = false;
   invS->iterative_mode = false;

   BlockDiagonalPreconditioner stokesPrecond(block_offsets);
   stokesPrecond.SetDiagonalBlock(0, invA);
   stokesPrecond.SetDiagonalBlock(1, invS);

   // 10. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.
   int maxIter(1000);
   double rtol(1.e-6);
   double atol(1.e-10);

   chrono.Clear();
   chrono.Start();
   MINRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(stokesMatrix);
   solver.SetPreconditioner(stokesPrecond);
   solver.SetPrintLevel(1);
   x = 0.0;
   solver.Mult(rhs, x);
   chrono.Stop();

   if (solver.GetConverged())
      std::cout << "MINRES converged in " << solver.GetNumIterations()
                << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
   else
      std::cout << "MINRES did not converge in " << solver.GetNumIterations()
                << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
   std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";

   // 11. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p;
   u.MakeRef(Vh_space, x.GetBlock(0), 0);
   p.MakeRef(Qh_space, x.GetBlock(1), 0);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   // 12. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m exStokes.mesh -g sol_u.gf" or "glvis -m exStokes.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("exStokes.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream u_ofs("sol_u.gf");
      u_ofs.precision(8);
      u.Save(u_ofs);

      ofstream p_ofs("sol_p.gf");
      p_ofs.precision(8);
      p.Save(p_ofs);
   }

   // 13. Save data in the VisIt format
   VisItDataCollection visit_dc("ExampleStokes", mesh);
   visit_dc.RegisterField("velocity", &u);
   visit_dc.RegisterField("pressure", &p);
   visit_dc.Save();

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << *mesh << u << "window_title 'Velocity'" << endl;
      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << *mesh << p << "window_title 'Pressure'" << endl;
   }

   // 15. Free the used memory.
   // delete fform;
   delete invA;
   delete invS;
   delete S;
   delete MinvBt;
   delete BT;
   delete aVarf;
   delete bVarf;
   delete Qh_space;
   delete Vh_space;
   delete pres_fecoll;
   delete vel_fecoll;
   delete mesh;

   return 0;
}


void uFun_bdr(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   double top_boundary=1.0;
   double epsilon=1e-14;

   if( abs(yi-top_boundary)<epsilon )
     u(0) = xi*(1-xi);
   else
     u(0) = 0.;
   u(1) = 0.;
   if (x.Size() == 3)
   {
     u(2) = 0.;
   }
}

void fFun(const Vector & x, Vector & f)
{
   f = 0.0;
}
