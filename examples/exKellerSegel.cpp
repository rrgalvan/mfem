//                      Linear Keller-Segel test, based on MFEM Example 16
//
// Compile with: make exKellerSegel
//
// Sample runs:  exKellerSegel
//               exKellerSegel -m ../data/inline-tri.mesh
//               exKellerSegel -m ../data/disc-nurbs.mesh -tf 2
//               exKellerSegel -s 1 -a 0.0 -k 1.0
//               exKellerSegel -s 2 -a 1.0 -k 0.0
//               exKellerSegel -s 3 -a 0.5 -k 0.5 -o 4
//               exKellerSegel -s 14 -dt 1.0e-4 -tf 4.0e-2 -vs 40
//               exKellerSegel -m ../data/fichera-q2.mesh
//               exKellerSegel -m ../data/escher.mesh
//               exKellerSegel -m ../data/beam-tet.mesh -tf 10 -dt 0.1
//               exKellerSegel -m ../data/amr-quad.mesh -o 4 -r 0
//               exKellerSegel -m ../data/amr-hex.mesh -o 2 -r 0
//
// Description:  This example solves a (uncoupled) time dependent Keller-Segel
//         equation problem of the form
//
//         du^{m+1}/dt - k0 \Delta u^{m+1} + k1 \nabla \cdot (u^{m+1} \nabla v^m) = 0
//         dv^{m+1}/dt - k2 \Delta v^{m+1} = - k2 v^m + k3 u^m,
//
//         with homogeneous Neumann boundary conditions. The unknown u represents
//         biological cell distribution while v stands for a chemical substance
//         whose gradient influences cell growth

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#define PRINT_INFO(x) {if(x) cout << "#" << (x) << endl;}

using namespace std;
using namespace mfem;

/** After spatial discretization, equations can be written as:
 *
 *     du/dt = M^{-1}( -K_u u + R(v^m)u )
 *     dv/dt = M^{-1}( -K_v v + S(u^m,v^m) )
 *
 *  where u=u^{m+1} is the vector representing the density of live cells at
 *  t=t^{m+1}, M is the mass matrix, K_u is the diffusion operator with
 *  diffusivity constant k0 and R(v^m)u is the explicit cross diffusion
 *  operator R(v^m) u w = k1 u \nabla v^m \nabla w.
 *
 *  On the other hand, v=v^{m+1} is the vector representing chemical
 *  substance at t=t^{m+1}, K_v is the diffusion operator with diffusivity
 *  constant k2 and S(u^m, v^m) = -k3 v^m + k4 u^m.
 *
 *  Class KellerSegelOperator represents the right-hand side of the above ODE.
 */
class KellerSegelOperator : public TimeDependentOperator
{
protected:
  FiniteElementSpace &fespace, &grad_fespace;
  Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

  BilinearForm *M;
  BilinearForm *K_u;
  BilinearForm *K_v;
  MixedBilinearForm *R; // To use MixedScalarWeakDivergenceIntegrator
  DiscreteLinearOperator *G; // For computing gradient

  SparseMatrix Mmat, KUmat, KVmat, Rmat, Gmat;
  SparseMatrix KuRmat; // KuRmat = - K_u + k2 R(v^m)
  SparseMatrix *T; // T = M + dt K_u
  double current_dt;

  CGSolver M_solver; // Krylov solver for inverting the mass matrix M
  DSmoother M_prec;  // Preconditioner for the mass matrix M

  CGSolver T_solver; // Implicit solver for T = M + dt K_u
  DSmoother T_prec;  // Preconditioner for the implicit solver

  double k0, k1, k2, k3, k4;

  mutable Vector z; // auxiliary vector

public:
  KellerSegelOperator(FiniteElementSpace &f, FiniteElementSpace &g_f,
		      double k0_, double k1_, double k2_, double k3_, double k4_,
		      const Vector &u, const Vector &v);

  virtual void Mult(const Vector &uv, Vector &duv_dt) const;
  /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
      This is the only requirement for high-order SDIRK implicit integration.*/
  virtual void ImplicitSolve(const double dt, const Vector &uv, Vector &duv_dt);

  /// Update the matrix KuRmat using the gradient of v
  void SetParameters(const Vector &u, const Vector &v);

  virtual ~KellerSegelOperator();
};

double InitialU(const Vector &x);
double InitialV(const Vector &x);

int main(int argc, char *argv[])
{
  // 1. Parse command-line options.
  const char *mesh_file = "../data/periodic-square.mesh";
  int ref_levels = 2;
  int order = 2;
  int ode_solver_type = 3; //11; //3;
  double t_final = 0.5;
  double dt = 1.0e-2;
  double k0 = 1.0;
  double k1 = 1.0;
  double k2 = 1.0;
  double k3 = 1.0;
  double k4 = 1.0;
  bool visualization = true;
  bool visit = false;
  int vis_steps = 5;
  const char *vishost = "localhost";

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
		 "Mesh file to use.");
  args.AddOption(&ref_levels, "-r", "--refine",
		 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
		 "Order (degree) of the finite elements.");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
		 "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
		 "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
  args.AddOption(&t_final, "-tf", "--t-final",
		 "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step",
		 "Time step.");
  args.AddOption(&k0, "-k0", "--k0",
		 "k0 coefficient.");
  args.AddOption(&k1, "-k1", "--k1",
		 "k1 coefficient.");
  args.AddOption(&k2, "-k2", "--k2",
		 "k2 coefficient.");
  args.AddOption(&k3, "-k3", "--k3",
		 "k3 coefficient.");
  args.AddOption(&k4, "-k4", "--k4",
		 "k4 coefficient.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
		 "--no-visualization",
		 "Enable or disable GLVis visualization.");
  args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
		 "--no-visit-datafiles",
		 "Save data files for VisIt (visit.llnl.gov) visualization.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps",
		 "Visualize every n-th timestep.");
  args.AddOption(&vishost, "-vh", "--vishost",
		 "Host running glvis.");
  args.Parse();
  if (!args.Good())
    {
      args.PrintUsage(cout);
      return 1;
    }
  args.PrintOptions(cout);

  // 2. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
  //
  //    Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement, where 'ref_levels' is a
  //    command-line parameter.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();
  for (int lev = 0; lev < ref_levels; lev++)
    {
      mesh->UniformRefinement();
    }

  // 3. Define the ODE solver used for time integration. Several implicit
  //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
  //    explicit Runge-Kutta methods are available.
  ODESolver *ode_solver;
  switch (ode_solver_type)
    {
      // Implicit L-stable methods
    case 1:  ode_solver = new BackwardEulerSolver; break;
    case 2:  ode_solver = new SDIRK23Solver(2); break;
    case 3:  ode_solver = new SDIRK33Solver; break;
      // Explicit methods
    case 11: ode_solver = new ForwardEulerSolver; break;
    case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
    case 13: ode_solver = new RK3SSPSolver; break;
    case 14: ode_solver = new RK4Solver; break;
      // Implicit A-stable methods (not L-stable)
    case 22: ode_solver = new ImplicitMidpointSolver; break;
    case 23: ode_solver = new SDIRK23Solver; break;
    case 24: ode_solver = new SDIRK34Solver; break;
    default:
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 3;
    }

  // 4. Define the finite element space representing the live
  //    cell density, u, and the chemical substance, v.
  H1_FECollection fe_coll(order, dim);
  FiniteElementSpace fespace(mesh, &fe_coll);

  //  Define the BlockStructure of the problem, i.e. define the array of
  // offsets for each variable. The last component of the Array is the sum
  //  of the dimensions of each block.
  Array<int> block_offsets(3); // number of variables + 1
  block_offsets[0] = 0;
  block_offsets[1] = fespace.GetVSize();
  block_offsets[2] = fespace.GetVSize();
  block_offsets.PartialSum();

  std::cout << "***********************************************************\n";
  std::cout << "dim(Vh) = " << block_offsets[1] - block_offsets[0] << "\n";
  std::cout << "dim(Qh) = " << block_offsets[2] - block_offsets[1] << "\n";
  std::cout << "dim(Vh+Qh) = " << block_offsets.Last() << "\n";
  std::cout << "***********************************************************\n";

  // 7. Allocate memory in uv for the analytical solution.  Define the GridFunction u,p for the finite
  //    element solution and linear form fform or the right hand
  //    side. The data allocated by x and rhs are passed as a
  //    reference to the grid functions (u,p) and the linear form
  //    form fform.
  BlockVector uv(block_offsets);

  Vector u, v;
  uv.GetBlockView(0, u);
  uv.GetBlockView(1, v);

  GridFunction u_gf(&fespace);
  GridFunction v_gf(&fespace);

  // Define also the vector finite element space representing the
  // gradient of v
  L2_FECollection grad_fec(order, dim);
  FiniteElementSpace grad_fespace(mesh, &grad_fec, dim);

  // 5. Set the initial conditions for u. All boundaries are considered
  //    natural.
  FunctionCoefficient u_0(InitialU);
  u_gf.ProjectCoefficient(u_0);
  u_gf.GetTrueDofs(u);
  FunctionCoefficient v_0(InitialV);
  v_gf.ProjectCoefficient(v_0);
  v_gf.GetTrueDofs(v);

  // 6. Visualize initial data
  socketstream u_sock, v_sock;
  if (visualization)
    {
      int  visport   = 19916;
      u_sock.open(vishost, visport);
      v_sock.open(vishost, visport);
      if (!u_sock || !v_sock)
	{
	  cout << "Unable to connect to GLVis server at "
	       << vishost << ':' << visport << endl;
	  visualization = false;
	  cout << "GLVis visualization disabled.\n";
	}
      else
	{
	  u_sock.precision(precision);
	  u_sock << "solution\n" << *mesh << u_gf << "window_title 'u'" << endl;
	  u_sock << "pause\n";
	  u_sock << flush;
	  cout << "GLVis visualization paused."
	       << " Press space (in the GLVis window) to resume it.\n";

	  v_sock.precision(precision);
	  v_sock << "solution\n" << *mesh << v_gf << "window_title 'v'" << endl;
	  v_sock << "pause\n";
	  v_sock << flush;
	}
    }

  // 7. Initialize the Keller-Segel operator

  KellerSegelOperator oper(fespace, grad_fespace, k0, k1, k2, k3, k4, u, v);

  PRINT_INFO(1000);

  // 8. Perform time-integration (looping over the time iterations, ti, with a
  //    time-step dt).
  double t = 0.0;
  oper.SetTime(t);
  ode_solver->Init(oper);

  PRINT_INFO(1001);

  bool last_step = false;
  for (int ti = 1; !last_step; ti++)
    {
      if (t + dt >= t_final - dt/2) last_step = true;

      ode_solver->Step(uv, t, dt);

      PRINT_INFO(1002);

      if (last_step || (ti % vis_steps) == 0)
	{
	  cout << "step " << ti << ", t = " << t << endl;

	  u_gf.SetFromTrueDofs(u);
	  v_gf.SetFromTrueDofs(v);
	  if (visualization)
	    {
	      u_sock << "solution\n" << *mesh << u_gf << "window_title 'u'" << endl << flush;
	      v_sock << "solution\n" << *mesh << v_gf << "window_title 'v'" << endl << flush;
	    }
	  // if (visit)
	  //   {
	  //     visit_dc.SetCycle(ti);
	  //     visit_dc.SetTime(t);
	  //     visit_dc.Save();
	  //   }
	}

      oper.SetParameters(u,v);
      PRINT_INFO(1004);
    }
  // 10. Free the used memory.
  delete ode_solver;
  delete mesh;

  return 0;
} // END main

KellerSegelOperator::KellerSegelOperator(FiniteElementSpace &f,
					 FiniteElementSpace &g_f,
					 double k0_, double k1_, double k2_,
					 double k3_, double k4_,
					 const Vector& u,
					 const Vector& v)
  : TimeDependentOperator(2*f.GetTrueVSize(), 0.0),
    fespace(f), grad_fespace(g_f),
    M(NULL), K_u(NULL), K_v(NULL), R(NULL), G(NULL), T(NULL),
    current_dt(0.0), z(height)
{
  const double rel_tol = 1e-8;

  k0 = k0_; k1 = k1_; k2 = k2_; k3 = k3_; k4 = k4_;

  M = new BilinearForm(&fespace);
  M->AddDomainIntegrator(new MassIntegrator());
  M->Assemble();
  M->FormSystemMatrix(ess_tdof_list, Mmat);

  M_solver.iterative_mode = false;
  M_solver.SetRelTol(rel_tol);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(30);
  M_solver.SetPrintLevel(0);
  M_solver.SetPreconditioner(M_prec);
  M_solver.SetOperator(Mmat);

  K_u = new BilinearForm(&fespace);
  ConstantCoefficient k0_coeff(k0);
  K_u->AddDomainIntegrator(new DiffusionIntegrator(k0_coeff));
  K_u->Assemble();
  K_u->FormSystemMatrix(ess_tdof_list, KUmat);

  K_v = new BilinearForm(&fespace);
  ConstantCoefficient k2_coeff(k2);
  K_v->AddDomainIntegrator(new DiffusionIntegrator(k2_coeff));
  K_v->Assemble();
  K_v->FormSystemMatrix(ess_tdof_list, KVmat);

  // PRINT_INFO(1)

  // G = new DiscreteLinearOperator(&fespace, &grad_fespace);
  // G->AddDomainInterpolator(new GradientInterpolator);
  // G->Assemble();
  // G->Finalize();
  // // G->FormSystemMatrix(ess_tdof_list, Gmat);

  PRINT_INFO(2);

    T_solver.iterative_mode = false;
  T_solver.SetRelTol(rel_tol);
  T_solver.SetAbsTol(0.0);
  T_solver.SetMaxIter(100);
  T_solver.SetPrintLevel(0);
  T_solver.SetPreconditioner(T_prec);

  PRINT_INFO(3);

    SetParameters(u, v);

  PRINT_INFO(300);
} // END KellerSegelOperator::KellerSegelOperator()

void KellerSegelOperator::Mult(const Vector &uv, Vector &duv_dt) const
{
  // Create views to the sub-vectors u, v of uv, and du_dt, dv_dt of duv_dt
  int sc = height/2;
  Vector u(uv.GetData() +  0, sc);
  Vector v(uv.GetData() + sc, sc);
  Vector du_dt(duv_dt.GetData() +  0, sc);
  Vector dv_dt(duv_dt.GetData() + sc, sc);

  // Compute: du/dt = M^{-1}( (-K_u + k1 R(v^m) ) u )
  // KuRmat.Mult(u, z);

  // TODO: CHANGE KUmat -> KuRmat
  KUmat.Mult(u, z);
  z.Neg(); // DELETE?
  M_solver.Mult(z, du_dt);

  // Compute: dv/dt = M^{-1}( -K_v v + S(u^m,v^m) )
  KVmat.Mult(v, z);
  z.Neg(); // z = -z
  M_solver.Mult(z, dv_dt);

  // TODO: TENER EN CUENTA S(u^m,v^m) !!!
}

void KellerSegelOperator::ImplicitSolve(const double dt,
					const Vector &uv, Vector &duv_dt)
{
  // Create views to the sub-vectors u, v of uv, and du_dt, dv_dt of duv_dt
  int sc = height/2;
  Vector u(uv.GetData() +  0, sc);
  Vector v(uv.GetData() + sc, sc);
  Vector du_dt(duv_dt.GetData() +  0, sc);
  Vector dv_dt(duv_dt.GetData() + sc, sc);

  // Solve the equation:
  //    du_dt = M^{-1}*[ KuRMat (u + dt*du_dt)]
  // for du_dt, where KuRMat = (-K_u + k1 R(v^m))
  if (!T)
    {
      // TODO: CHANGE KUmat -> KuRmat
      // T = Add(1.0, Mmat, dt, KuRmat);
      T = Add(1.0, Mmat, dt, KUmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
    }
  MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt

  // KuRmat.Mult(u, z);
  // TODO: CHANGE KUmat -> KuRmat
  KUmat.Mult(u, z);
  z.Neg();
  T_solver.Mult(z, du_dt);

  // Solve the equation:
  //    dv_dt = M^{-1}*[ -K_v (v + dt*dv_dt) + S(u^m,v^m) )]
  // for dv_dt
  delete T;
  T = Add(1.0, Mmat, dt, KVmat);
  current_dt = dt;
  T_solver.SetOperator(*T);
  MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt

  KVmat.Mult(v, z);
  z.Neg();
  T_solver.Mult(z, dv_dt);

  // TODO: TENER EN CUENTA S(u^m,v^m) !!!
}

void KellerSegelOperator::SetParameters(const Vector &u, const Vector &v)
{
  // // Compute gradient of v
  // Vector grad_v( grad_fespace.GetTrueVSize() );
  // // Gmat.Mult(v, grad_v);
  // G->Mult(v, grad_v);

  // PRINT_INFO(10);

  // // Define R(v) u = div(u grad_v) i.e. (R(v) u, w) = (-grad_v u, grad(w))
  // delete R;
  // R = new MixedBilinearForm(&fespace,&fespace);
  // PRINT_INFO(11);
  // VectorConstantCoefficient gv_coeff(grad_v);
  // PRINT_INFO(12);
  // R->AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(gv_coeff));
  // PRINT_INFO(13);
  // R->Assemble();
  // PRINT_INFO(14);
  // R->Finalize();
  // PRINT_INFO(15);
  // // R->FormSystemMatrix(ess_tdof_list, Rmat);
  // Rmat = R->SpMat();

  // PRINT_INFO(20);

  // // Define KuR(v) = - K_u + k1 R(v)
  // KuRmat.Clear();
  // KuRmat.Add(-1.0, KUmat);
  // KuRmat.Add(  k1, Rmat);

  // PRINT_INFO(100);

  delete T;
  T = NULL; // re-compute T on the next ImplicitSolve
}

KellerSegelOperator::~KellerSegelOperator()
{
  delete T;
  delete M;
  delete K_u;
  delete K_v;
  delete R;
  delete G;
}

double InitialU(const Vector &x)
{
  double norm_x2 = x.Norml2();
  norm_x2 *= norm_x2;
  if (norm_x2 < 0.5)
    {
      return 0.5-norm_x2;
    }
  else
    {
      return 0.0;
    }
}


double InitialV(const Vector &x)
{
  return InitialU(x);
}
