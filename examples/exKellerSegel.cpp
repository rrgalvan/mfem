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
//               equation problem of the form
//
//               du^{m+1}/dt - k0 \Delta u^{m+1} = - k1 \nabla \cdot (u^m \nabla v^m),
//               dv^{m+1}/dt - k2 \Delta v^{m+1} = - k2 v^m + k3 u^m,
//
//               with homogeneous Neumann boundary conditions. The unknown u represents
//               biological cell distribution while v stands for a chemical substance
//               whose gradient influences cell growth

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** After spatial discretization, equations can be written as:
 *
 *     du/dt = M^{-1}( -KU u + R(u^m,v^m) )
 *     dv/dt = M^{-1}( -KV u + S(u^m,v^m) )
 *
 *  where u=u^{m+1} is the vector representing the density of live cells at
 *  t=t^{m+1}, M is the mass matrix, KU is the diffusion operator with
 *  diffusivity constant k0 and R(u^m,v^m) is the explicit cross diffusion
 *  operator R(u^m,v^m)(w) = k1 u^m \nabla v^m \nabla w.
 *
 *  On the other hand, v=v^{m+1} is the vector representing chemical
 *  substance at t=t^{m+1}, KV is the diffusion operator with diffusivity
 *  constant k2 and S(u^m, v^m) = -k3 v^m + k4 u^m.
 *
 *
 *  Class KellerSegelOperator represents the right-hand side of the above ODE.
 */
class KellerSegelOperator : public TimeDependentOperator
{
protected:
  FiniteElementSpace &fespace;
  Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

  BilinearForm *M;
  BilinearForm *KU;
  BilinearForm *KV;

  SparseMatrix Mmat, KUmat, KVmat;
  SparseMatrix *T; // T = M + dt KU
  double current_dt;

  CGSolver M_solver; // Krylov solver for inverting the mass matrix M
  DSmoother M_prec;  // Preconditioner for the mass matrix M

  CGSolver T_solver; // Implicit solver for T = M + dt KU
  DSmoother T_prec;  // Preconditioner for the implicit solver

  double k0, k1, k2, k3, k4;

  mutable Vector z; // auxiliary vector

public:
  KellerSegelOperator(FiniteElementSpace &f, double k0_, double k1_,
		      double k2_, double k3_, double k4_, const Vector &u);

  virtual void Mult(const Vector &uv, Vector &duv_dt) const;
  /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
      This is the only requirement for high-order SDIRK implicit integration.*/
  virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

  /// Update the diffusion BilinearForm KU using the given true-dof vector `u`.
  void SetParameters(const Vector &u);

  virtual ~KellerSegelOperator();
};

double InitialU(const Vector &x);

int main(int argc, char *argv[])
{
  // 1. Parse command-line options.
  const char *mesh_file = "../data/periodic-square.mesh";
  int ref_levels = 2;
  int order = 2;
  int ode_solver_type = 3;
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
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

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

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement, where 'ref_levels' is a
  //    command-line parameter.
  for (int lev = 0; lev < ref_levels; lev++)
    {
      mesh->UniformRefinement();
    }

  // 5. Define the vector finite element space representing the current and the
  //    initial temperature, u_ref.
  H1_FECollection fe_coll(order, dim);
  FiniteElementSpace fespace(mesh, &fe_coll);

  int fe_size = fespace.GetTrueVSize();
  cout << "Number of temperature unknowns: " << fe_size << endl;

  GridFunction u_gf(&fespace);

  // 6. Set the initial conditions for u. All boundaries are considered
  //    natural.
  FunctionCoefficient u_0(InitialU);
  u_gf.ProjectCoefficient(u_0);
  Vector u;
  u_gf.GetTrueDofs(u);

  // 7. Initialize the conduction operator and the visualization.
  KellerSegelOperator oper(fespace, k0, k1, k2, k3, k4, u);

  u_gf.SetFromTrueDofs(u);
  {
    ofstream omesh("exKellerSegel.mesh");
    omesh.precision(precision);
    mesh->Print(omesh);
    ofstream osol("exKellerSegel-init.gf");
    osol.precision(precision);
    u_gf.Save(osol);
  }

  VisItDataCollection visit_dc("ExampleKS", mesh);
  visit_dc.RegisterField("Keller-Segel", &u_gf);
  if (visit)
    {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
    }

  socketstream sout;
  if (visualization)
    {
      // char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
	{
	  cout << "Unable to connect to GLVis server at "
	       << vishost << ':' << visport << endl;
	  visualization = false;
	  cout << "GLVis visualization disabled.\n";
	}
      else
	{
	  sout.precision(precision);
	  sout << "solution\n" << *mesh << u_gf;
	  sout << "pause\n";
	  sout << flush;
	  cout << "GLVis visualization paused."
	       << " Press space (in the GLVis window) to resume it.\n";
	}
    }

  // 8. Perform time-integration (looping over the time iterations, ti, with a
  //    time-step dt).
  ode_solver->Init(oper);
  double t = 0.0;

  bool last_step = false;
  for (int ti = 1; !last_step; ti++)
    {
      if (t + dt >= t_final - dt/2)
	{
	  last_step = true;
	}

      ode_solver->Step(u, t, dt);

      if (last_step || (ti % vis_steps) == 0)
	{
	  cout << "step " << ti << ", t = " << t << endl;

	  u_gf.SetFromTrueDofs(u);
	  if (visualization)
	    {
	      sout << "solution\n" << *mesh << u_gf << flush;
	    }

	  if (visit)
	    {
	      visit_dc.SetCycle(ti);
	      visit_dc.SetTime(t);
	      visit_dc.Save();
	    }
	}
      oper.SetParameters(u);
    }

  // 9. Save the final solution. This output can be viewed later using GLVis:
  //    "glvis -m exKellerSegel.mesh -g exHeat-final.gf".
  {
    ofstream osol("exKellerSegel-final.gf");
    osol.precision(precision);
    u_gf.Save(osol);
  }

  // 10. Free the used memory.
  delete ode_solver;
  delete mesh;

  return 0;
}

KellerSegelOperator::KellerSegelOperator(FiniteElementSpace &f,
					 double k0_, double k1_, double k2_,
					 double k3_, double k4_, const Vector &u)
  : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f),
    M(NULL), KU(NULL), KV(NULL), T(NULL), current_dt(0.0), z(height)
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

  KU = new BilinearForm(&fespace);
  ConstantCoefficient k0_coeff(k0);
  KU->AddDomainIntegrator(new DiffusionIntegrator(k0_coeff));
  KU->Assemble();
  KU->FormSystemMatrix(ess_tdof_list, KUmat);

  KV = new BilinearForm(&fespace);
  ConstantCoefficient k2_coeff(k2);
  KV->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
  KV->Assemble();
  KV->FormSystemMatrix(ess_tdof_list, KVmat);

  T_solver.iterative_mode = false;
  T_solver.SetRelTol(rel_tol);
  T_solver.SetAbsTol(0.0);
  T_solver.SetMaxIter(100);
  T_solver.SetPrintLevel(0);
  T_solver.SetPreconditioner(T_prec);

  SetParameters(u);
}

void KellerSegelOperator::Mult(const Vector &uv, Vector &duv_dt) const
{
   // Create views to the sub-vectors u, v of uv, and du_dt, dv_dt of duv_dt
   int sc = height/2;
   Vector u(uv.GetData() +  0, sc);
   Vector v(uv.GetData() + sc, sc);
   Vector du_dt(duv_dt.GetData() +  0, sc);
   Vector dv_dt(duv_dt.GetData() + sc, sc);
  // Compute:
  //    du_dt = M^{-1}*(-KU(u) + R)
  // for du_dt
  KUmat.Mult(u, z);
  z.Neg(); // z = -z
  M_solver.Mult(z, du_dt);

   H.Mult(x, z);
   if (viscosity != 0.0)
   {
      S.AddMult(v, z);
   }
   z.Neg(); // z = -z
   M_solver.Mult(z, dv_dt);

   dx_dt = v;
}

void KellerSegelOperator::ImplicitSolve(const double dt,
					const Vector &u, Vector &du_dt)
{
  // Solve the equation:
  //    du_dt = M^{-1}*[-KU(u + dt*du_dt)]
  // for du_dt
  if (!T)
    {
      T = Add(1.0, Mmat, dt, KUmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
    }
  MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
  KUmat.Mult(u, z);
  z.Neg();
  T_solver.Mult(z, du_dt);
}

void KellerSegelOperator::SetParameters(const Vector &u)
{
  delete T;
  T = NULL; // re-compute T on the next ImplicitSolve
}

KellerSegelOperator::~KellerSegelOperator()
{
  delete T;
  delete M;
  delete KU;
  delete KV;
}

double InitialU(const Vector &x)
{
  if (x.Norml2() < 0.5)
    {
      return 2.0;
    }
  else
    {
      return 1.0;
    }
}
