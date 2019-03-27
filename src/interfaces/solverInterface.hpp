/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SOLVER_H
#define SOLVER_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "multiscaleInterface.hpp"
#include "discretizationInterface.hpp"
#include "discretizationTools.hpp"
#include "cell.hpp"
#include "assemblyManager.hpp"
#include "parameterManager.hpp"

void static solverHelp(const string & details) {
  cout << "********** Help and Documentation for the Solver Interface **********" << endl;
}

class solver {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  solver(const Teuchos::RCP<LA_MpiComm> & Comm_, Teuchos::RCP<Teuchos::ParameterList> & settings,
         Teuchos::RCP<meshInterface> & mesh_,
         Teuchos::RCP<discretization> & disc_,
         Teuchos::RCP<physics> & phys_, Teuchos::RCP<panzer::DOFManager<int,int> > & DOF_,
         Teuchos::RCP<AssemblyManager> & assembler_,
         Teuchos::RCP<ParameterManager> & params_);
  
  
  // ========================================================================================
  // Set up the Epetra objects (maps, importers, exporters and graphs)
  // These do need to be recomputed whenever the mesh changes */
  // ========================================================================================
  
  void setupLinearAlgebra();
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Epetra_CrsGraph> buildEpetraOverlappedGraph(Epetra_MpiComm & EP_Comm);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<Epetra_CrsGraph> buildEpetraOwnedGraph(Epetra_MpiComm & EP_Comm);
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  void finalizeWorkset();
  
  /////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////
  
  vector_RCP forwardModel(DFAD & obj);
  
  // ========================================================================================
  /* given the parameters, solve the fractional forward  problem */
  // ========================================================================================
  
  vector_RCP forwardModel_fr(DFAD & obj, ScalarT yt, ScalarT st);
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP adjointModel(vector_RCP & F_soln, vector<ScalarT> & gradient);
  
  
  // ========================================================================================
  /* solve the problem */
  // ========================================================================================
  
  void transientSolver(vector_RCP & initial, vector_RCP & L_soln,
                       vector_RCP & SolMat, DFAD & obj, vector<ScalarT> & gradient);
  
  // ========================================================================================
  // ========================================================================================
  
  void nonlinearSolver(vector_RCP & u, vector_RCP & u_dot,
                       vector_RCP & phi, vector_RCP & phi_dot,
                       const ScalarT & alpha, const ScalarT & beta);
  
  
  
  
  
  
  // ========================================================================================
  // ========================================================================================
  
  DFAD computeObjective(const vector_RCP & F_soln, const ScalarT & time, const size_t & tindex);
  
  // ========================================================================================
  // ========================================================================================
  
  vector<ScalarT> computeSensitivities(const vector_RCP & GF_soln,
                                      const vector_RCP & GA_soln);
  
  // ========================================================================================
  // Compute the sensitivity of the objective with respect to discretized parameters
  // ========================================================================================
  
  vector<ScalarT> computeDiscretizedSensitivities(const vector_RCP & F_soln,
                                                 const vector_RCP & A_soln);
  
  // ========================================================================================
  // ========================================================================================
  
  void computeSensitivities(vector_RCP & u, vector_RCP & u_dot,
                            vector_RCP & a2, vector<ScalarT> & gradient,
                            const ScalarT & alpha, const ScalarT & beta);
  
  // ========================================================================================
  // The following function is the adjoint-based error estimate
  // Not to be confused with the postprocess::computeError function which uses a true
  //   solution to perform verification studies
  // ========================================================================================
  
  ScalarT computeError(const vector_RCP & GF_soln, const vector_RCP & GA_soln);
  
  
  
  
  
  
  
  // ========================================================================================
  // ========================================================================================
  
  void setDirichlet(vector_RCP & initial);
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP setInitialParams();
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP setInitial();
  
  // ========================================================================================
  // Linear solver for Tpetra stack
  // ========================================================================================
  
  void linearSolver(matrix_RCP & J, vector_RCP & r, vector_RCP & soln);
  
  // ========================================================================================
  // Linear solver for Epetra stack
  // ========================================================================================
  
  void linearSolver(Teuchos::RCP<Epetra_CrsMatrix> & J,
                    Teuchos::RCP<Epetra_MultiVector> & r,
                    Teuchos::RCP<Epetra_MultiVector> & soln);
  
  // ========================================================================================
  // Preconditioner for Tpetra stack
  // ========================================================================================
  
  Teuchos::RCP<MueLu::TpetraOperator<ScalarT, LO, GO, HostNode> > buildPreconditioner(const matrix_RCP & J);
  
  // ========================================================================================
  // Preconditioner for Epetra stack
  // ========================================================================================
  
  ML_Epetra::MultiLevelPreconditioner* buildPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix> & J);
  
  // ========================================================================================
  // ========================================================================================
  
  void setBatchID(const int & bID);
  
  // ========================================================================================
  // ========================================================================================
  
  vector_RCP blankState();
  
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  void finalizeMultiscale() ;
  
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Public data members
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<AssemblyManager> assembler;
  Teuchos::RCP<ParameterManager> params;
  Teuchos::RCP<meshInterface>  mesh;
    
  Teuchos::RCP<const LA_Map> LA_owned_map, LA_overlapped_map;
  Teuchos::RCP<LA_CrsGraph> LA_owned_graph, LA_overlapped_graph;
  Teuchos::RCP<LA_Export> exporter;
  Teuchos::RCP<LA_Import> importer;
  
  int numUnknowns, numUnknownsOS, globalNumUnknowns, MaxNLiter, time_order, liniter, kspace;
  int verbosity, batchID, spaceDim, numsteps, gNLiter;
  
  vector<LO> owned, ownedAndShared, LA_owned, LA_ownedAndShared;
  
  ScalarT NLtol, finaltime, lintol, dropTol, fillParam, current_time;
  
  string solver_type, NLsolver, initial_type, response_type, multigrid_type, smoother_type;
  
  bool line_search, useL2proj, allow_remesh, useDomDecomp, useDirect, usePrec, discretized_stochastic;
  bool isInitial, isTransient, useadjoint, is_final_time, usestrongDBCs;
  bool compute_objective, compute_sensitivity, use_custom_initial_param_guess, store_adjPrev, use_meas_as_dbcs;
  
  vector<ScalarT> solvetimes;
  
  vector<vector_RCP> fwdsol;
  vector<vector_RCP> adjsol;
  vector<string> blocknames;
  vector<vector<string> > varlist;
  
  vector<vector<int> > numBasis, useBasis;
  vector<int> maxBasis, numVars;
  
  Teuchos::RCP<MultiScale> multiscale_manager;
  
private:
  
  Teuchos::RCP<LA_MpiComm> Comm;
  Teuchos::RCP<discretization> disc;
  Teuchos::RCP<physics> phys;
  Teuchos::RCP<const panzer::DOFManager<int,int> > DOF;
  
  Teuchos::RCP<Teuchos::Time> assemblytimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - total assembly");
  Teuchos::RCP<Teuchos::Time> linearsolvertimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::linearSolver()");
  Teuchos::RCP<Teuchos::Time> gathertimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - gather");
  Teuchos::RCP<Teuchos::Time> phystimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - physics evaluation");
  Teuchos::RCP<Teuchos::Time> boundarytimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - boundary evaluation");
  Teuchos::RCP<Teuchos::Time> inserttimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - insert");
  Teuchos::RCP<Teuchos::Time> dbctimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - strong Dirichlet BCs");
  Teuchos::RCP<Teuchos::Time> completetimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - fill complete");
  Teuchos::RCP<Teuchos::Time> msprojtimer = Teuchos::TimeMonitor::getNewCounter("MILO::solver::computeJacRes() - multiscale projection");
  
};

#endif
