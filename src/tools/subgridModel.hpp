/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef SUBGRIDMODEL_H
#define SUBGRIDMODEL_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "solutionStorage.hpp"

class SubGridModel {
public:
  
  SubGridModel() {} ;
  
  ~SubGridModel() {};
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  virtual int addMacro(const DRV macronodes_, Kokkos::View<int****,HostDevice> macrosideinfo_,
                       vector<string> & macrosidenames,
                       Kokkos::View<GO**,HostDevice> & macroGIDs,
                       Kokkos::View<LO***,HostDevice> & macroindex) = 0;

  
  virtual void finalize() = 0;
  
  virtual void subgridSolver(Kokkos::View<ScalarT***,AssemblyDevice> gl_u,
                             Kokkos::View<ScalarT***,AssemblyDevice> gl_phi,
                             const ScalarT & time, const bool & isTransient, const bool & isAdjoint,
                             const bool & compute_jacobian, const bool & compute_sens,
                             const int & num_active_params,
                             const bool & compute_disc_sens, const bool & compute_aux_sens,
                             workset & macrowkset, const int & macroelemindex,
                             const int & usernum,
                             Kokkos::View<ScalarT**,AssemblyDevice> subgradient, const bool & store_adjPrev) = 0;
  
  virtual Kokkos::View<ScalarT**,AssemblyDevice> computeError(const ScalarT & time,
                                                             const int & usernum) = 0;
  
  virtual Kokkos::View<AD*,AssemblyDevice> computeObjective(const string & response_type,
                                                            const int & seedwhat,
                                                            const ScalarT & time,
                                                            const int & usernum) = 0;
  
  virtual void writeSolution(const string & filename, const int & usernum) = 0;

  virtual void addSensors(const Kokkos::View<ScalarT**,HostDevice> sensor_points,
                          const ScalarT & sensor_loc_tol,
                          const vector<Kokkos::View<ScalarT**,HostDevice> > & sensor_data,
                          const bool & have_sensor_data,
                          const vector<basis_RCP> & basisTypes, const int & usernum) = 0;
  
  virtual matrix_RCP getProjectionMatrix() = 0;
  
  virtual DRV getIP() = 0;
  
  virtual DRV getIPWts() = 0;
  
  virtual pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis(const DRV & ip) = 0;

  virtual pair<Kokkos::View<int**,AssemblyDevice>, vector<DRV> > evaluateBasis2(const DRV & ip) = 0;
  
  virtual matrix_RCP getEvaluationMatrix(const DRV & newip, Teuchos::RCP<LA_Map> & ip_map) = 0;
  
  virtual Kokkos::View<GO**,HostDevice> getCellGIDs(const int & cellnum) = 0;
  
  virtual void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params, const vector<string> & paramnames) = 0;
  
  virtual Kokkos::View<ScalarT**,AssemblyDevice> getCellFields(const int & usernum, const ScalarT & time) = 0;
  
  virtual void addMeshData() = 0;

  virtual void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data) = 0;
  
  Teuchos::RCP<LA_MpiComm> LocalComm;
  //bvbw  Teuchos::RCP<SolutionStorage<LA_MultiVector> > soln, solndot, adjsoln;
  Teuchos::RCP<SolutionStorage<LA_MultiVector> > soln, solndot, adjsoln;
  //Teuchos::RCP<SolutionStorage<Tpetra::MultiVector<ScalarT,LO,int,HostNode> > > soln;
  
  bool useMachineLearning = false;
  
  vector<Teuchos::RCP<workset> > wkset;
  vector<basis_RCP> macro_basis_pointers;
  vector<string> macro_basis_types;
  vector<string> macro_varlist;
  vector<int> macro_usebasis;
  vector<vector<int> > macro_offsets;
  vector<string> macro_paramnames, macro_disc_paramnames;
  int macro_block;
  ScalarT cost_estimate;
  
  //bvbw  Teuchos::RCP<const LA_Map> owned_map, overlapped_map;
  Teuchos::RCP<const Tpetra::Map<LO, GO, HostNode> > owned_map, overlapped_map;
  //bvbw Teuchos::RCP<LA_CrsGraph> owned_graph, overlapped_graph;
  Teuchos::RCP<LA_CrsGraph> owned_graph;
  Teuchos::RCP<Tpetra::CrsGraph<LO,GO,HostNode> > overlapped_graph;
  
  //bvbw  Teuchos::RCP<LA_Export> exporter;
  Teuchos::RCP<Tpetra::Export<LO, GO, HostNode> > exporter;
  Teuchos::RCP<LA_Import> importer;
  
  vector<Teuchos::RCP<vector<AD> > > paramvals_AD;

  string usage;
  Kokkos::View<AD**,AssemblyDevice> paramvals_KVAD;
  
  
};
#endif
  
