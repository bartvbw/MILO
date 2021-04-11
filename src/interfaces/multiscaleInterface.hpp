/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MULTISCALE_INT_H
#define MULTISCALE_INT_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "cell.hpp"
#include "subgridModel.hpp"
#include "Amesos2.hpp"

using namespace std;
using namespace Intrepid2;

void static multiscaleHelp(const string & details) {
  cout << "********** Help and Documentation for the Multiscale Interface **********" << endl;
}

class MultiScale {
  public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  MultiScale(const Teuchos::RCP<LA_MpiComm> & MacroComm_,
             const Teuchos::RCP<LA_MpiComm> & Comm_,
             Teuchos::RCP<Teuchos::ParameterList> & settings_,
             vector<vector<Teuchos::RCP<cell> > > & cells_,
             vector<Teuchos::RCP<SubGridModel> > subgridModels_,
             Teuchos::RCP<FunctionInterface> macro_functionManager_);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Set the information from the macro-scale that does not depend on the specific cell
  ////////////////////////////////////////////////////////////////////////////////
  
  void setMacroInfo(vector<vector<basis_RCP> > & macro_basis_pointers,
                    vector<vector<string> > & macro_basis_types,
                    vector<vector<string> > & macro_varlist,
                    vector<vector<int> > macro_usebasis,
                    vector<vector<vector<int> > > & macro_offsets,
                    vector<string> & macro_paramnames,
                    vector<string> & macro_disc_paramnames);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Initial assignment of subgrid models to cells
  ////////////////////////////////////////////////////////////////////////////////
  
  ScalarT initialize();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Re-assignment of subgrid models to cells
  ////////////////////////////////////////////////////////////////////////////////
  
  ScalarT update();
  
  void reset();
  
  ////////////////////////////////////////////////////////////////////////////////
  // Post-processing
  ////////////////////////////////////////////////////////////////////////////////
  
  void writeSolution(const string & macrofilename, const vector<ScalarT> & solvetimes,
                     const int & globalPID);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Update parameters
  ////////////////////////////////////////////////////////////////////////////////
  
  void updateParameters(vector<Teuchos::RCP<vector<AD> > > & params,
                        const vector<string> & paramnames);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Get the mean subgrid cell fields
  ////////////////////////////////////////////////////////////////////////////////
  
  
  Kokkos::View<ScalarT**,HostDevice> getMeanCellFields(const size_t & block, const int & timeindex,
                                                      const ScalarT & time, const int & numfields);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Update the mesh data (for UQ studies)
  ////////////////////////////////////////////////////////////////////////////////
  
  void updateMeshData(Kokkos::View<ScalarT**,HostDevice> & rotation_data);

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  
  bool subgrid_static;
  int milo_debug_level;
  vector<Teuchos::RCP<SubGridModel> > subgridModels;
  Teuchos::RCP<LA_MpiComm> Comm, MacroComm;
  Teuchos::RCP<Teuchos::ParameterList> settings;
  vector<vector<Teuchos::RCP<cell> > > cells;
  vector<Teuchos::RCP<workset> > macro_wkset;
  vector<vector<Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode> > > > subgrid_projection_maps;
  vector<Teuchos::RCP<Amesos2::Solver<Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>,Tpetra::MultiVector<ScalarT,LO,GO,HostNode> > > > subgrid_projection_solvers;
  Teuchos::RCP<FunctionInterface> macro_functionManager;
};

#endif
