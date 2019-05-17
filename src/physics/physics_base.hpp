/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef PHYSBASE_H
#define PHYSBASE_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "data.hpp"
#include "klexpansion.hpp"
#include "workset.hpp"
#include "functionInterface.hpp"

class physicsbase {
  
public:
  
  physicsbase() {} ;
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  ~physicsbase() {};
  
  // ========================================================================================
  // The volumetric contributions to the residual
  // ========================================================================================
  
  virtual
  void volumeResidual() = 0;

  // ========================================================================================
  // The boundary contributions to the residual
  // ========================================================================================
  
  virtual
  void boundaryResidual() = 0;
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  virtual
  void computeFlux() = 0;

  // ========================================================================================
  // Set the global index for each variable
  // ========================================================================================
  
  virtual void setVars(vector<string> & varlist_) = 0;
  
  // ========================================================================================
  // ========================================================================================
  
  virtual void updateParameters(const vector<Teuchos::RCP<vector<AD> > > & params, const std::vector<string> & paramnames) {} ;
  
  // ========================================================================================
  // ========================================================================================
  
  void setWorkset(Teuchos::RCP<workset> & wkset_) {
    wkset = wkset_;
    
    sol = wkset->local_soln;
    sol_dot = wkset->local_soln_dot;
    sol_grad = wkset->local_soln_grad;
    sol_div = wkset->local_soln_div;
    sol_curl = wkset->local_soln_curl;
    
    sol_side = wkset->local_soln_side;
    sol_grad_side = wkset->local_soln_grad_side;
    
    aux = wkset->local_aux;
    aux_side = wkset->local_aux_side;
    
    offsets = wkset->offsets;
    res = wkset->res;
    adjrhs = wkset->adjrhs;
    flux = wkset->flux;
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  string label;
  
  Teuchos::RCP<workset> wkset;
  Teuchos::RCP<FunctionInterface> functionManager;
  int spaceDim;
  vector<string> myvars, mybasistypes;
  
  // All of these point to specific information in the workset
  Kokkos::View<AD****,AssemblyDevice> sol, sol_dot, sol_grad, sol_side, sol_grad_side, aux_grad_side, sol_curl;
  Kokkos::View<AD***,AssemblyDevice> aux, aux_side, sol_div, flux;
  Kokkos::View<AD**,AssemblyDevice> res, adjrhs;
  Kokkos::View<int**,AssemblyDevice> offsets;
  
  // The basis functions change depending on the variable, so these cannot be set just once
  DRV basis, basis_grad, basis_div, basis_curl, normals;
  
  
};

#endif
