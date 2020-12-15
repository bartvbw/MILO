/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef MSCONVDIFF_H
#define MSCONVDIFF_H

#include "physics_base.hpp"
class msconvdiff : public physicsbase {
public:
  
  msconvdiff() {} ;
  
  ~msconvdiff() {};
  
  // ========================================================================================
  // ========================================================================================
  
  msconvdiff(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
      const size_t & numip_side_, const int & numElem_,
      Teuchos::RCP<FunctionInterface> & functionManager_,
      const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
  
    label = "msconvdiff";
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    myvars.push_back("ca");
    myvars.push_back("cb");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    
    //velFromNS = settings->sublist("Physics").get<bool>("Get velocity from navierstokes",false);
    //burgersflux = settings->sublist("Physics").get<bool>("Add Burgers",false);
    
    // Functions
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("source",fs.get<string>("source","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("xdiffusion",fs.get<string>("xdiffusion","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("ydiffusion",fs.get<string>("ydiffusion","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("density",fs.get<string>("density","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("ca_reaction",fs.get<string>("ca_reaction","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("cb_reaction",fs.get<string>("cb_reaction","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("xvel",fs.get<string>("xvel","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("yvel",fs.get<string>("yvel","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("zvel",fs.get<string>("zvel","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("SUPG tau",fs.get<string>("SUPG tau","0.0"),numElem,numip,"ip",blocknum);
    
    //functionManager->addFunction("thermal Neumann source",fs.get<string>("thermal Neumann source","0.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("diffusion",fs.get<string>("diffusion","1.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),numElem,numip_side,"side ip",blocknum);
    
    //regParam = settings->sublist("Analysis").sublist("ROL").get<ScalarT>("regularization parameter",1.e-6);
    //moveVort = settings->sublist("Physics").get<bool>("moving vortices",true);
    //finTime = settings->sublist("Solver").get<ScalarT>("finaltime",1.0);
    //data_noise_std = settings->sublist("Analysis").get("Additive Normal Noise Standard Dev",0.0);
  }
  
  // ========================================================================================
  // ========================================================================================
 
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    int c_basis_num = wkset->usebasis[canum];
    basis = wkset->basis[c_basis_num];
    basis_grad = wkset->basis_grad[c_basis_num];
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source = functionManager->evaluate("source","ip",blocknum);
      xdiff = functionManager->evaluate("xdiffusion","ip",blocknum);
      ydiff = functionManager->evaluate("ydiffusion","ip",blocknum);
      cp = functionManager->evaluate("specific heat","ip",blocknum);
      rho = functionManager->evaluate("density","ip",blocknum);
      ca_reax = functionManager->evaluate("ca_reaction","ip",blocknum);
      cb_reax = functionManager->evaluate("cb_reaction","ip",blocknum);
      xvel = functionManager->evaluate("xvel","ip",blocknum);
      yvel = functionManager->evaluate("yvel","ip",blocknum);
      zvel = functionManager->evaluate("zvel","ip",blocknum);
      tau = functionManager->evaluate("SUPG tau","ip",blocknum);
    }
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    if (spaceDim == 1) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.extent(2); k++ ) {
          for (int i=0; i<basis.extent(1); i++ ) {
            resindex = offsets(canum,i); // TMW: e_num is not on the assembly device
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,canum,k,0)*basis(e,i,k) + // transient term
            xdiff(e,k)*(sol_grad(e,canum,k,0)*basis_grad(e,i,k,0)) + // diffusion terms
            (xvel(e,k)*sol_grad(e,canum,k,0))*basis(e,i,k) + // convection terms
            ca_reax(e,k)*basis(e,i,k) - source(e,k)*basis(e,i,k); // reaction and source terms
            
            res(e,resindex) += tau(e,k)*(rho(e,k)*cp(e,k)*sol_dot(e,canum,k,0) + xvel(e,k)*sol_grad(e,canum,k,0) + ca_reax(e,k) - source(e,k))*(xvel(e,k)*basis_grad(e,i,k,0));

            resindex = offsets(cbnum,i); // TMW: e_num is not on the assembly device
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,cbnum,k,0)*basis(e,i,k) + // transient term
            xdiff(e,k)*(sol_grad(e,cbnum,k,0)*basis_grad(e,i,k,0)) + // diffusion terms
            (xvel(e,k)*sol_grad(e,cbnum,k,0))*basis(e,i,k) + // convection terms
            cb_reax(e,k)*basis(e,i,k) - source(e,k)*basis(e,i,k); // reaction and source terms
            
            res(e,resindex) += tau(e,k)*(rho(e,k)*cp(e,k)*sol_dot(e,cbnum,k,0) + xvel(e,k)*sol_grad(e,cbnum,k,0) + cb_reax(e,k) - source(e,k))*(xvel(e,k)*basis_grad(e,i,k,0));

            
          }
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.extent(2); k++ ) {
	  //	    AD ca = sol(e,cnum,k,0);
	  AD dcadx = sol_grad(e,canum,k,0);
	  AD dcady = sol_grad(e,canum,k,1);
	  AD dcbdx = sol_grad(e,cbnum,k,0);
	  AD dcbdy = sol_grad(e,cbnum,k,1);
          for (int i=0; i<basis.extent(1); i++ ) {
            resindex = offsets(canum,i); 
	    res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,canum,k,0)*basis(e,i,k) + // transient term
	      (xdiff(e,k)*dcadx*basis_grad(e,i,k,0) + ydiff(e,k)*dcady*basis_grad(e,i,k,1)) + // diffusion terms
	      (xvel(e,k)*dcadx + yvel(e,k)*dcady)*basis(e,i,k) + // convection terms
	      ca_reax(e,k)*basis(e,i,k) - source(e,k)*basis(e,i,k); // reaction and source terms

	    res(e,resindex) += tau(e,k)*(rho(e,k)*cp(e,k)*sol_dot(e,canum,k,0) + xvel(e,k)*dcadx + yvel(e,k)*dcady       
					 + ca_reax(e,k) - source(e,k))*(xvel(e,k)*basis_grad(e,i,k,0) + yvel(e,k)*basis_grad(e,i,k,1));   //SUPG            

            resindex = offsets(cbnum,i); 
	    res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,cbnum,k,0)*basis(e,i,k) + // transient term
	      (xdiff(e,k)*dcbdx*basis_grad(e,i,k,0) + ydiff(e,k)*dcbdy*basis_grad(e,i,k,1)) + // diffusion terms
	      (xvel(e,k)*dcbdx + yvel(e,k)*dcbdy)*basis(e,i,k) + // convection terms
	      cb_reax(e,k)*basis(e,i,k) - source(e,k)*basis(e,i,k); // reaction and source terms


	    res(e,resindex) += tau(e,k)*(rho(e,k)*cp(e,k)*sol_dot(e,cbnum,k,0) + xvel(e,k)*dcbdx + yvel(e,k)*dcbdy           
					 + cb_reax(e,k) - source(e,k))*(xvel(e,k)*basis_grad(e,i,k,0) + yvel(e,k)*basis_grad(e,i,k,1));   //SUPG         
          }
        }
      });
    }
    else if (spaceDim == 3) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.extent(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.extent(2); k++ ) {
          for (int i=0; i<basis.extent(1); i++ ) {
            resindex = offsets(canum,i); // TMW: e_num is not on the assembly device
            res(e,resindex) += rho(e,k)*cp(e,k)*sol_dot(e,canum,k,0)*basis(e,i,k) + // transient term
            xdiff(e,k)*(sol_grad(e,canum,k,0)*basis_grad(e,i,k,0) + sol_grad(e,canum,k,1)*basis_grad(e,i,k,1) + sol_grad(e,canum,k,2)*basis_grad(e,i,k,2)) + // diffusion terms
            (xvel(e,k)*sol_grad(e,canum,k,0) + yvel(e,k)*sol_grad(e,canum,k,1) + zvel(e,k)*sol_grad(e,canum,k,2))*basis(e,i,k) + // convection terms
            ca_reax(e,k)*basis(e,i,k) - source(e,k)*basis(e,i,k); // reaction and source terms
            
            res(e,resindex) += tau(e,k)*(rho(e,k)*cp(e,k)*sol_dot(e,canum,k,0) + xvel(e,k)*sol_grad(e,canum,k,0) + yvel(e,k)*sol_grad(e,canum,k,1) + zvel(e,k)*sol_grad(e,canum,k,2) +ca_reax(e,k) - source(e,k))*(xvel(e,k)*basis_grad(e,i,k,0) + yvel(e,k)*basis_grad(e,i,k,1) + zvel(e,k)*basis_grad(e,i,k,2));
            
          }
        }
      });
    }
  }
  
  // ========================================================================================
  // ========================================================================================
 
  void boundaryResidual() {}
  
  // ========================================================================================
  // ========================================================================================
 
  void edgeResidual() {}
 
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================

  void computeFlux() {}

  // ========================================================================================
  // ========================================================================================
  
  void setVars(vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "ca") {
        canum = i;
      }
      if (varlist[i] == "cb") {
        cbnum = i;
      }
    }
  }
  
  // ========================================================================================
  // return the value of the stabilization parameter 
  // ========================================================================================
  
  template<class T>  
  T computeTau(const T & localdiff, const T & xvl, const T & yvl, const T & zvl, const ScalarT & h) const {
    
    ScalarT C1 = 4.0;
    ScalarT C2 = 2.0;
    
    T nvel;
    if (spaceDim == 1)
      nvel = xvl*xvl;
    else if (spaceDim == 2)
      nvel = xvl*xvl + yvl*yvl;
    else if (spaceDim == 3)
      nvel = xvl*xvl + yvl*yvl + zvl*zvl;
    
    if (nvel > 1E-12)
      nvel = sqrt(nvel);
    
    return 4.0/(C1*localdiff/h/h + C2*(nvel)/h); //msconvdiff has a 1.0 instead of a 4.0 in the numerator
    
  }
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  FDATA xdiff, ydiff, rho, cp, xvel, yvel, zvel, ca_reax, cb_reax, tau, source, nsource, diff_side, robin_alpha;
  
  int spaceDim, numElem;
  size_t numip, numip_side, blocknum;
  vector<string> varlist;
  int canum, cbnum,resindex;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::msconvdiff::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::msconvdiff::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::msconvdiff::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::msconvdiff::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::msconvdiff::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::msconvdiff::computeFlux() - evaluation of flux");
  
  
};

#endif
