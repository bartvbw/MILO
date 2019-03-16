/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef THERMAL_ENTHALPY_H
#define THERMAL_ENTHALPY_H

#include "physics_base.hpp"

static void thermal_enthalpyHelp() {
  cout << "********** Help and Documentation for the Thermal Enthalpy Physics Module **********" << endl << endl;
  cout << "Model:" << endl << endl;
  cout << "User defined functions: " << endl << endl;
}

class thermal_enthalpy : public physicsbase {
public:
  
  thermal_enthalpy() {} ;
  
  ~thermal_enthalpy() {};
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  thermal_enthalpy(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
                   const size_t & numip_side_, const int & numElem_,
                   Teuchos::RCP<FunctionInterface> & functionManager_,
                   const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
    
    label = "thermal_enthalpy";
    
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    
    myvars.push_back("e");
    myvars.push_back("H");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    
    if (settings->sublist("Physics").get<int>("solver",0) == 1)
      isTD = true;
    else
      isTD = false;
   
    multiscale = settings->isSublist("Subgrid"); 
    analysis_type = settings->sublist("Analysis").get<string>("analysis type","forward");
    
    numResponses = settings->sublist("Physics").get<int>("numResp_thermal",2); 
    useScalarRespFx = settings->sublist("Physics").get<bool>("use scalar response function (thermal)",false);
    
    formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
    
    have_nsvel = false;
    
    // Functions
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("thermal source",fs.get<string>("thermal source","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("specific heat",fs.get<string>("specific heat","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("density",fs.get<string>("density","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("thermal Neumann source",fs.get<string>("thermal Neumann source","0.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("thermal diffusion",fs.get<string>("thermal diffusion","1.0"),numElem,numip_side,"side ip",blocknum);
    functionManager->addFunction("robin alpha",fs.get<string>("robin alpha","0.0"),numElem,numip_side,"side ip",blocknum);
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
    
    e_basis = wkset->usebasis[e_num];
    numBasis = wkset->basis[e_basis].dimension(1);
    H_basis = wkset->usebasis[H_num];
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      source = functionManager->evaluate("thermal source","ip",blocknum);
      diff = functionManager->evaluate("thermal diffusion","ip",blocknum);
      cp = functionManager->evaluate("specific heat","ip",blocknum);
      rho = functionManager->evaluate("density","ip",blocknum);
    }


    sol = wkset->local_soln;
    sol_dot = wkset->local_soln_dot;
    sol_grad = wkset->local_soln_grad;
    
    offsets = wkset->offsets;
    
    res = wkset->res;
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    
    basis = wkset->basis[e_basis];
    basis_grad = wkset->basis_grad[e_basis];
    
    parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT v = 0.0;
      ScalarT dvdx = 0.0;
      ScalarT dvdy = 0.0;
      ScalarT dvdz = 0.0;
      
      for (int k=0; k<sol.dimension(2); k++ ) {
        AD T = sol(e,e_num,k,0);
        AD T_dot = sol_dot(e,e_num,k,0);
        AD dTdx = sol_grad(e,e_num,k,0);
        AD H = sol(e,H_num,k,0);
        AD H_dot = sol_dot(e,H_num,k,0);
        AD dHdx = sol_grad(e,H_num,k,0);
        AD dTdy, dHdy, dTdz, dHdz;
        if (spaceDim > 1) {
          dTdy = sol_grad(e,e_num,k,1);
          dHdy = sol_grad(e,H_num,k,1);
        }
        if (spaceDim > 2) {
          dTdz = sol_grad(e,e_num,k,2);
          dHdz = sol_grad(e,H_num,k,2);
        }
        AD ux, uy, uz;
        if (have_nsvel) {
          ux = sol(e,ux_num,k,0);
          if (spaceDim > 1) {
            uy = sol(e,uy_num,k,1);
          }
          if (spaceDim > 2) {
            uz = sol(e,uz_num,k,2);
          }
        }
        for (int i=0; i<basis.dimension(1); i++ ) {
          
          resindex = offsets(e_num,i);
          v = basis(e,i,k);
          dvdx = basis_grad(e,i,k,0);
          if (spaceDim > 1) {
            dvdy = basis_grad(e,i,k,1);
          }
          if (spaceDim > 2) {
            dvdz = basis_grad(e,i,k,2);
          }
          res(e,resindex) += H_dot*v + diff(e,k)*(dTdx*dvdx + dTdy*dvdy + dTdz*dvdz) - source(e,k)*v;
          if (have_nsvel) {
            res(e,resindex) += (ux*dvdx + uy*dvdy + uz*dvdz);
          }
        }
      }
    });
    

    basis = wkset->basis[H_basis];
    basis_grad = wkset->basis_grad[H_basis];
    
    parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
      
      ScalarT v = 0.0;
      ScalarT dvdx = 0.0;
      ScalarT dvdy = 0.0;
      ScalarT dvdz = 0.0;
      
      for (int k=0; k<sol.dimension(2); k++ ) {
        AD T = sol(e,e_num,k,0);
        AD H = sol(e,H_num,k,0);
        
        for (int i=0; i<basis.dimension(1); i++ ) {
          resindex = wkset->offsets(H_num,i);
          v = basis(e,i,k);
          // make cp_integral and gfunc udfuncs
          //cp_integral = 320.3*e + 0.379/2.0*e*e;
          AD cp_integral = 438.0*T + 0.169/2.0*T*T;
          //        if (e.val() <= 1648.0) {
          AD gfunc;
          if (T.val() <= 1673.0) {
            gfunc = 0.0;
          }
          //else if (e.val() >= 1673.0) {
          else if (T.val() >= 1723.0) {
            gfunc = 1.0;
          }
          else {
            //gfunc = (e - 1648.0)/(1673.0 - 1648.0);
            gfunc = (T - 1673.0)/(1723.0 - 1673.0);
          }
          // T_ref = 293.75
          //wkset->res(resindex) += -(H - rho(k)*cp_integral - rho(k)*latent_heat*gfunc + rho(k)*(320.3*293.75 + 0.379*293.75*293.75/2.0))*v;
          res(e,resindex) += -(H - rho(e,k)*cp_integral - rho(e,k)*latent_heat*gfunc + rho(e,k)*(438.0*293.75 + 0.169*293.75*293.75/2.0))*v;
        }
      }
    });

  }  
  
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    // NOTES:
    // 1. basis and basis_grad already include the integration weights
   
    int e_basis_num = wkset->usebasis[e_num];
    numBasis = wkset->basis_side[e_basis_num].dimension(1);
    
    {
      Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
      nsource = functionManager->evaluate("thermal Neumann source","side ip",blocknum);
      diff_side = functionManager->evaluate("thermal diffusion","side ip",blocknum);
      robin_alpha = functionManager->evaluate("robin alpha","side ip",blocknum);
    }
    
    ScalarT sf = formparam;
    if (wkset->isAdjoint) {
      sf = 1.0;
    }
    
    sideinfo = wkset->sideinfo;
    sol = wkset->local_soln_side;
    sol_grad = wkset->local_soln_grad_side;
    basis = wkset->basis_side[e_basis_num];
    basis_grad = wkset->basis_grad_side[e_basis_num];
    offsets = wkset->offsets;
    aux = wkset->local_aux_side;
    DRV ip = wkset->ip_side;
    DRV normals = wkset->normals;
    adjrhs = wkset->adjrhs;
    res = wkset->res;
    
    Teuchos::TimeMonitor localtime(*boundaryResidualFill);
    
    int cside = wkset->currentside;
    for (int e=0; e<sideinfo.dimension(0); e++) {
      if (sideinfo(e,e_num,cside,0) == 2) { // Element e is on the side
        for (int k=0; k<basis.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(e_num,i);
            res(e,resindex) += -nsource(e,k)*basis(e,i,k);
          }
        }
      }
      else if (sideinfo(e,e_num,cside,0) == 1){ // Weak Dirichlet
        for (int k=0; k<basis.dimension(2); k++ ) {
          AD eval = sol(e,e_num,k,0);
          dedx = sol_grad(e,e_num,k,0);
          ScalarT x = ip(e,k,0);
          ScalarT y = 0.0;
          ScalarT z = 0.0;
          if (spaceDim > 1) {
            dedy = sol_grad(e,e_num,k,1);
            y = ip(e,k,1);
          }
          if (spaceDim > 2) {
            dedz = sol_grad(e,e_num,k,2);
            z = ip(e,k,2);
          }
          
          if (sideinfo(e,e_num,cside,1) == -1)
            lambda = aux(e,e_num,k);
          else {
            lambda = 0.0; //udfunc->boundaryDirichletValue(label,"e",x,y,z,wkset->time,wkset->sidename,wkset->isAdjoint);
            
          //  lambda = this->getDirichletValue("e", x, y, z, wkset->time,
           //                                  wkset->sidename, wkset->isAdjoint);
          }
          
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(e_num,i);
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            if (spaceDim > 1)
              dvdy = basis_grad(e,i,k,1);
            if (spaceDim > 2)
              dvdz = basis_grad(e,i,k,2);
            
            weakDiriScale = 10.0*diff_side(e,k)/wkset->h(e);
            res(e,resindex) += -diff_side(e,k)*dedx*normals(e,k,0)*v - sf*diff_side(e,k)*dvdx*normals(e,k,0)*(eval-lambda) + weakDiriScale*(eval-lambda)*v;
            if (spaceDim > 1) {
              res(e,resindex) += -diff_side(e,k)*dedy*normals(e,k,1)*v - sf*diff_side(e,k)*dvdy*normals(e,k,1)*(eval-lambda);
            }
            if (spaceDim > 2) {
              res(e,resindex) += -diff_side(e,k)*dedz*normals(e,k,2)*v - sf*diff_side(e,k)*dvdz*normals(e,k,2)*(eval-lambda);
            }
            if (wkset->isAdjoint) {
              adjrhs(e,resindex) += sf*diff_side(e,k)*dvdx*normals(e,k,0)*lambda - weakDiriScale*lambda*v;
              if (spaceDim > 1)
                adjrhs(e,resindex) += sf*diff_side(e,k)*dvdy*normals(e,k,1)*lambda;
              if (spaceDim > 2)
                adjrhs(e,resindex) += sf*diff_side(e,k)*dvdz*normals(e,k,2)*lambda;
            }
          }
        }
      }
      
    }
    
  }
  
  // ========================================================================================
  // ========================================================================================

  void edgeResidual() {}
  
  // ========================================================================================
  // The boundary/edge flux
  // ========================================================================================
  
  void computeFlux() {

    ScalarT sf = 1.0;
    if (wkset->isAdjoint) {
      sf = formparam;
    }
    
    {
      Teuchos::TimeMonitor localtime(*fluxFunc);
      diff_side = functionManager->evaluate("thermal diffusion","side ip",blocknum);
    }
    
    
    {
      Teuchos::TimeMonitor localtime(*fluxFill);
      
      for (int n=0; n<numElem; n++) {
        
        for (size_t i=0; i<wkset->ip_side.dimension(1); i++) {
          penalty = 10.0*diff_side(n,i)/wkset->h(n);
          wkset->flux(n,e_num,i) += sf*diff_side(n,i)*wkset->local_soln_grad_side(n,e_num,i,0)*wkset->normals(n,i,0) + penalty*(wkset->local_aux_side(n,e_num,i)-wkset->local_soln_side(n,e_num,i,0));
          if (spaceDim > 1)
            wkset->flux(n,e_num,i) += sf*diff_side(n,i)*wkset->local_soln_grad_side(n,e_num,i,1)*wkset->normals(n,i,1);
          if (spaceDim > 2)
            wkset->flux(n,e_num,i) += sf*diff_side(n,i)*wkset->local_soln_grad_side(n,e_num,i,2)*wkset->normals(n,i,2);
          
        }
      }
    }
  }
  
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    ux_num = -1;
    uy_num = -1;
    uz_num = -1;
    
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "e")
        e_num = i;
      if (varlist[i] == "H")
        H_num = i;
      if (varlist[i] == "ux")
        ux_num = i;
      if (varlist[i] == "uy")
        uy_num = i;
      if (varlist[i] == "uz")
        uz_num = i;
    }
    if (ux_num >=0)
      have_nsvel = true;
  }
  
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  data grains;
 
  size_t numip, numip_side, blocknum;
  
  int spaceDim, numElem, numParams, numResponses;
  vector<string> varlist;
  int e_num, e_basis, numBasis, ux_num, uy_num, uz_num;
  int H_num, H_basis; // for melt fraction variable
  ScalarT alpha;
  bool isTD;
  //int test, simNum;
  //string simName;
  
  ScalarT v, dvdx, dvdy, dvdz, x, y, z;
  AD e, e_dot, dedx, dedy, dedz, reax, weakDiriScale, lambda, penalty;
  AD H, H_dot, dHdx, dHdy, dHdz; // spatial derivatives of g are not explicity needed atm
  AD ux, uy, uz;
  ScalarT latent_heat = 2.7e5;
  
  int resindex;
  
  FDATA diff, rho, cp, source, nsource, diff_side, robin_alpha;
  
  Kokkos::View<AD****,AssemblyDevice> sol, sol_dot, sol_grad;
  Kokkos::View<AD***,AssemblyDevice> aux;
  Kokkos::View<AD**,AssemblyDevice> res, adjrhs;
  Kokkos::View<int**,AssemblyDevice> offsets;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  DRV basis, basis_grad;
  
  string analysis_type; //to know when parameter is a sample that needs to be transformed
  
  bool useScalarRespFx;
  bool multiscale, have_nsvel;
  ScalarT formparam;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::thermal_enthalpy::computeFlux() - evaluation of flux");
  
  //Teuchos::RCP<DRVAD> src_test;
  //Teuchos::RCP<FunctionBase> diff_fct, rho_fct, cp_fct, source_fct, nsource_fct, diff_side_fct, robin_alpha_fct;
  
};

#endif
