/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef TWOPHASEPONO_H
#define TWOPHASEPONO_H

#include "physics_base.hpp"

class twophasePoNo : public physicsbase {
public:
  
  // ========================================================================================
  /* Constructor to set up the problem */
  // ========================================================================================
  
  twophasePoNo() {} ;
  
  ~twophasePoNo() {};
  
  twophasePoNo(Teuchos::RCP<Teuchos::ParameterList> & settings, const int & numip_,
               const size_t & numip_side_, const int & numElem_,
               Teuchos::RCP<FunctionInterface> & functionManager_,
               const size_t & blocknum_) :
  numip(numip_), numip_side(numip_side_), numElem(numElem_), functionManager(functionManager_),
  blocknum(blocknum_) {
    
    // Standard data
    label = "twophase";
    spaceDim = settings->sublist("Mesh").get<int>("dim",2);
    myvars.push_back("Po");
    myvars.push_back("No");
    mybasistypes.push_back("HGRAD");
    mybasistypes.push_back("HGRAD");
    formparam = settings->sublist("Physics").get<ScalarT>("form_param",1.0);
    
    
    // Functions
    Teuchos::ParameterList fs = settings->sublist("Functions");
    
    functionManager->addFunction("permeability",fs.get<string>("permeability","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("porosity",fs.get<string>("porosity","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("gravity",fs.get<string>("gravity","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("cap press",fs.get<string>("capillary pressure","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("dcap press",fs.get<string>("derivative capillary pressure","0.0"),numElem,numip,"ip",blocknum);
    
    functionManager->addFunction("source oil",fs.get<string>("source oil","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("viscosity oil",fs.get<string>("viscosity oil","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("relative permeability oil",fs.get<string>("relative permeability oil","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference density oil",fs.get<string>("reference density oil","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference pressure oil",fs.get<string>("reference pressure oil","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("compressibility oil",fs.get<string>("compressibility oil","0.0"),numElem,numip,"ip",blocknum);
    
    functionManager->addFunction("source water",fs.get<string>("source water","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("viscosity water",fs.get<string>("viscosity water","0.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("relative permeability water",fs.get<string>("relative permeability water","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference density water",fs.get<string>("reference density water","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("reference pressure water",fs.get<string>("reference pressure water","1.0"),numElem,numip,"ip",blocknum);
    functionManager->addFunction("compressibility water",fs.get<string>("compressibility water","0.0"),numElem,numip,"ip",blocknum);
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void volumeResidual() {
    
    // NOTES:
    // sol, sol_grad, etc. are set by the physics_base class
    
    // This formulation solves for N_o and P_o
    
    // This does assume that both pw and po use the same basis ... easy to generalize
    
    
    int p_basis_num = wkset->usebasis[Ponum];
    basis = wkset->basis[p_basis_num];
    basis_grad = wkset->basis_grad[p_basis_num];
    
    {
      Teuchos::TimeMonitor funceval(*volumeResidualFunc);
      porosity = functionManager->evaluate("porosity","ip",blocknum);
      perm = functionManager->evaluate("permeability","ip",blocknum);
      
      relperm_o = functionManager->evaluate("relative permeability oil","ip",blocknum);
      source_o = functionManager->evaluate("source oil","ip",blocknum);
      viscosity_o = functionManager->evaluate("viscosity oil","ip",blocknum);
      densref_o = functionManager->evaluate("reference density oil","ip",blocknum);
      pref_o = functionManager->evaluate("reference pressure oil","ip",blocknum);
      comp_o = functionManager->evaluate("compressibility oil","ip",blocknum);
      
      relperm_w = functionManager->evaluate("relative permeability water","ip",blocknum);
      source_w = functionManager->evaluate("source water","ip",blocknum);
      viscosity_w = functionManager->evaluate("viscosity water","ip",blocknum);
      densref_w = functionManager->evaluate("reference density water","ip",blocknum);
      pref_w = functionManager->evaluate("reference pressure water","ip",blocknum);
      comp_w = functionManager->evaluate("compressibility water","ip",blocknum);
      
      gravity = functionManager->evaluate("gravity","ip",blocknum);
      cp = functionManager->evaluate("cap press","ip",blocknum);
      dcp = functionManager->evaluate("dcap press","ip",blocknum);
      
    }
    
    Teuchos::TimeMonitor resideval(*volumeResidualFill);
    //AD rho_o, rho_w, S_o, S_w, P_w, dP_o_dt, dS_w_dt, dP_w_dt, drho_w_dt, dN_w_dt;
    //AD dS_w_dx, dP_w_dx, dS_w_dy, dP_w_dy, dS_w_dz, dP_w_dz;
    
    if (spaceDim == 1) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) { // loop over integration points
          AD Po = sol(e,Ponum,k,0);
          AD No = sol(e,Nonum,k,0);
          AD dPo_dx = sol_grad(e,Ponum,k,0);
          AD dPo_dy = sol_grad(e,Ponum,k,1);
          AD dPo_dz = sol_grad(e,Ponum,k,2);
          AD dNo_dx = sol_grad(e,Nonum,k,0);
          AD dNo_dy = sol_grad(e,Nonum,k,1);
          AD dNo_dz = sol_grad(e,Nonum,k,2);
          AD dNo_dt = sol_dot(e,Nonum,k,0);
          AD dPo_dt = sol_dot(e,Ponum,k,0);
          
          AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
          AD So = No / rhoo;
          AD Sw = 1.0 - So;
          //AD dPo_dt = densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dt);
          AD dSw_dt = -1.0/(rhoo*rhoo)*(dNo_dt*rhoo - No*dPo_dt);
          AD Pw = Po - cp(e,k);
          AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
          AD dPw_dt = dPo_dt - dcp(e,k)*dSw_dt;
          AD drhow_dt = densref_w(e,k)*(0.0+comp_w(e,k)*dPw_dt);
          AD dNw_dt = Sw*drhow_dt + dSw_dt*rhow;
          AD dSw_dx = -1.0/(rhoo*rhoo)*(rhoo*dNo_dx - No*densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dx));
          AD dPw_dx = dPo_dx - dcp(e,k)*dSw_dx;
          
          for (int i=0; i<basis.dimension(1); i++ ) { // loop over basis functions
            
            // No equation
            resindex = offsets(Nonum,i);
            
            res(e,resindex) += porosity(e,k)*dNo_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo*(dPo_dx*basis_grad(e,i,k,0)) // diffusion terms
            - source_o(e,k)*basis(e,i,k);
            
            // Po equation
            resindex = offsets(Ponum,i);
            
            res(e,resindex) += porosity(e,k)*dNw_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow*(dPw_dx*basis_grad(e,i,k,0)) // diffusion terms
            -source_w(e,k)*basis(e,i,k);
          }
        }
      });
    }
    else if (spaceDim == 2) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          AD Po = sol(e,Ponum,k,0);
          AD No = sol(e,Nonum,k,0);
          AD dPo_dx = sol_grad(e,Ponum,k,0);
          AD dPo_dy = sol_grad(e,Ponum,k,1);
          AD dPo_dz = sol_grad(e,Ponum,k,2);
          AD dNo_dx = sol_grad(e,Nonum,k,0);
          AD dNo_dy = sol_grad(e,Nonum,k,1);
          AD dNo_dz = sol_grad(e,Nonum,k,2);
          AD dNo_dt = sol_dot(e,Nonum,k,0);
          AD dPo_dt = sol_dot(e,Ponum,k,0);

          AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
          AD So = No / rhoo;
          AD Sw = 1.0 - So;
          //AD dPo_dt = densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dt);
          AD dSw_dt = -1.0/(rhoo*rhoo)*(dNo_dt*rhoo - No*dPo_dt);
          AD Pw = Po - cp(e,k);
          AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
          AD dPw_dt = dPo_dt - dcp(e,k)*dSw_dt;
          AD drhow_dt = densref_w(e,k)*(0.0+comp_w(e,k)*dPw_dt);
          AD dNw_dt = Sw*drhow_dt + dSw_dt*rhow;
          AD dSw_dx = -1.0/(rhoo*rhoo)*(rhoo*dNo_dx - No*densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dx));
          AD dPw_dx = dPo_dx - dcp(e,k)*dSw_dx;
          AD dSw_dy = -1.0/(rhoo*rhoo)*(rhoo*dNo_dy - No*densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dy));
          AD dPw_dy = dPo_dy - dcp(e,k)*dSw_dy;
          
          for (int i=0; i<basis.dimension(1); i++ ) { // loop over basis functions
            
            // No equation
            resindex = offsets(Ponum,i);
            
            res(e,resindex) += porosity(e,k)*dNo_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo*(dPo_dx*basis_grad(e,i,k,0) +
                                                            dPo_dy*basis_grad(e,i,k,1)) // diffusion terms
            -source_o(e,k)*basis(e,i,k);
            
            // Po equation
            resindex = offsets(Nonum,i);
            
            res(e,resindex) += porosity(e,k)*dNw_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow*(dPw_dx*basis_grad(e,i,k,0) +
                                                            dPw_dy*basis_grad(e,i,k,1)) // diffusion terms
            -source_w(e,k)*basis(e,i,k);
          }
        }
      });
    }
    else if (spaceDim == 3) {
      parallel_for(RangePolicy<AssemblyDevice>(0,res.dimension(0)), KOKKOS_LAMBDA (const int e ) {
        for (int k=0; k<sol.dimension(2); k++ ) {
          AD Po = sol(e,Ponum,k,0);
          AD No = sol(e,Nonum,k,0);
          AD dPo_dx = sol_grad(e,Ponum,k,0);
          AD dPo_dy = sol_grad(e,Ponum,k,1);
          AD dPo_dz = sol_grad(e,Ponum,k,2);
          AD dNo_dx = sol_grad(e,Nonum,k,0);
          AD dNo_dy = sol_grad(e,Nonum,k,1);
          AD dNo_dz = sol_grad(e,Nonum,k,2);
          AD dNo_dt = sol_dot(e,Nonum,k,0);
          AD dPo_dt = sol_dot(e,Ponum,k,0);

          AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
          AD So = No / rhoo;
          AD Sw = 1.0 - So;
          //AD dPo_dt = densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dt);
          AD dSw_dt = -1.0/(rhoo*rhoo)*(dNo_dt*rhoo - No*dPo_dt);
          AD Pw = Po - cp(e,k);
          AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
          AD dPw_dt = dPo_dt - dcp(e,k)*dSw_dt;
          AD drhow_dt = densref_w(e,k)*(0.0+comp_w(e,k)*dPw_dt);
          AD dNw_dt = Sw*drhow_dt + dSw_dt*rhow;
          AD dSw_dx = -1.0/(rhoo*rhoo)*(rhoo*dNo_dx - No*densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dx));
          AD dPw_dx = dPo_dx - dcp(e,k)*dSw_dx;
          AD dSw_dy = -1.0/(rhoo*rhoo)*(rhoo*dNo_dy - No*densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dy));
          AD dPw_dy = dPo_dy - dcp(e,k)*dSw_dy;
          AD dSw_dz = -1.0/(rhoo*rhoo)*(rhoo*dNo_dz - No*densref_o(e,k)*(0.0+comp_o(e,k)*dPo_dz));
          AD dPw_dz = dPo_dz - dcp(e,k)*dSw_dz;
          
          for (int i=0; i<basis.dimension(1); i++ ) { // loop over basis functions
            
            // Po equation
            resindex = offsets(Nonum,i);
            
            res(e,resindex) += porosity(e,k)*dNo_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo*(dPo_dx*basis_grad(e,i,k,0) +
                                                            dPo_dy*basis_grad(e,i,k,1) +
                                                            dPo_dz*basis_grad(e,i,k,2) -
                                                            rhoo*gravity(e,k)*1.0) // diffusion terms
            -source_o(e,k)*basis(e,i,k);
            
            // Pw equation
            resindex = offsets(Ponum,i);
            
            res(e,resindex) += porosity(e,k)*dNw_dt*basis(e,i,k) + // transient term
            perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow*(dPw_dx*basis_grad(e,i,k,0) +
                                                            dPw_dy*basis_grad(e,i,k,1) +
                                                            dPw_dz*basis_grad(e,i,k,2) -
                                                            rhow*gravity(e,k)*1.0) // diffusion terms
            -source_w(e,k)*basis(e,i,k);
            
          }
        }
      });
    }
    
  }
  
  
  // ========================================================================================
  // ========================================================================================
  
  void boundaryResidual() {
    
    sideinfo = wkset->sideinfo;
    Kokkos::View<int**,AssemblyDevice> bcs = wkset->var_bcs;
    
    int cside = wkset->currentside;
    int sidetype = bcs(Ponum,cside); // TMW to do: allow No to use different BCs
    
    int basis_num = wkset->usebasis[Ponum];
    int numBasis = wkset->basis_side[basis_num].dimension(1);
    basis = wkset->basis_side[basis_num];
    basis_grad = wkset->basis_grad_side[basis_num];
    
    {
      Teuchos::TimeMonitor localtime(*boundaryResidualFunc);
      
      if (sidetype == 4 ) {
        source_w = functionManager->evaluate("Dirichlet Pw " + wkset->sidename,"side ip",blocknum);
        source_o = functionManager->evaluate("Dirichlet No " + wkset->sidename,"side ip",blocknum);
      }
      else if (sidetype == 2) {
        source_w = functionManager->evaluate("Neumann Pw " + wkset->sidename,"side ip",blocknum);
        source_o = functionManager->evaluate("Neumann No " + wkset->sidename,"side ip",blocknum);
      }
      
      perm = functionManager->evaluate("permeability","side ip",blocknum);
      relperm_o = functionManager->evaluate("relative permeability oil","side ip",blocknum);
      source_o = functionManager->evaluate("source oil","side ip",blocknum);
      viscosity_o = functionManager->evaluate("viscosity oil","side ip",blocknum);
      densref_o = functionManager->evaluate("reference density oil","side ip",blocknum);
      pref_o = functionManager->evaluate("reference pressure oil","side ip",blocknum);
      comp_o = functionManager->evaluate("compressibility oil","side ip",blocknum);
      
      relperm_w = functionManager->evaluate("relative permeability water","side ip",blocknum);
      source_w = functionManager->evaluate("source water","side ip",blocknum);
      viscosity_w = functionManager->evaluate("viscosity water","side ip",blocknum);
      densref_w = functionManager->evaluate("reference density water","side ip",blocknum);
      pref_w = functionManager->evaluate("reference pressure water","side ip",blocknum);
      comp_w = functionManager->evaluate("compressibility water","side ip",blocknum);
      
      gravity = functionManager->evaluate("gravity","side ip",blocknum);
      cp = functionManager->evaluate("cap press","side ip",blocknum);
      dcp = functionManager->evaluate("dcap press","side ip",blocknum);
      
    }
    
    ScalarT sf = formparam;
    if (wkset->isAdjoint) {
      sf = 1.0;
      adjrhs = wkset->adjrhs;
    }
    
    // Since normals get recomputed often, this needs to be reset
    normals = wkset->normals;
    
    Teuchos::TimeMonitor localtime(*boundaryResidualFill);
    
    ScalarT v = 0.0;
    ScalarT dvdx = 0.0;
    ScalarT dvdy = 0.0;
    ScalarT dvdz = 0.0;
    
    for (int e=0; e<basis.dimension(0); e++) {
      if (bcs(Ponum,cside) == 2) {
        for (int k=0; k<basis.dimension(2); k++ ) {
          for (int i=0; i<basis.dimension(1); i++ ) {
            resindex = offsets(Ponum,i);
            res(e,resindex) += -source_o(e,k)*basis(e,i,k);
            
            resindex = offsets(Nonum,i); // index into Pw eqn
            res(e,resindex) += -source_w(e,k)*basis(e,i,k);
          }
        }
      }
      
      if (bcs(Ponum,cside) == 4 || bcs(Ponum,cside) == 5) {
        
        for (int k=0; k<basis.dimension(2); k++ ) {
          
          AD lambda_Po, lambda_Pw;
          
          if (bcs(Ponum,cside) == 5) {
            lambda_Po = aux_side(e,Ponum,k);
            AD lambda_No = aux_side(e,Nonum,k);
            //AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
            //AD So = No / rhoo;
            //AD Sw = 1.0 - So;
            AD cp_val = 0.0; // TMW: fix this
            lambda_Pw = lambda_Po - cp_val; // TMW:need to evaluate cp curve using this Sw
            
          }
          else {
            lambda_Po = source_o(e,k);
            lambda_Pw = source_w(e,k);
          }
          
          AD Po = sol_side(e,Ponum,k,0);
          AD No = sol_side(e,Nonum,k,0);
          AD dPo_dx = sol_grad_side(e,Ponum,k,0);
          AD dPo_dy = sol_grad_side(e,Ponum,k,1);
          AD dPo_dz = sol_grad_side(e,Ponum,k,2);
          AD dNo_dx = sol_grad_side(e,Nonum,k,0);
          AD dNo_dy = sol_grad_side(e,Nonum,k,1);
          AD dNo_dz = sol_grad_side(e,Nonum,k,2);
          
          AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
          //AD So = No / rhoo;
          //AD Sw = 1.0 - So;
          //AD dP_o_dt = densref_o(e,k)*comp_o(e,k)*sol_dot(e,Ponum,k,0);
          //AD dS_w_dt = -1.0/(rho_o*rho_o)*(sol_dot(e,Nonum,k,0)*rho_o - sol(e,Nonum,k,0)*dP_o_dt);
          AD Pw = Po - cp(e,k);
          AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
          //AD dPw_dt = sol_dot(e,Ponum,k,0) - dcp(e,k)*dS_w_dt;
          //AD drhow_dt = densref_w(e,k)*comp_w(e,k)*dP_w_dt;
          //AD dNw_dt = S_w*drho_w_dt + dS_w_dt*rho_w;
          
          AD dSw_dx = -1.0/(rhoo*rhoo)*(rhoo*dNo_dx - No*densref_o(e,k)*comp_o(e,k)*dPo_dx);
          AD dPw_dx = dPo_dx - dcp(e,k)*dSw_dx;
          AD dSw_dy = -1.0/(rhoo*rhoo)*(rhoo*dNo_dy - No*densref_o(e,k)*comp_o(e,k)*dPo_dy);
          AD dPw_dy = dPo_dy - dcp(e,k)*dSw_dy;
          AD dSw_dz = -1.0/(rhoo*rhoo)*(rhoo*dNo_dz - No*densref_o(e,k)*comp_o(e,k)*dPo_dz);
          AD dPw_dz = dPo_dz - dcp(e,k)*dSw_dz;
          
          
          for (int i=0; i<basis.dimension(1); i++ ) {
            v = basis(e,i,k);
            dvdx = basis_grad(e,i,k,0);
            if (spaceDim > 1)
              dvdy = basis_grad(e,i,k,1);
            if (spaceDim > 2)
              dvdz = basis_grad(e,i,k,2);
            
            // Po eqn
            AD Kval = perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo;
            AD weakDiriScale = 10.0*Kval/wkset->h(e);
            
            resindex = offsets(Ponum,i);
            
            res(e,resindex) += -Kval*dPo_dx*normals(e,k,0)*v - sf*Kval*dvdx*normals(e,k,0)*(Po-lambda_Po) + weakDiriScale*(Po-lambda_Po)*v;
            if (spaceDim > 1) {
              res(e,resindex) += -Kval*dPo_dy*normals(e,k,1)*v - sf*Kval*dvdy*normals(e,k,1)*(Po-lambda_Po);
            }
            if (spaceDim > 2) {
              res(e,resindex) += -Kval*(dPo_dz - rhow*gravity(e,k)*rhoo)*normals(e,k,2)*v - sf*Kval*dvdz*normals(e,k,2)*(Po-lambda_Po);
            }
            
            // Pw eqn
            Kval = perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow;
            weakDiriScale = 10.0*Kval/wkset->h(e);
            
            resindex = offsets(Nonum,i);
            
            res(e,resindex) += -Kval*dPw_dx*normals(e,k,0)*v - sf*Kval*dvdx*normals(e,k,0)*(Pw-lambda_Pw) + weakDiriScale*(Pw-lambda_Pw)*v;
            if (spaceDim > 1) {
              res(e,resindex) += -Kval*dPw_dy*normals(e,k,1)*v - sf*Kval*dvdy*normals(e,k,1)*(Pw-lambda_Pw);
            }
            if (spaceDim > 2) {
              res(e,resindex) += -Kval*(dPw_dz - rhow*gravity(e,k)*rhow)*normals(e,k,2)*v - sf*Kval*dvdz*normals(e,k,2)*(Pw-lambda_Pw);
            }
            
            //if (wkset->isAdjoint) {
            //  adjrhs(e,resindex) += sf*diff_side(e,k)*dvdx*normals(e,k,0)*lambda - weakDiriScale*lambda*v;
            //  if (spaceDim > 1)
            //  adjrhs(e,resindex) += sf*diff_side(e,k)*dvdy*normals(e,k,1)*lambda;
            //  if (spaceDim > 2)
            //  adjrhs(e,resindex) += sf*diff_side(e,k)*dvdz*normals(e,k,2)*lambda;
            //}
          }
          
        }
      }
    }
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void edgeResidual() {
    
  }
  
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
      perm = functionManager->evaluate("permeability","side ip",blocknum);
      relperm_o = functionManager->evaluate("relative permeability oil","side ip",blocknum);
      source_o = functionManager->evaluate("source oil","side ip",blocknum);
      viscosity_o = functionManager->evaluate("viscosity oil","side ip",blocknum);
      densref_o = functionManager->evaluate("reference density oil","side ip",blocknum);
      pref_o = functionManager->evaluate("reference pressure oil","side ip",blocknum);
      comp_o = functionManager->evaluate("compressibility oil","side ip",blocknum);
      
      relperm_w = functionManager->evaluate("relative permeability water","side ip",blocknum);
      source_w = functionManager->evaluate("source water","side ip",blocknum);
      viscosity_w = functionManager->evaluate("viscosity water","side ip",blocknum);
      densref_w = functionManager->evaluate("reference density water","side ip",blocknum);
      pref_w = functionManager->evaluate("reference pressure water","side ip",blocknum);
      comp_w = functionManager->evaluate("compressibility water","side ip",blocknum);
      
      gravity = functionManager->evaluate("gravity","side ip",blocknum);
      cp = functionManager->evaluate("cap press","side ip",blocknum);
      dcp = functionManager->evaluate("dcap press","side ip",blocknum);
      
    }
    
    // Since normals get recomputed often, this needs to be reset
    normals = wkset->normals;
    
    {
      Teuchos::TimeMonitor localtime(*fluxFill);
      
      for (int e=0; e<numElem; e++) {
        
        for (size_t k=0; k<wkset->ip_side.dimension(1); k++) {
          AD lambda_Po, lambda_Pw;
          
          lambda_Po = aux_side(e,Ponum,k);
          //lambda_No = aux_side(e,Nonum,k);
          //AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
          //AD So = No / rhoo;
          //AD Sw = 1.0 - So;
          AD cp_val = 0.0; //TMW: fix this
          lambda_Pw = lambda_Po - cp_val; // TMW:need to evaluate cp curve using this Sw
          
          AD Po = sol_side(e,Ponum,k,0);
          AD No = sol_side(e,Nonum,k,0);
          AD dPo_dx = sol_grad_side(e,Ponum,k,0);
          AD dPo_dy = sol_grad_side(e,Ponum,k,1);
          AD dPo_dz = sol_grad_side(e,Ponum,k,2);
          AD dNo_dx = sol_grad_side(e,Nonum,k,0);
          AD dNo_dy = sol_grad_side(e,Nonum,k,1);
          AD dNo_dz = sol_grad_side(e,Nonum,k,2);
          
          AD rhoo = densref_o(e,k)*(1.0+comp_o(e,k)*(Po - pref_o(e,k)));
          //AD So = No / rhoo;
          //AD Sw = 1.0 - So;
          //AD dP_o_dt = densref_o(e,k)*comp_o(e,k)*sol_dot(e,Ponum,k,0);
          //AD dS_w_dt = -1.0/(rho_o*rho_o)*(sol_dot(e,Nonum,k,0)*rho_o - sol(e,Nonum,k,0)*dP_o_dt);
          AD Pw = Po - cp(e,k);
          AD rhow = densref_w(e,k)*(1.0+comp_w(e,k)*(Pw - pref_w(e,k)));
          //AD dPw_dt = sol_dot(e,Ponum,k,0) - dcp(e,k)*dS_w_dt;
          //AD drhow_dt = densref_w(e,k)*comp_w(e,k)*dP_w_dt;
          //AD dNw_dt = S_w*drho_w_dt + dS_w_dt*rho_w;
          
          AD dSw_dx = -1.0/(rhoo*rhoo)*(rhoo*dNo_dx - No*densref_o(e,k)*comp_o(e,k)*dPo_dx);
          AD dPw_dx = dPo_dx - dcp(e,k)*dSw_dx;
          AD dSw_dy = -1.0/(rhoo*rhoo)*(rhoo*dNo_dy - No*densref_o(e,k)*comp_o(e,k)*dPo_dy);
          AD dPw_dy = dPo_dy - dcp(e,k)*dSw_dy;
          AD dSw_dz = -1.0/(rhoo*rhoo)*(rhoo*dNo_dz - No*densref_o(e,k)*comp_o(e,k)*dPo_dz);
          AD dPw_dz = dPo_dz - dcp(e,k)*dSw_dz;
          
          AD Kval_o = perm(e,k)*relperm_o(e,k)/viscosity_o(e,k)*rhoo;
          AD penalty_o = 10.0*Kval_o/wkset->h(e);
          AD Kval_w = perm(e,k)*relperm_w(e,k)/viscosity_w(e,k)*rhow;
          AD penalty_w = 10.0*Kval_w/wkset->h(e);
          
          flux(e,Ponum,k) += sf*Kval_o*dPo_dx*normals(e,k,0) + penalty_o*(lambda_Po-Po);
          flux(e,Nonum,k) += sf*Kval_w*dPw_dx*normals(e,k,0) + penalty_w*(lambda_Pw-Pw);
          if (spaceDim > 1) {
            flux(e,Ponum,k) += sf*Kval_o*dPo_dy*normals(e,k,1);
            flux(e,Nonum,k) += sf*Kval_w*dPw_dy*normals(e,k,1);
          }
          if (spaceDim > 2) {
            flux(e,Ponum,k) += sf*Kval_o*(dPo_dz - gravity(e,k)*rhoo)*normals(e,k,2);
            flux(e,Nonum,k) += sf*Kval_w*(dPw_dz - gravity(e,k)*rhow)*normals(e,k,2);
          }
        }
      }
    }
    
  }
  
  // ========================================================================================
  // ========================================================================================
  
  void setVars(std::vector<string> & varlist_) {
    varlist = varlist_;
    for (size_t i=0; i<varlist.size(); i++) {
      if (varlist[i] == "Po") {
        Ponum = i;
      }
      if (varlist[i] == "No") {
        Nonum = i;
      }
    }
  }
  
private:
  
  Teuchos::RCP<FunctionInterface> functionManager;
  
  int spaceDim, numElem, blocknum;
  size_t numip, numip_side;
  ScalarT formparam;
  int Ponum, Nonum, resindex;
  
  vector<string> varlist;
  
  FDATA perm, porosity, gravity, cp, dcp;
  FDATA relperm_o, source_o, viscosity_o, densref_o, pref_o, comp_o;
  FDATA relperm_w, source_w, viscosity_w, densref_w, pref_w, comp_w;
  Kokkos::View<int****,AssemblyDevice> sideinfo;
  
  Teuchos::RCP<Teuchos::Time> volumeResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::volumeResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> volumeResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::volumeResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::boundaryResidual() - function evaluation");
  Teuchos::RCP<Teuchos::Time> boundaryResidualFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::boundaryResidual() - evaluation of residual");
  Teuchos::RCP<Teuchos::Time> fluxFunc = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::computeFlux() - function evaluation");
  Teuchos::RCP<Teuchos::Time> fluxFill = Teuchos::TimeMonitor::getNewCounter("MILO::porous2p::computeFlux() - evaluation of flux");
  
};

#endif
