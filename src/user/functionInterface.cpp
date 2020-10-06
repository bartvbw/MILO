/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "functionInterface.hpp"
#include "interpreter.hpp"

FunctionInterface::FunctionInterface() {
  known_vars = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops = {"sin","cos","exp","log","tan","abs","max","min","mean"};
  verbosity = 0;
}


FunctionInterface::FunctionInterface(Teuchos::RCP<Teuchos::ParameterList> & settings) {
  known_vars = {"x","y","z","t","nx","ny","nz","pi","h"};
  known_ops = {"sin","cos","exp","log","tan","abs","max","min","mean"};
  verbosity = settings->get<int>("verbosity",0);
}

//////////////////////////////////////////////////////////////////////////////////////
// Add a user defined function
//////////////////////////////////////////////////////////////////////////////////////

int FunctionInterface::addFunction(const string & fname, const string & expression,
                                   const size_t & dim0, const size_t & dim1,
                                   const string & location, const size_t & blocknum) {
  bool found = false;
  int findex = 0;
  
  if (functions.size() <= blocknum) {
    vector<function_class> blockfun;
    functions.push_back(blockfun);
  }
  for (size_t k=0; k<functions[blocknum].size(); k++) {
    if (functions[blocknum][k].function_name == fname && functions[blocknum][k].location == location) {
      found = true;
      findex = k;
    }
  }
  if (!found) {
    functions[blocknum].push_back(function_class(fname, expression, dim0, dim1, location));
    findex = functions[blocknum].size()-1;
  }
  return findex;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Set the lists of variables, parameters and discretized parameters
//////////////////////////////////////////////////////////////////////////////////////

void FunctionInterface::setupLists(const vector<string> & variables_,
                                   const vector<string> & parameters_,
                                   const vector<string> & disc_parameters_) {
  variables = variables_;
  parameters = parameters_;
  disc_parameters = disc_parameters_;
}

//////////////////////////////////////////////////////////////////////////////////////
// Validate all of the functions
//////////////////////////////////////////////////////////////////////////////////////

void FunctionInterface::validateFunctions(){
  for (size_t b=0; b<functions.size(); b++) {
    vector<string> function_names;
    for (size_t k=0; k<functions[b].size(); k++) {
      function_names.push_back(functions[b][k].function_name);
    }
    for (size_t k=0; k<functions[b].size(); k++) {
      vector<string> vars = getVars(functions[b][k].expression, known_ops);
      
      int numfails = validateTerms(vars,known_vars,variables,parameters,disc_parameters,function_names);
      if (numfails > 0) {
        TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"Error: MILO could not identify one or more terms in: " + functions[b][k].function_name);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Decompose the functions into terms and set the evaluation tree
// Also sets up the Kokkos::Views (subviews) to the data for all of the terms
//////////////////////////////////////////////////////////////////////////////////////

void FunctionInterface::decomposeFunctions() {
  
  Teuchos::TimeMonitor ttimer(*decomposeTimer);
  
  for (size_t b=0; b<functions.size(); b++) {
    for (size_t fiter=0; fiter<functions[b].size(); fiter++) {
      
      bool done = false; // will turn to "true" when the function is fully decomposed
      int maxiter = 20; // maximum number of recursions
      int iter = 0;
      
      while (!done && iter < maxiter) {
        
        iter++;
        size_t Nterms = functions[b][fiter].terms.size();
        
        for (size_t k=0; k<Nterms; k++) {
          
          // HAVE WE ALREADY LOOKED AT THIS TERM?
          bool decompose = true;
          if (functions[b][fiter].terms[k].isRoot || functions[b][fiter].terms[k].beenDecomposed) {
            decompose = false;
          }
          
          // IS THE TERM ONE OF THE KNOWN VARIABLES: x,y,z,t
          if (decompose) {
            for (size_t j=0; j<known_vars.size(); j++) {
              if (functions[b][fiter].terms[k].expression == known_vars[j]) {
                decompose = false;
                bool have_data = false;
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                functions[b][fiter].terms[k].isAD = false;
                if (known_vars[j] == "x") {
                  if (functions[b][fiter].location == "side ip") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->ip_side_KV, Kokkos::ALL(), Kokkos::ALL(), 0);
                  }
                  else if (functions[b][fiter].location == "point") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->point_KV, Kokkos::ALL(), Kokkos::ALL(), 0);
                  }
                  else {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->ip_KV, Kokkos::ALL(), Kokkos::ALL(), 0);
                  }
                }
                else if (known_vars[j] == "y") {
                  if (functions[b][fiter].location == "side ip") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->ip_side_KV, Kokkos::ALL(), Kokkos::ALL(), 1);
                  }
                  else if (functions[b][fiter].location == "point") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->point_KV, Kokkos::ALL(), Kokkos::ALL(), 1);
                  }
                  else {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->ip_KV, Kokkos::ALL(), Kokkos::ALL(), 1);
                  }
                }
                else if (known_vars[j] == "z") {
                  if (functions[b][fiter].location == "side ip") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->ip_side_KV, Kokkos::ALL(), Kokkos::ALL(), 2);
                  }
                  else if (functions[b][fiter].location == "point") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->point_KV, Kokkos::ALL(), Kokkos::ALL(), 2);
                  }
                  else {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->ip_KV, Kokkos::ALL(), Kokkos::ALL(), 2);
                  }
                }
                else if (known_vars[j] == "t") {
                  //functions[b][fiter].terms[k].scalar_ddata = Kokkos::subview(wkset->time_KV, Kokkos::ALL(), 0);
                  functions[b][fiter].terms[k].scalar_ddata = wkset->time_KV;
                  functions[b][fiter].terms[k].isScalar = true;
                  functions[b][fiter].terms[k].isConstant = false;
                  Kokkos::View<double***,AssemblyDevice> tdata("data",functions[b][fiter].dim0,functions[b][fiter].dim1,1);
                  functions[b][fiter].terms[k].ddata = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
                  
                }
                else if (known_vars[j] == "nx") {
                  if (functions[b][fiter].location == "side ip") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->normals_KV, Kokkos::ALL(), Kokkos::ALL(), 0);
                  }
                  else {
                    //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: normals can only be used in functions defined on boundaries or faces");
                  }
                }
                else if (known_vars[j] == "ny") {
                  if (functions[b][fiter].location == "side ip") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->normals_KV, Kokkos::ALL(), Kokkos::ALL(), 1);
                  }
                  else {
                    //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: normals can only be used in functions defined on boundaries or faces");
                  }
                }
                else if (known_vars[j] == "nz") {
                  if (functions[b][fiter].location == "side ip") {
                    functions[b][fiter].terms[k].ddata = Kokkos::subview(wkset->normals_KV, Kokkos::ALL(), Kokkos::ALL(), 2);
                  }
                  else {
                    //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: normals can only be used in functions defined on boundaries or faces");
                  }
                }
                else if (known_vars[j] == "pi") {
                  functions[b][fiter].terms[k].isRoot = true;
                  functions[b][fiter].terms[k].isAD = false;
                  functions[b][fiter].terms[k].beenDecomposed = true;
                  functions[b][fiter].terms[k].isScalar = true;
                  functions[b][fiter].terms[k].isConstant = true; // means in does not need to be copied every time
                  have_data = true;
                  // Copy the data just once
                  Kokkos::View<double***,AssemblyDevice> tdata("scalar data",functions[b][fiter].dim0,functions[b][fiter].dim1,1);
                  functions[b][fiter].terms[k].ddata = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
                  for (size_t k2=0; k2<functions[b][fiter].dim0; k2++) {
                    for (size_t j2=0; j2<functions[b][fiter].dim1; j2++) {
                      functions[b][fiter].terms[k].ddata(k2,j2) = PI;
                    }
                  }
                  decompose = false;
                }
              }
            }
          } // end known_vars
          
          // IS THIS TERM ONE OF THE KNOWN OPERATORS: sin(...), exp(...), etc.
          if (decompose) {
            bool isop = isOperator(functions[b][fiter].terms, k, known_ops);
            // isOperator takes care of the decomposition if it is of this form
            if (isop) {
              decompose = false;
            }
          }
          
          // IS IT ONE OF THE VARIABLES (
          if (decompose) {
            for (int j=0; j<variables.size(); j++) {
              if (functions[b][fiter].terms[k].expression == variables[j]) { // just scalar variables
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                if (functions[b][fiter].location == "side ip") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_side, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
                else if (functions[b][fiter].location == "point") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_point, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
                else {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
              }
              else if (functions[b][fiter].terms[k].expression == (variables[j]+"_x")) { // deriv. of scalar var. w.r.t x
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                if (functions[b][fiter].location == "side ip") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
                else if (functions[b][fiter].location == "point") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_point, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
                else {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
              }
              else if (functions[b][fiter].terms[k].expression == (variables[j]+"_y")) { // deriv. of scalar var. w.r.t y
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                if (functions[b][fiter].location == "side ip") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 1);
                }
                else if (functions[b][fiter].location == "point") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_point, Kokkos::ALL(), j, Kokkos::ALL(), 1);
                }
                else {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 1);
                }
              }
              else if (functions[b][fiter].terms[k].expression == (variables[j]+"_z")) { // deriv. of scalar var. w.r.t z
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                if (functions[b][fiter].location == "side ip") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 2);
                }
                else if (functions[b][fiter].location == "point") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_point, Kokkos::ALL(), j, Kokkos::ALL(), 2);
                }
                else {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 2);
                }
              }
              else if (functions[b][fiter].terms[k].expression == (variables[j]+"_t")) { // deriv. of scalar var. w.r.t x
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                if (functions[b][fiter].location == "side ip" || functions[b][fiter].location == "point") {
                  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MILO currently does not support the time derivative of a variable on boundaries or point evaluation points.");
                }
                else {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_dot, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
              }
              else if (functions[b][fiter].terms[k].expression == (variables[j]+"[x]")) { // x-component of vector scalar var.
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                if (functions[b][fiter].location == "side ip") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
                else { // TMW: NOT UPDATED FOR point
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 0);
                }
              }
              else if (functions[b][fiter].terms[k].expression == (variables[j]+"[y]")) { // y-component of vector scalar var.
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                if (functions[b][fiter].location == "side ip") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 1);
                }
                else { // TMW: NOT UPDATED FOR point
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 1);
                }
              }
              else if (functions[b][fiter].terms[k].expression == (variables[j]+"[z]")) { // z-component of vector scalar var.
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                if (functions[b][fiter].location == "side ip") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad_side, Kokkos::ALL(), j, Kokkos::ALL(), 2);
                }
                else { // TMW: NOT UPDATED FOR point
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_soln_grad, Kokkos::ALL(), j, Kokkos::ALL(), 2);
                }
              }
              
            }
          }
          
          // IS THE TERM A SIMPLE SCALAR: 2.03, 1.0E2, etc.
          if (decompose) {
            bool isnum = isScalar(functions[b][fiter].terms[k].expression);
            if (isnum) {
              functions[b][fiter].terms[k].isRoot = true;
              functions[b][fiter].terms[k].isAD = false;
              functions[b][fiter].terms[k].beenDecomposed = true;
              functions[b][fiter].terms[k].isScalar = true;
              functions[b][fiter].terms[k].isConstant = true; // means in does not need to be copied every time
              functions[b][fiter].terms[k].scalar_ddata = Kokkos::View<double*,AssemblyDevice>("scalar double data",1);
              functions[b][fiter].terms[k].scalar_ddata(0) = std::stod(functions[b][fiter].terms[k].expression);
              
              // Copy the data just once
              Kokkos::View<double***,AssemblyDevice> tdata("scalar data",functions[b][fiter].dim0,functions[b][fiter].dim1,1);
              functions[b][fiter].terms[k].ddata = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
              for (size_t k2=0; k2<functions[b][fiter].dim0; k2++) {
                for (size_t j2=0; j2<functions[b][fiter].dim1; j2++) {
                  functions[b][fiter].terms[k].ddata(k2,j2) = functions[b][fiter].terms[k].scalar_ddata(0);
                }
              }
              decompose = false;
            }
          }
          
          // check if it is a discretized parameter
          if (decompose) { // TMW: NOT UPDATED FOR PARAM GRAD
            
            for (int j=0; j<disc_parameters.size(); j++) {
              if (functions[b][fiter].terms[k].expression == disc_parameters[j]) {
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                decompose = false;
                
                if (functions[b][fiter].location == "side ip") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_param_side, Kokkos::ALL(), j, Kokkos::ALL());
                }
                else if (functions[b][fiter].location == "point") {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_param_point, Kokkos::ALL(), j, Kokkos::ALL());
                }
                else {
                  functions[b][fiter].terms[k].data = Kokkos::subview(wkset->local_param, Kokkos::ALL(), j, Kokkos::ALL());
                }
              }
            }
          }

          
          // check if it is a parameter
          if (decompose) {
            
            for (int j=0; j<parameters.size(); j++) {
              
              if (functions[b][fiter].terms[k].expression == parameters[j]) {
                functions[b][fiter].terms[k].isRoot = true;
                functions[b][fiter].terms[k].isAD = true;
                functions[b][fiter].terms[k].beenDecomposed = true;
                functions[b][fiter].terms[k].isScalar = true;
                functions[b][fiter].terms[k].isConstant = false; // needs to be copied
                functions[b][fiter].terms[k].scalarIndex = 0;
                
                decompose = false;
                
                functions[b][fiter].terms[k].scalar_data = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                
                Kokkos::View<AD***,AssemblyDevice> tdata("scalar data",functions[b][fiter].dim0,functions[b][fiter].dim1,1);
                functions[b][fiter].terms[k].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
                
              }
              else { // look for param(*) or param(**)
                bool found = true;
                int sindex = 0;
                size_t nexp = functions[b][fiter].terms[k].expression.length();
                if (nexp == parameters[j].length()+3) {
                  for (size_t n=0; n<parameters[j].length(); n++) {
                    if (functions[b][fiter].terms[k].expression[n] != parameters[j][n]) {
                      found = false;
                    }
                  }
                  if (found) {
                    if (functions[b][fiter].terms[k].expression[nexp-3] == '(' && functions[b][fiter].terms[k].expression[nexp-1] == ')') {
                      string check = "";
                      check += functions[b][fiter].terms[k].expression[nexp-2];
                      if (isdigit(check[0])) {
                        sindex = std::stoi(check);
                      }
                      else {
                        found = false;
                      }
                    }
                    else {
                      found = false;
                    }
                  }
                }
                else if (nexp == parameters[j].length()+4) {
                  for (size_t n=0; n<parameters[j].length(); n++) {
                    if (functions[b][fiter].terms[k].expression[n] != parameters[j][n]) {
                      found = false;
                    }
                  }
                  if (found) {
                    if (functions[b][fiter].terms[k].expression[nexp-4] == '(' && functions[b][fiter].terms[k].expression[nexp-1] == ')') {
                      string check = "";
                      check += functions[b][fiter].terms[k].expression[nexp-3];
                      check += functions[b][fiter].terms[k].expression[nexp-2];
                      if (isdigit(check[0]) && isdigit(check[1])) {
                        sindex = std::stoi(check);
                      }
                      else {
                        found = false;
                      }
                    }
                    else {
                      found = false;
                    }
                  }
                }
                else {
                  found = false;
                }
                
                if (found) {
                  functions[b][fiter].terms[k].isRoot = true;
                  functions[b][fiter].terms[k].isAD = true;
                  functions[b][fiter].terms[k].beenDecomposed = true;
                  functions[b][fiter].terms[k].isScalar = true;
                  functions[b][fiter].terms[k].isConstant = false; // needs to be copied
                  functions[b][fiter].terms[k].scalarIndex = sindex;
                  
                  decompose = false;
                  
                  functions[b][fiter].terms[k].scalar_data = Kokkos::subview(wkset->params_AD, j, Kokkos::ALL());
                  
                  Kokkos::View<AD***,AssemblyDevice> tdata("scalar data",functions[b][fiter].dim0,functions[b][fiter].dim1,1);
                  functions[b][fiter].terms[k].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
                }
              }
            }
          }
          
          // check if it is a function
          if (decompose) {
            for (int j=0; j<functions[b].size(); j++) {
              if (functions[b][fiter].terms[k].expression == functions[b][j].function_name &&
                  functions[b][fiter].location == functions[b][j].location) {
                functions[b][fiter].terms[k].isFunc = true;
                functions[b][fiter].terms[k].isAD = functions[b][j].terms[0].isAD;
                functions[b][fiter].terms[k].funcIndex = j;
                functions[b][fiter].terms[k].beenDecomposed = true;
                functions[b][fiter].terms[k].data = functions[b][j].terms[0].data;
                functions[b][fiter].terms[k].ddata = functions[b][j].terms[0].ddata;
                decompose = false;
              }
            }
          }
          
          if (decompose) {
            int numterms = 0;
            numterms = split(functions[b][fiter].terms,k);
            functions[b][fiter].terms[k].beenDecomposed = true;
          }
        }
        
        bool isdone = true;
        for (size_t k=0; k<functions[b][fiter].terms.size(); k++) {
          if (!functions[b][fiter].terms[k].isRoot && !functions[b][fiter].terms[k].beenDecomposed) {
            isdone = false;
          }
        }
        done = isdone;
        
      }
      
      if (!done && iter >= maxiter) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: MILO reached the maximum number of recursive function calls for " + functions[b][fiter].function_name + ".  See functionInterface.hpp to increase this");
      }
    }
    
    // After all of the functions have been decomposed, we can determine if we need to use arrays of ScalarT or AD
    // Only the roots should be designated as ScalarT or AD at this point
    
    for (size_t k=0; k<functions[b].size(); k++) {
      for (size_t j=0; j<functions[b][k].terms.size(); j++) {
        bool termcheck = this->isScalarTerm(b,k,j); // is this term a ScalarT
        if (termcheck) {
          functions[b][k].terms[j].isAD = false;
          if (!functions[b][k].terms[j].isRoot) {
            Kokkos::View<double***,AssemblyDevice> tdata("data",functions[b][k].dim0,functions[b][k].dim1,1);
            functions[b][k].terms[j].ddata = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
          }
          if (j==0) { // always need this allocated
            Kokkos::View<AD***,AssemblyDevice> tdata("data",functions[b][k].dim0,functions[b][k].dim1,1);
            functions[b][k].terms[j].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
          }
        }
        else if (!functions[b][k].terms[j].isRoot) {
          functions[b][k].terms[j].isAD = true;
          Kokkos::View<AD***,AssemblyDevice> tdata("data",functions[b][k].dim0,functions[b][k].dim1,1);
          functions[b][k].terms[j].data = Kokkos::subview(tdata, Kokkos::ALL(), Kokkos::ALL(), 0);
        }
        //functions[k].terms[j].print();
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Determine if a term is a ScalarT or needs to be an AD type
//////////////////////////////////////////////////////////////////////////////////////

bool FunctionInterface::isScalarTerm(const size_t & block, const int & findex, const int & tindex) {
  bool is_scalar = true;
  if (functions[block][findex].terms[tindex].isRoot) {
    if (functions[block][findex].terms[tindex].isAD) {
      is_scalar = false;
    }
  }
  //else if (functions[block][findex].terms[tindex].isFunc) {
    //is_scalar = false;
  //}
  else {
    for (size_t k=0; k<functions[block][findex].terms[tindex].dep_list.size(); k++){
      bool depcheck = isScalarTerm(block, findex, functions[block][findex].terms[tindex].dep_list[k]);
      if (!depcheck) {
        is_scalar = false;
      }
    }
  }
  return is_scalar;
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function (probably will be deprecated)
//////////////////////////////////////////////////////////////////////////////////////

FDATA FunctionInterface::evaluate(const string & fname, const string & location,
                                  const size_t & block) {
  Teuchos::TimeMonitor ttimer(*evaluateTimer);
  
  if (verbosity > 10) {
    cout << endl;
    cout << "Evaluating: " << fname << " at " << location << endl;
  }
  
  int findex = -1;
  for (size_t i=0; i<functions[block].size(); i++) {
    if (fname == functions[block][i].function_name && functions[block][i].location == location) {
      evaluate(block,i,0);
      findex = i;
    }
  }
  
  if (verbosity > 10) {
    cout << "Finished evaluating: " << fname << " at " << location << endl;
  }
  
  if (findex == -1) { // meaning that the requested function was not registered at this location
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: function manager could not evaluate: " + fname + " at " + location);
  }
  
  
  if (!functions[block][findex].terms[0].isAD) {
    parallel_for(RangePolicy<AssemblyDevice>(0,functions[block][findex].dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<functions[block][findex].dim1; n++) {
        functions[block][findex].terms[0].data(e,n) = functions[block][findex].terms[0].ddata(e,n);
      }
    });
  }
  return functions[block][findex].terms[0].data;
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate a function
//////////////////////////////////////////////////////////////////////////////////////

void FunctionInterface::evaluate(const size_t & block, const size_t & findex, const size_t & tindex) {
  
  if (verbosity > 10) {
    cout << "------- Evaluating: " << functions[block][findex].terms[tindex].expression << endl;
  }
  
  //functions[block][findex].terms[tindex].print();
  
  if (functions[block][findex].terms[tindex].isRoot) {
    if (functions[block][findex].terms[tindex].isScalar && !functions[block][findex].terms[tindex].isConstant) {
      if (functions[block][findex].terms[tindex].isAD) {
        parallel_for(RangePolicy<AssemblyDevice>(0,functions[block][findex].dim0), KOKKOS_LAMBDA (const int e ) {
          for (int n=0; n<functions[block][findex].dim1; n++) {
            functions[block][findex].terms[tindex].data(e,n) = functions[block][findex].terms[tindex].scalar_data(0);
          }
        });
      }
      else {
        parallel_for(RangePolicy<AssemblyDevice>(0,functions[block][findex].dim0), KOKKOS_LAMBDA (const int e ) {
          for (int n=0; n<functions[block][findex].dim1; n++) {
            functions[block][findex].terms[tindex].ddata(e,n) = functions[block][findex].terms[tindex].scalar_ddata(0);
          }
        });
      }
    }
  }
  else if (functions[block][findex].terms[tindex].isFunc) {
    int funcIndex = functions[block][findex].terms[tindex].funcIndex;
    this->evaluate(block, funcIndex, 0);
    if (functions[block][findex].terms[tindex].isAD) {
      if (functions[block][funcIndex].terms[0].isAD) {
        functions[block][findex].terms[tindex].data = functions[block][funcIndex].terms[0].data;
      }
      else {
        parallel_for(RangePolicy<AssemblyDevice>(0,functions[block][findex].dim0), KOKKOS_LAMBDA (const int e ) {
          for (int n=0; n<functions[block][findex].dim1; n++) {
            functions[block][findex].terms[tindex].data(e,n) = functions[block][funcIndex].terms[0].ddata(e,n);
          }
        });
      }
    }
    else {
      functions[block][findex].terms[tindex].ddata = functions[block][funcIndex].terms[0].ddata;
    }
  }
  else {
    bool isAD = functions[block][findex].terms[tindex].isAD;
    for (size_t k=0; k<functions[block][findex].terms[tindex].dep_list.size(); k++) {
      
      int dep = functions[block][findex].terms[tindex].dep_list[k];
      this->evaluate(block, findex, dep);
      
      bool termisAD = functions[block][findex].terms[dep].isAD;
      if (isAD) {
        if (termisAD) {
          this->evaluateOp(functions[block][findex].terms[tindex].data,
                           functions[block][findex].terms[dep].data,
                           functions[block][findex].terms[tindex].dep_ops[k]);
          
        }
        else {
          this->evaluateOp(functions[block][findex].terms[tindex].data,
                           functions[block][findex].terms[dep].ddata,
                           functions[block][findex].terms[tindex].dep_ops[k]);
        }
      }
      else { // termisAD must also be false
        this->evaluateOp(functions[block][findex].terms[tindex].ddata,
                         functions[block][findex].terms[dep].ddata,
                         functions[block][findex].terms[tindex].dep_ops[k]);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////
// Evaluate an operator
//////////////////////////////////////////////////////////////////////////////////////

template<class T1, class T2>
void FunctionInterface::evaluateOp(T1 data, T2 tdata, const string & op) {
  size_t dim0 = std::min(data.extent(0),tdata.extent(0));
  size_t dim1 = std::min(data.extent(1),tdata.extent(1));
  
  if (op == "") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) = tdata(e,n);
      }
    });
  }
  else if (op == "plus") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) += tdata(e,n);
      }
    });
  }
  else if (op == "minus") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) += -tdata(e,n);
      }
    });
  }
  else if (op == "times") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) *= tdata(e,n);
      }
    });
  }
  else if (op == "divide") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) /= tdata(e,n);
      }
    });
  }
  else if (op == "power") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) = pow(data(e,n),tdata(e,n));
      }
    });
  }
  else if (op == "sin") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) = sin(tdata(e,n));
      }
    });
  }
  else if (op == "cos") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) = cos(tdata(e,n));
      }
    });
  }
  else if (op == "tan") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) = tan(tdata(e,n));
      }
    });
  }
  else if (op == "exp") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) = exp(tdata(e,n));
      }
    });
  }
  else if (op == "log") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        data(e,n) = log(tdata(e,n));
      }
    });
  }
  else if (op == "abs") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        if (tdata(e,n) < 0.0) {
          data(e,n) = -tdata(e,n);
        }
        else {
          data(e,n) = tdata(e,n);
        }
      }
    });
  }
  else if (op == "max") { // maximum over rows ... usually corr. to max over element/face at ip
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      data(e,0) = tdata(e,0);
      for (int n=0; n<dim1; n++) {
        if (tdata(e,n) > tdata(e,0)) {
          data(e,0) = tdata(e,n);
        }
      }
      for (int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "min") { // minimum over rows ... usually corr. to min over element/face at ip
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      data(e,0) = tdata(e,0);
      for (int n=0; n<dim1; n++) {
        if (tdata(e,n) < tdata(e,0)) {
          data(e,0) = tdata(e,n);
        }
      }
      for (int n=0; n<dim1; n++) { // copy min value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "mean") { // mean over rows ... usually corr. to mean over element/face
    double scale = (double)dim1;
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      data(e,0) = tdata(e,0)/scale;
      for (int n=0; n<dim1; n++) {
        data(e,0) += tdata(e,n)/scale;
      }
      for (int n=0; n<dim1; n++) { // copy max value at all ip
        data(e,n) = data(e,0);
      }
    });
  }
  else if (op == "lt") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        if (data(e,n) < tdata(e,n)) {
          data(e,n) = 1.0;
        }
        else {
          data(e,n) = 0.0;
        }
      }
    });
  }
  else if (op == "lte") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        if (data(e,n) <= tdata(e,n)) {
          data(e,n) = 1.0;
        }
        else {
          data(e,n) = 0.0;
        }
      }
    });
  }
  else if (op == "gt") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        if (data(e,n) > tdata(e,n)) {
          data(e,n) = 1.0;
        }
        else {
          data(e,n) = 0.0;
        }
      }
    });
  }
  else if (op == "gte") {
    parallel_for(RangePolicy<AssemblyDevice>(0,dim0), KOKKOS_LAMBDA (const int e ) {
      for (int n=0; n<dim1; n++) {
        if (data(e,n) >= tdata(e,n)) {
          data(e,n) = 1.0;
        }
        else {
          data(e,n) = 0.0;
        }
      }
    });
  }
  
}

//////////////////////////////////////////////////////////////////////////////////////
// Print out the function information (mostly for debugging)
//////////////////////////////////////////////////////////////////////////////////////

void FunctionInterface::printFunctions() {
  
  for (size_t b=0; b<functions.size(); b++) {
    cout << "Block Number: " << b << endl;
    for (size_t n=0; n<functions[b].size(); n++) {
      cout << "Function Name:" << functions[b][n].function_name << endl;
      cout << "Location: " << functions[b][n].location << endl << endl;
      cout << "Terms: " << endl;
      for (size_t t=0; t<functions[b][n].terms.size(); t++) {
        cout << "    " << functions[b][n].terms[t].expression << endl;
      }
      cout << endl;
      cout << "First term information:" << endl;
      functions[b][n].terms[0].print();
      cout << endl << endl;
    }
  }
  
}


