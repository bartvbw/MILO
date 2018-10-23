/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#ifndef USERINTERFACE_H
#define USERINTERFACE_H

#include "trilinos.hpp"
#include "preferences.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////
// Figure out if a file is .xml or .yaml (default)
//////////////////////////////////////////////////////////////////////////////////////////////

int getFileType(const string & filename)
{
  int type = -1;
  if(filename.find_last_of(".") != string::npos) {
    string extension = filename.substr(filename.find_last_of(".")+1);
    if (extension == "yaml") {
      type = 0;
    }
    else if (extension == "xml") {
      type = 1;
    }
    else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unrecognized file extension: " + filename);
    }
  }
  return type;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Function to print out help information
//////////////////////////////////////////////////////////////////////////////////////////////

void userHelp(const string & details) {
  cout << "********** Help and Documentation for the User Interface **********" << endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Standard constructor
//////////////////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<Teuchos::ParameterList> userInterface(const string & filename) {
  
  using Teuchos::RCP;
  using Teuchos::rcp;
  
  RCP<Teuchos::ParameterList> settings = rcp(new Teuchos::ParameterList("MILO"));
  
  // MILO uses a set of input files ... one for each interface: mesh, physics, solver, analysis, postprocessing, parameters
  
  bool have_mesh = false;
  bool have_phys = false;
  bool have_disc = false;
  bool have_solver = false;
  bool have_analysis = false;
  bool have_pp = false; // optional
  bool have_params = false; //optional
  bool have_subgrid = false; //optional
  bool have_functions = false; //optional
  
  //////////////////////////////////////////////////////////////////////////////////////////
  // Import the main input.xml file
  //////////////////////////////////////////////////////////////////////////////////////////
  
  ifstream fnmast(filename.c_str());
  if (fnmast.good()) {
    Teuchos::RCP<Teuchos::ParameterList> main_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
    int type = getFileType(filename);
    if (type == 0)
      Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*main_parlist) );
    else if (type == 1)
      Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*main_parlist) );
    
    settings->setParameters( *main_parlist );
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(!fnmast.good(),std::runtime_error,"Error: MILO could not find the main input file: " + filename);
  }
  
  
  if (settings->isSublist("Mesh"))
    have_mesh = true;
  if (settings->isSublist("Physics"))
    have_phys = true;
  if (settings->isSublist("Discretization"))
    have_disc = true;
  if (settings->isSublist("Solver"))
    have_solver = true;
  if (settings->isSublist("Analysis"))
    have_analysis = true;
  if (settings->isSublist("Postprocess"))
    have_pp = true;
  if (settings->isSublist("Parameters"))
    have_params = true;
  if (settings->isSublist("Subgrid"))
    have_subgrid = true;
  if (settings->isSublist("Functions"))
    have_functions = true;
  
  //////////////////////////////////////////////////////////////////////////////////////////
  // Some of the sublists are required (mesh, physics, solver, analysis)
  // If they do not appear in input.xml, then a file needs to be provided
  // This allows input.xml to be rather clean and easily point to different xml files
  //////////////////////////////////////////////////////////////////////////////////////////
  
  if (!have_mesh) {
    if (settings->isParameter("Mesh Settings File")) {
      std::string filename = settings->get<std::string>("Mesh Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> mesh_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*mesh_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*mesh_parlist) );
        
        settings->setParameters( *mesh_parlist );
      }
      else {
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the mesh settings file: " + filename);
      }
    }
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either a Mesh sublist or a path to a mesh settings file!");
  }
  
  if (!have_phys) {
    if (settings->isParameter("Physics Settings File")) {
      std::string filename = settings->get<std::string>("Physics Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> phys_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*phys_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*phys_parlist) );
        
        settings->setParameters( *phys_parlist );
      }
      else
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the physics settings file: " + filename);
    }
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either a Physics sublist or a path to a physics settings file!");
  }
  
  if (!have_disc) {
    if (settings->isParameter("Discretization Settings File")) {
      std::string filename = settings->get<std::string>("Discretization Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> disc_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*disc_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*disc_parlist) );
        
        settings->setParameters( *disc_parlist );
      }
      else
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the discretization settings file: " + filename);
    }
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either a Discretization sublist or a path to a Discretization settings file!");
  }
  
  if (!have_solver) {
    if (settings->isParameter("Solver Settings File")) {
      std::string filename = settings->get<std::string>("Solver Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> solver_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*solver_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*solver_parlist) );
        
        settings->setParameters( *solver_parlist );
      }
      else
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the solver settings file:" + filename);
    }
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either a Solver sublist or a path to a solver settings file!");
  }
  
  if (!have_analysis) {
    if (settings->isParameter("Analysis Settings File")) {
      std::string filename = settings->get<std::string>("Analysis Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> analysis_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*analysis_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*analysis_parlist) );
    
        settings->setParameters( *analysis_parlist );
      }
      else
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the analysis settings file: " + filename);
    }
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: the input.xml needs to contain either an Analysis sublist or a path to an analysis settings file!");
  }
  
  if (!have_pp) { // this is optional (but recommended!)
    if (settings->isParameter("Postprocess Settings File")) {
      std::string filename = settings->get<std::string>("Postprocess Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> pp_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*pp_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*pp_parlist) );
    
        settings->setParameters( *pp_parlist );
      }
      else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the postprocess settings file: " + filename);
    }
    else
      settings->sublist("Postprocess",false,"Empty sublist for postprocessing.");
  }
  
  if (!have_params) { // this is optional
    if (settings->isParameter("Parameters Settings File")) {
      std::string filename = settings->get<std::string>("Parameters Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> param_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*param_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*param_parlist) );
    
        settings->setParameters( *param_parlist );
      }
      else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the parameters settings file: " + filename);
    }
    else
      settings->sublist("Parameters",false,"Empty sublist for parameters.");
  }
  
  if (!have_subgrid) { // this is optional
    if (settings->isParameter("Subgrid Settings File")) {
      std::string filename = settings->get<std::string>("Subgrid Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> subgrid_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*subgrid_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*subgrid_parlist) );
    
        settings->setParameters( *subgrid_parlist );
      }
      else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the subgrid settings file: " + filename);
    }
  }
  
  if (!have_functions) { // this is optional
    if (settings->isParameter("Functions Settings File")) {
      std::string filename = settings->get<std::string>("Functions Settings File");
      ifstream fn(filename.c_str());
      if (fn.good()) {
        Teuchos::RCP<Teuchos::ParameterList> functions_parlist = Teuchos::rcp( new Teuchos::ParameterList() );
        int type = getFileType(filename);
        if (type == 0)
          Teuchos::updateParametersFromYamlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*functions_parlist) );
        else if (type == 1)
          Teuchos::updateParametersFromXmlFile( filename, Teuchos::Ptr<Teuchos::ParameterList>(&*functions_parlist) );
    
        settings->setParameters( *functions_parlist );
      }
      else // this sublist is not required, but if you specify a file then an exception will be thrown if it cannot be found
        TEUCHOS_TEST_FOR_EXCEPTION(!fn.good(),std::runtime_error,"Error: MILO could not find the functions settings file: " + filename);
    }
  }
  return settings;
}

#endif
