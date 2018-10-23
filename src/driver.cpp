/***********************************************************************
Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)

Copyright 2018 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
U.S. Government retains certain rights in this software.”

Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
Bart van Bloemen Waanders (bartv@sandia.gov)
***********************************************************************/

#include "userInterface.hpp"
#include "cell.hpp"
#include "meshInterface.hpp"
#include "physicsInterface.hpp"
#include "discretizationInterface.hpp"
#include "solverInterface.hpp"
#include "postprocessInterface.hpp"
#include "analysisInterface.hpp"
#include "trilinos.hpp"
#include "preferences.hpp"
#include "subgridGenerator.hpp"
#include "milo_help.hpp"
#include "functionInterface.hpp"
#include "split_mpi_communicators.hpp"

int main(int argc,char * argv[]) {
  
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  LA_MpiComm Comm(MPI_COMM_WORLD);
#else
  EPIC_FAIL // MILO requires MPI for HostDevice
#endif
  
  int verbosity = 0;
  bool profile = false;
  
  Kokkos::initialize();
  Teuchos::RCP<LA_MpiComm> tcomm_LA;
  Teuchos::RCP<LA_MpiComm> tcomm_S;
  
  Teuchos::RCP<Teuchos::Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("driver::total setup and execution time");
  Teuchos::RCP<Teuchos::Time> runTimer = Teuchos::TimeMonitor::getNewCounter("driver::total run time");
  
  string input_file_name = "input.yaml";
  if (argc > 1) {
    input_file_name = argv[1];
  }
  if (input_file_name == "--version") {
    cout << endl << "MILO - Multiscale/Multiphysics Interfaces for Large-scale Optimization -- Version " << MILO_VERSION << endl << endl;
    cout << "Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS)." << endl;
    cout << "Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software." << endl << endl;
    cout << "Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or Bart van Bloemen Waanders (bartv@sandia.gov)" << endl << endl;
  }
  else if (input_file_name == "--help") {
    if (argc > 2) {
      string helpwhat = argv[2];
      string details = "none";
      if (argc > 3) {
        details = argv[3];
      }
      MILOHelp::printHelp(helpwhat, details);
    }
    else {
      cout << endl << "MILO - Multiscale/Multiphysics Interfaces for Large-scale Optimization" << endl << endl;
      cout << "Suggested help topics: user, physics, solver, analysis, postprocess" << endl;
      cout << "Example: mpiexec -n 1 milo --help solver" << endl << endl;
    }
  }
  else {
    
    Teuchos::TimeMonitor ttimer(*totalTimer);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Import default and user-defined settings into a parameter list
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<Teuchos::ParameterList> settings = userInterface(input_file_name);
    
    verbosity = settings->get<int>("verbosity",0);
    profile = settings->get<bool>("profile",false);
    
    ////////////////////////////////////////////////////////////////////////////////
    // split comm for SOL or multiscale runs
    ////////////////////////////////////////////////////////////////////////////////
    
    SplitComm(settings, Comm, tcomm_LA, tcomm_S);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the mesh
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<meshInterface> mesh = Teuchos::rcp(new meshInterface(settings, tcomm_LA) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the function manager
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<FunctionInterface> functionManager = Teuchos::rcp(new FunctionInterface(settings));
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the physics
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<physics> phys = Teuchos::rcp( new physics(settings, tcomm_LA,
                                                           mesh->cellTopo,
                                                           mesh->sideTopo,
                                                           functionManager) );
    mesh->finalize(phys);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the cells
    ////////////////////////////////////////////////////////////////////////////////
    
    vector<vector<Teuchos::RCP<cell> > > cells;
    mesh->createCells(phys,cells);
    phys->setPeriBCs(settings, mesh->mesh);

    ////////////////////////////////////////////////////////////////////////////////
    // Define the discretization(s)
    ////////////////////////////////////////////////////////////////////////////////
        
    Teuchos::RCP<discretization> disc = Teuchos::rcp( new discretization(settings, mesh->mesh,
                                                                         phys->unique_orders,
                                                                         phys->unique_types,
                                                                         cells) );
    
    //udfunc->setVarlist(phys->varlist); // TMW: this assumes one block
    
    ////////////////////////////////////////////////////////////////////////////////
    // The DOF-manager needs to be aware of the physics and the discretization(s)
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<panzer::DOFManager<int,int> > DOF = phys->buildDOF(mesh->mesh);
    phys->setBCData(settings, mesh->mesh, DOF, disc->cards);
    
    
    disc->setIntegrationInfo(cells, DOF, phys);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Set up the subgrid discretizations/models if using multiscale method
    ////////////////////////////////////////////////////////////////////////////////
    
    vector<Teuchos::RCP<SubGridModel> > subgridModels = subgridGenerator(tcomm_S, settings, mesh->mesh);
    
    Teuchos::RCP<MultiScale> multiscale_manager = Teuchos::rcp( new MultiScale(tcomm_S, settings,
                                                                               cells, subgridModels,
                                                                               functionManager) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the solver object
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<solver> solve = Teuchos::rcp( new solver(tcomm_LA, settings, mesh->mesh,
                                                          disc, phys, DOF, cells) );
    
    solve->multiscale_manager = multiscale_manager;
    //solve->finalizeMultiscale();
    solve->setBatchID(tcomm_S->MyPID());
    
    ////////////////////////////////////////////////////////////////////////////////
    // Finalize the functions
    ////////////////////////////////////////////////////////////////////////////////
    
    functionManager->setupLists(phys->varlist[0], solve->paramnames,
                                solve->discretized_param_names);
    
    functionManager->wkset = solve->wkset[0];
    
    ////////////////////////////////////////////////////////////////////////////////
    // Create the postprocessing object
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<postprocess> postproc = Teuchos::rcp( new postprocess(tcomm_LA, settings,
                                                                       mesh->mesh, disc, phys,
                                                                       solve, DOF, cells,
                                                                       functionManager) );
    
    ////////////////////////////////////////////////////////////////////////////////
    // Perform the requested analysis (fwd solve, adj solve, dakota run, etc.)
    // stored in settings->get<string>("analysis_type")
    ////////////////////////////////////////////////////////////////////////////////
    
    Teuchos::RCP<analysis> analys = Teuchos::rcp( new analysis(tcomm_LA, tcomm_S, settings, solve, postproc) );
    
    functionManager->validateFunctions();
    functionManager->decomposeFunctions();
    
    solve->finalizeMultiscale();
    solve->setupSensors(settings); // moved here so subcells can have sensors
    
    if (verbosity >= 20) {
      functionManager->printFunctions();
    }
    
    {
      Teuchos::TimeMonitor rtimer(*runTimer);
      analys->run();
    }
    
  }
  
  if (verbosity >= 10) {
    Teuchos::TimeMonitor::summarize();
  }
  else if (profile) {
    std::filebuf fb;
    fb.open ("MILO.profile",std::ios::out);
    std::ostream os(&fb);
    Teuchos::RCP<Teuchos::ParameterList> outlist = rcp(new Teuchos::ParameterList("options"));
    outlist->set("Report format","YAML");
    std::string filter = "";
    Teuchos::TimeMonitor::report(os, filter, outlist);
    fb.close();
  }
  
  Kokkos::finalize();
  
}
