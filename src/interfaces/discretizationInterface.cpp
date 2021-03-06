/***********************************************************************
 Multiscale/Multiphysics Interfaces for Large-scale Optimization (MILO)
 
 Copyright 2018 National Technology & Engineering Solutions of Sandia,
 LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
 U.S. Government retains certain rights in this software.”
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) and/or
 Bart van Bloemen Waanders (bartv@sandia.gov)
 ************************************************************************/

#include "discretizationInterface.hpp"
#include "discretizationTools.hpp"

discretization::discretization(Teuchos::RCP<Teuchos::ParameterList> & settings,
                               Teuchos::RCP<LA_MpiComm> & Comm_,
                               Teuchos::RCP<panzer_stk::STK_Interface> & mesh,
                               vector<vector<int> > & orders, vector<vector<string> > & types,
                               vector<vector<Teuchos::RCP<cell> > > & cells) :
Commptr(Comm_) {
  
  milo_debug_level = settings->get<int>("debug level",0);
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Starting discretization constructor..." << endl;
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Collect some information
  ////////////////////////////////////////////////////////////////////////////////
  
  int spaceDim = settings->sublist("Mesh").get<int>("dim",2);
  std::vector<string> blocknames;
  mesh->getElementBlockNames(blocknames);
  
  string shape = settings->sublist("Mesh").get<string>("shape","quad");
  
  ////////////////////////////////////////////////////////////////////////////////
  // Assemble the information we always store
  ////////////////////////////////////////////////////////////////////////////////
  
  for (size_t b=0; b<cells.size(); b++) {
    
    string blockID = blocknames[b];
    Teuchos::ParameterList db_settings;
    if (settings->sublist("Discretization").isSublist(blocknames[b])) {
      db_settings = settings->sublist("Discretization").sublist(blocknames[b]);
    }
    else {
      db_settings = settings->sublist("Discretization");
    }
    
    vector<stk::mesh::Entity> stk_meshElems;
    mesh->getMyElements(blockID, stk_meshElems);
    
    // list of all elements on this processor
    vector<size_t> blockmyElements = vector<size_t>(stk_meshElems.size());
    for( size_t e=0; e<stk_meshElems.size(); e++ ) {
      blockmyElements[e] = mesh->elementLocalId(stk_meshElems[e]);
    }
    myElements.push_back(blockmyElements);
    
    ///////////////////////////////////////////////////////////////////////////
    // Modify the mesh (if requested)
    ///////////////////////////////////////////////////////////////////////////
    
    //FC blocknodePert = this->modifyMesh(settings, blocknodeVert);
    
    //nodeVert.push_back(blocknodeVert_vec);
    //nodePert.push_back(blocknodePert);
    
    ///////////////////////////////////////////////////////////////////////////
    // Get the cardinality of the basis functions  on this block
    ///////////////////////////////////////////////////////////////////////////
    
    topo_RCP cellTopo = mesh->getCellTopology(blockID);
    vector<int> blockcards;
    vector<basis_RCP> blockbasis;//(blockmaxorder);
    
    vector<int> doneorders;
    vector<string> donetypes;
    for (size_t n=0; n<orders[b].size(); n++) {
      bool go = true;
      for (size_t i=0; i<doneorders.size(); i++){
        if (doneorders[i] == orders[b][n] && donetypes[i] == types[b][n]) {
          go = false;
        }
      }
      if (go) {
        basis_RCP basis = DiscTools::getBasis(spaceDim, cellTopo, types[b][n], orders[b][n]);
        int bsize = basis->getCardinality();
        blockcards.push_back(bsize); // cardinality of the basis
        blockbasis.push_back(basis);
        doneorders.push_back(orders[b][n]);
        donetypes.push_back(types[b][n]);
      }
    }
    basis_types.push_back(donetypes);
    cards.push_back(blockcards);
    
    ///////////////////////////////////////////////////////////////////////////
    // Quadrature
    ///////////////////////////////////////////////////////////////////////////
    
    int mxorder = 0;
    for (int i=0; i<orders[b].size(); i++) {
      if (orders[b][i]>mxorder) {
        mxorder = orders[b][i];
      }
    }
    
    DRV qpts, qwts;
    int quadorder = db_settings.get<int>("quadrature",mxorder+1);
    DiscTools::getQuadrature(cellTopo, quadorder, qpts, qwts);
    
    ///////////////////////////////////////////////////////////////////////////
    // Side Quadrature
    ///////////////////////////////////////////////////////////////////////////
    
    topo_RCP sideTopo;
    
    int sideDim = spaceDim-1;
    if (spaceDim == 1) {
      if (shape == "line") {
	sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Particle>() ));
      }
    }
    if (spaceDim == 2) {
      if (shape == "quad") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
      if (shape == "tri") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<> >() ));
      }
    }
    if (spaceDim == 3) {
      if (shape == "hex") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >() ));
      }
      if (shape == "tet") {
        sideTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<> >() ));
      }
    }
    
    DRV side_qpts, side_qwts;
    int side_quadorder = db_settings.sublist(blockID).get<int>("side quadrature",mxorder+1);

    if(spaceDim > 1) 
      DiscTools::getQuadrature(sideTopo, side_quadorder, side_qpts, side_qwts);
    
    ///////////////////////////////////////////////////////////////////////////
    // Store locally
    ///////////////////////////////////////////////////////////////////////////
    
    basis_pointers.push_back(blockbasis);
    ref_ip.push_back(qpts);
    ref_wts.push_back(qwts);
    ref_side_ip.push_back(side_qpts);
    ref_side_wts.push_back(side_qwts);
    numip.push_back(qpts.extent(0));
    numip_side.push_back(side_qpts.extent(0));
    
  } // block loop
  
  if (milo_debug_level > 0) {
    if (Commptr->getRank() == 0) {
      cout << "**** Finished discretization constructor" << endl;
    }
  }
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void discretization::setIntegrationInfo(vector<vector<Teuchos::RCP<cell> > > & cells,
                                        vector<vector<Teuchos::RCP<BoundaryCell> > > & boundaryCells,
                                        Teuchos::RCP<panzer::DOFManager> & DOF,
                                        Teuchos::RCP<physics> & phys) {
  for (size_t b=0; b<cells.size(); b++) {
    int eprog = 0;
    for (size_t e=0; e<cells[b].size(); e++) {
      int numElem = cells[b][e]->numElem;
      
      // Build the Kokkos View of the cell GIDs ------
      vector<vector<panzer::GlobalOrdinal> > cellGIDs;
      int numLocalDOF = 0;
      for (int i=0; i<numElem; i++) {
	std::vector<panzer::GlobalOrdinal> GIDs;
	panzer::LocalOrdinal elemID = this->myElements[b][eprog+i];
        DOF->getElementGIDs(elemID, GIDs);
        cellGIDs.push_back(GIDs);
        numLocalDOF = GIDs.size(); // should be the same for all elements
      }
      Kokkos::View<GO**,HostDevice> hostGIDs("GIDs on host device",numElem,numLocalDOF);
      for (int i=0; i<numElem; i++) {
        for (int j=0; j<numLocalDOF; j++) {
          hostGIDs(i,j) = cellGIDs[i][j];
        }
      }
      cells[b][e]->GIDs = hostGIDs;
      //-----------------------------------------------
      
      Kokkos::View<int*> localEID = cells[b][e]->localElemID;
      
      // Set the side information (soon to be removed)-
      Kokkos::View<int****,HostDevice> sideinfo = phys->getSideInfo(b,localEID);
      cells[b][e]->sideinfo = sideinfo;
      cells[b][e]->sidenames = phys->sideSets;
      //-----------------------------------------------
      
      // Set the cell orientation (not really used) ---
      vector<vector<ScalarT> > cellOrient;
      for (int i=0; i<numElem; i++) {
        vector<ScalarT> orient;
        size_t elemID = this->myElements[b][eprog+i];
        DOF->getElementOrientation(elemID, orient);
        cellOrient.push_back(orient);
      }
      cells[b][e]->orientation = cellOrient;
      //-----------------------------------------------
      
      eprog += numElem;
      
    }
  }
  
  for (size_t b=0; b<boundaryCells.size(); b++) {
    for (size_t e=0; e<boundaryCells[b].size(); e++) {
      int numElem = boundaryCells[b][e]->numElem;
      
      // Build the Kokkos View of the cell GIDs ------
      vector<vector<panzer::GlobalOrdinal> > cellGIDs;
      int numLocalDOF = 0;
      for (int i=0; i<numElem; i++) {
	std::vector<panzer::GlobalOrdinal> GIDs;
	panzer::LocalOrdinal elemID = boundaryCells[b][e]->localElemID(i);
        DOF->getElementGIDs(elemID, GIDs);
        cellGIDs.push_back(GIDs);
        numLocalDOF = GIDs.size(); // should be the same for all elements
      }
      Kokkos::View<GO**,HostDevice> hostGIDs("GIDs on host device",numElem,numLocalDOF);
      for (int i=0; i<numElem; i++) {
        for (int j=0; j<numLocalDOF; j++) {
          hostGIDs(i,j) = cellGIDs[i][j];
        }
      }
      boundaryCells[b][e]->GIDs = hostGIDs;
      //-----------------------------------------------
    
      Kokkos::View<int*> localEID = boundaryCells[b][e]->localElemID;
      
      // Set the side information (soon to be removed)-
      Kokkos::View<int****,HostDevice> sideinfo = phys->getSideInfo(b,localEID);
      boundaryCells[b][e]->sideinfo = sideinfo;
      //-----------------------------------------------
      
    }
  }
  // Set the cell integration points/wts-----------
  for (size_t b=0; b<cells.size(); b++) {
    for (size_t e=0; e<cells[b].size(); e++) {
      cells[b][e]->setIP(ref_ip[b]);
      //cells[b][e]->setSideIP(ref_side_ip[b], ref_side_wts[b]);
    }
  }
  //-----------------------------------------------
  
  // Set the boundary cell integration points/wts -
  //for (size_t b=0; b<boundaryCells.size(); b++) {
  //  for (size_t e=0; e<boundaryCells[b].size(); e++) {
  //    int s = boundaryCells[b][e]->sidenum;
      //boundaryCells[b][e]->setIP(ref_side_ip[b], ref_side_wts[b]);
      //cells[b][e]->setSideIP(ref_side_ip[b], ref_side_wts[b]);
  //  }
  //}
  //-----------------------------------------------
  
  
}


