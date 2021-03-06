// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include "../config/config.hpp"

#ifdef MFEM_USE_SIDRE

#include "fem.hpp"

#ifdef MFEM_USE_MPI
#include <spio/IOManager.hpp>
#endif

#include <string>
#include <iomanip>      // for setw, setfill
#include <cstdio>       // for snprintf()

namespace sidre = axom::sidre;

namespace mfem
{

// Constructor that will automatically create the sidre data store and necessary
// data groups for domain and global data.
SidreDataCollection::SidreDataCollection(const std::string& collection_name,
                                         Mesh * the_mesh, bool own_mesh_data)
   : mfem::DataCollection(collection_name, the_mesh),
     m_owns_datastore(true),
     m_owns_mesh_data(own_mesh_data),
     m_meshNodesGFName("mesh_nodes")
{
   m_datastore_ptr = new sidre::DataStore();

   sidre::DataGroup * global_grp =
      m_datastore_ptr->getRoot()->createGroup(collection_name + "_global");
   sidre::DataGroup * domain_grp =
      m_datastore_ptr->getRoot()->createGroup(collection_name);

   bp_grp = domain_grp->createGroup("blueprint");
   // Currently only rank 0 adds anything to bp_index.
   bp_index_grp = global_grp->createGroup("blueprint_index/" + name);

   named_bufs_grp = domain_grp->createGroup("named_buffers");

   if (the_mesh)
   {
      SetMesh(the_mesh);
   }
#ifdef MFEM_USE_MPI
   else
   {
      m_comm = MPI_COMM_NULL;
   }
#endif
}

// Second constructor that allows external code to specify data groups to place
// domain and global data in.

// TODO - Conduit will have the capability to generate a blueprint index group
// in the future.  When this is available, all the blueprint index code can be
// removed from the data collection class.
SidreDataCollection::SidreDataCollection(const std::string& collection_name,
                                         axom::sidre::DataGroup* global_grp,
                                         axom::sidre::DataGroup* domain_grp,
                                         bool own_mesh_data)
   : mfem::DataCollection(collection_name),
     m_owns_datastore(false),
     m_owns_mesh_data(own_mesh_data),
     m_meshNodesGFName("mesh_nodes"),
     m_datastore_ptr(NULL)
{
   bp_grp = domain_grp->createGroup("blueprint");

   // Currently only rank 0 adds anything to bp_index.
   bp_index_grp = global_grp->createGroup("blueprint_index/" + name);

   named_bufs_grp = domain_grp->createGroup("named_buffers");

#ifdef MFEM_USE_MPI
   m_comm = MPI_COMM_NULL;
#endif
}

SidreDataCollection::~SidreDataCollection()
{
   if (m_owns_datastore)
   {
      delete m_datastore_ptr;
   }
}

#ifdef MFEM_USE_MPI
void SidreDataCollection::SetComm(MPI_Comm comm)
{
   m_comm = comm;
   serial = false;
   appendRankToFileName = true;
   MPI_Comm_rank(m_comm, &myid);
   MPI_Comm_size(m_comm, &num_procs);
}
#endif

// protected method
sidre::DataGroup *SidreDataCollection::named_buffers_grp() const
{
   MFEM_ASSERT(named_bufs_grp != NULL,
               "No group 'named_buffers' in data collection.  Verify that"
               " SetMesh was called to set the mesh in the data collection.");
   return named_bufs_grp;
}

// protected method
axom::sidre::DataView *
SidreDataCollection::alloc_view(axom::sidre::DataGroup *grp,
                                const std::string &view_name)
{
   MFEM_ASSERT(grp, "DataGroup pointer is NULL");
   sidre::DataView *v = grp->getView(view_name);
   if (!v)
   {
      v = grp->createView(view_name);
      MFEM_ASSERT(v, "error allocating DataView " << view_name
                  << " in group " << grp->getPathName());
   }
   return v;
}

// protected method
axom::sidre::DataView *
SidreDataCollection::alloc_view(axom::sidre::DataGroup *grp,
                                const std::string &view_name,
                                const axom::sidre::DataType &dtype)
{
   MFEM_ASSERT(grp, "DataGroup pointer is NULL");
   sidre::DataView *v = grp->getView(view_name);
   if (!v)
   {
      v = grp->createView(view_name, dtype);
      MFEM_ASSERT(v, "error allocating DataView " << view_name
                  << " in group " << grp->getPathName());
   }
   else
   {
      MFEM_ASSERT(v->getSchema().dtype().equals(dtype), "");
   }
   return v;
}

// protected method
axom::sidre::DataGroup *
SidreDataCollection::alloc_group(axom::sidre::DataGroup *grp,
                                 const std::string &group_name)
{
   MFEM_ASSERT(grp, "DataGroup pointer is NULL");
   sidre::DataGroup *g = grp->getGroup(group_name);
   if (!g)
   {
      g = grp->createGroup(group_name);
      MFEM_ASSERT(g, "error allocating DataGroup " << group_name
                  << " in group " << grp->getPathName());
   }
   return g;
}

// protected method
std::string
SidreDataCollection::get_file_path(const std::string &filename) const
{
   std::stringstream fNameSstr;

   // Note: If non-empty, prefix_path has a separator ('/') at the end
   fNameSstr << prefix_path << filename;

   if (GetCycle() >= 0)
   {
      fNameSstr << "_" << std::setfill('0') << std::setw(pad_digits)
                << GetCycle();
   }

   return fNameSstr.str();
}

axom::sidre::DataView *
SidreDataCollection::AllocNamedBuffer(const std::string& buffer_name,
                                      axom::sidre::SidreLength sz,
                                      axom::sidre::TypeID type)
{
   sz = std::max(sz, sidre::SidreLength(0));
   sidre::DataGroup *f = named_buffers_grp();
   sidre::DataView  *v = f->getView(buffer_name);
   if ( v == NULL )
   {
      // create a buffer view
      v = f->createViewAndAllocate(buffer_name, type, sz);
   }
   else
   {
      MFEM_ASSERT(v->getTypeID() == type, "type does not match existing type");

      // Here v is the view holding the buffer in the named_buffers group, so
      // its size is the full size of the buffer.

      // check if we need to resize.
      if (!v->isApplied() || v->getNumElements() < sz)
      {
         // resize, even if the buffer has more than 1 DataView.
         // v->reallocate(sz); // this will not work for more than 1 view.
         sidre::DataType dtype(v->getSchema().dtype());
         dtype.set_number_of_elements(sz);
         f->destroyViewAndData(buffer_name);
         v = f->createViewAndAllocate(buffer_name, dtype);
      }
   }
   MFEM_ASSERT(v && v->isApplied(), "allocation failed");
   return v;
}

// private method
void SidreDataCollection::createMeshBlueprintStubs(bool hasBP)
{
   if (!hasBP)
   {
      bp_grp->createGroup("state");
      bp_grp->createGroup("coordsets");
      bp_grp->createGroup("topologies");
      bp_grp->createGroup("fields");
   }

   // If rank is 0, set up blueprint index state group.
   if (myid == 0)
   {
      bp_index_grp->createGroup("state");
      bp_index_grp->createGroup("coordsets");
      bp_index_grp->createGroup("topologies");
      bp_index_grp->createGroup("fields");
   }
}

// private method
void SidreDataCollection::createMeshBlueprintState(bool hasBP)
{
   if (!hasBP)
   {
      // Set up blueprint state group.
      bp_grp->createViewScalar("state/cycle", 0);
      bp_grp->createViewScalar("state/time", 0.);
      bp_grp->createViewScalar("state/domain", myid);
      bp_grp->createViewScalar("state/time_step", 0.);
   }

   // If rank is 0, set up blueprint index state group.
   if (myid == 0)
   {
      bp_index_grp->createViewScalar("state/cycle", 0);
      bp_index_grp->createViewScalar("state/time", 0.);
      bp_index_grp->createViewScalar("state/number_of_domains", num_procs);
   }
}

// private method
void SidreDataCollection::createMeshBlueprintCoordset(bool hasBP)
{
   int dim = mesh->SpaceDimension();
   MFEM_ASSERT(dim >= 1 && dim <= 3, "invalid mesh dimension");

   // Assuming mfem::Vertex has the layout of a double array.
   const int NUM_COORDS = sizeof(mfem::Vertex) / sizeof(double);

   const int num_vertices = mesh->GetNV();
   const int coordset_len = NUM_COORDS * num_vertices;

   // Add blueprint if not present
   if ( !hasBP )
   {
      bp_grp->createViewString("coordsets/coords/type", "explicit");

      sidre::DataType dtype =
         sidre::DataType::c_double(num_vertices);
      const size_t stride = dtype.stride();
      dtype.set_stride(stride*NUM_COORDS);

      // Set up views for x, y, z values
      sidre::DataView *vx, *vy = NULL, *vz = NULL;
      vx = bp_grp->createView("coordsets/coords/values/x", dtype);

      if (dim >= 2)
      {
         dtype.set_offset(dtype.offset() + stride);
         vy = bp_grp->createView("coordsets/coords/values/y", dtype);
      }
      if (dim >= 3)
      {
         dtype.set_offset(dtype.offset() + stride);
         vz = bp_grp->createView("coordsets/coords/values/z", dtype);
      }

      if (m_owns_mesh_data)
      {
         // Allocate buffer for coord values.
         sidre::DataBuffer* coordbuf =
            AllocNamedBuffer("vertex_coords", coordset_len)->getBuffer();

         vx->attachBuffer(coordbuf);
         if (dim >= 2) { vy->attachBuffer(coordbuf); }
         if (dim >= 3) { vz->attachBuffer(coordbuf); }
      }
      else
      {
         double *coordbuf = mesh->GetVertex(0);

         vx->setExternalDataPtr(coordbuf);
         if (dim >= 2) { vy->setExternalDataPtr(coordbuf); }
         if (dim >= 3) { vz->setExternalDataPtr(coordbuf); }
      }

   }

   // If rank 0, set up blueprint index for coordinate set.
   if (myid == 0)
   {
      bp_index_grp->createViewString(
         "coordsets/coords/path", bp_grp->getPathName() + "/coordsets/coords");

      bp_index_grp->getGroup("coordsets/coords")->copyView(
         bp_grp->getView("coordsets/coords/type") );

      bp_index_grp->createViewString(
         "coordsets/coords/coord_system/type", "cartesian");

      // These are empty views, their existence in the group tree is used to
      // define the number of dims
      bp_index_grp->createView("coordsets/coords/coord_system/axes/x");

      if (dim >= 2)
      {
         bp_index_grp->createView("coordsets/coords/coord_system/axes/y");
      }

      if (dim == 3)
      {
         bp_index_grp->createView("coordsets/coords/coord_system/axes/z");
      }
   }

   if (m_owns_mesh_data)
   {
      double *coord_values = GetNamedBuffer("vertex_coords")->getData();
      // Change ownership of the mesh vertex data to sidre
      mesh->ChangeVertexDataOwnership(coord_values, coordset_len, hasBP);
   }
}

// private method
void SidreDataCollection::
createMeshBlueprintTopologies(bool hasBP, const std::string& mesh_name)
{
   const bool isBdry = (mesh_name == "boundary");

   const int num_elements = !isBdry
                            ? mesh->GetNE()
                            : mesh->GetNBE();

   MFEM_VERIFY(num_elements > 0,
               "TODO: processors with 0 " << mesh_name << " elements");

   const int element_size = !isBdry
                            ? mesh->GetElement(0)->GetNVertices()
                            : mesh->GetBdrElement(0)->GetNVertices();

   const int num_indices = num_elements * element_size;

   // Find the element shape
   // Note: Assumes homogeneous elements, so only check the first element
   const int geom =
      isBdry ?
      mesh->GetBdrElementBaseGeometry(0) :
      mesh->GetElementBaseGeometry(0);
   const std::string eltTypeStr =
      !isBdry
      ? getElementName( static_cast<Element::Type>(
                           mesh->GetElement(0)->GetType() ) )
      : getElementName( static_cast<Element::Type>(
                           mesh->GetBdrElement(0)->GetType() ) );

   const std::string mesh_topo_str = "topologies/" + mesh_name;
   const std::string mesh_attr_str = "fields/"+mesh_name+"_material_attribute";

   if ( !hasBP )
   {
      sidre::DataGroup* topology_grp = bp_grp->createGroup(mesh_topo_str);

      // Add mesh topology
      topology_grp->createViewString("type", "unstructured");
      // Note: eltTypeStr comes form the mesh
      topology_grp->createViewString("elements/shape", eltTypeStr);
      topology_grp->createViewAndAllocate(
         "elements/connectivity", sidre::INT_ID, num_indices);
      topology_grp->createViewString("coordset", "coords");

      // If the mesh has nodes, set the name of the GridFunction holding the
      // mesh nodes in the blueprint group.
      if (!isBdry && mesh->GetNodes() != NULL)
      {
         topology_grp->createViewString("grid_function",m_meshNodesGFName);
      }

      // Add material attribute field to blueprint
      sidre::DataGroup* attr_grp = bp_grp->createGroup(mesh_attr_str);
      attr_grp->createViewString("association", "element");
      attr_grp->createViewAndAllocate("values", sidre::INT_ID, num_elements);
      attr_grp->createViewString("topology", mesh_name);
   }

   // If rank 0, set up blueprint index for topologies group and material
   // attribute field.
   if (myid == 0)
   {
      const std::string bp_grp_path = bp_grp->getPathName();

      // Create blueprint index for topologies.
      if (isBdry)
      {
         // "Shallow" copy the bp_grp view into the bp_index_grp sub-group.
         // Note that the "topologies/mesh" sub-group has to exist, i.e. this
         // method should be called first with mesh_name = "mesh".
         bp_index_grp->getGroup("topologies/mesh")
         ->copyView( bp_grp->getView("topologies/mesh/boundary_topology") );
      }

      sidre::DataGroup *bp_index_topo_grp =
         bp_index_grp->createGroup(mesh_topo_str);
      sidre::DataGroup *topology_grp = bp_grp->getGroup(mesh_topo_str);

      bp_index_topo_grp->createViewString(
         "path", bp_grp_path + "/" + mesh_topo_str);
      bp_index_topo_grp->copyView( topology_grp->getView("type") );
      bp_index_topo_grp->copyView( topology_grp->getView("coordset") );

      // If the mesh has nodes, set the name of the GridFunction holding the
      // mesh nodes in the blueprint_index group.
      if (!isBdry && mesh->GetNodes() != NULL)
      {
         bp_index_topo_grp->copyView(topology_grp->getView("grid_function"));
      }

      // Create blueprint index for material attributes.
      sidre::DataGroup *bp_index_attr_grp =
         bp_index_grp->createGroup(mesh_attr_str);
      sidre::DataGroup *attr_grp = bp_grp->getGroup(mesh_attr_str);

      bp_index_attr_grp->createViewString(
         "path", bp_grp_path + "/" + mesh_attr_str );
      bp_index_attr_grp->copyView( attr_grp->getView("association") );
      bp_index_attr_grp->copyView( attr_grp->getView("topology") );

      int number_of_components = 1;
      bp_index_attr_grp->createViewScalar("number_of_components",
                                          number_of_components);
   }

   // Finally, change ownership or copy the element arrays into Sidre
   sidre::DataView* conn_view =
      bp_grp->getGroup(mesh_topo_str)->getView("elements/connectivity");
   sidre::DataView* attr_view =
      bp_grp->getGroup(mesh_attr_str)->getView("values");
   // The SidreDataCollection always owns these arrays:
   Array<int> conn_array(conn_view->getData<int*>(), num_indices);
   Array<int> attr_array(attr_view->getData<int*>(), num_elements);
   if (!isBdry)
   {
      mesh->GetElementData(geom, conn_array, attr_array);
   }
   else
   {
      mesh->GetBdrElementData(geom, conn_array, attr_array);
   }
   MFEM_ASSERT(!conn_array.OwnsData(), "");
   MFEM_ASSERT(!attr_array.OwnsData(), "");
}

// private method
void SidreDataCollection::verifyMeshBlueprint()
{
   // Conduit will have a verify mesh blueprint capability in the future.
   // Add call to that when it's available to check actual contents in sidre.
}

void SidreDataCollection::SetMesh(Mesh *new_mesh)
{
   DataCollection::SetMesh(new_mesh);

#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(new_mesh);
   m_comm = pmesh ? pmesh->GetComm() : MPI_COMM_NULL;
#endif

   // hasBP is used to indicate if the data currently in the blueprint should be
   // used to replace the data in the mesh.
   bool hasBP = bp_grp->getNumViews() > 0 || bp_grp->getNumGroups() > 0;
   bool has_bnd_elts = (new_mesh->GetNBE() > 0);

   createMeshBlueprintStubs(hasBP);
   createMeshBlueprintState(hasBP);
   createMeshBlueprintCoordset(hasBP);

   GridFunction *nodes = new_mesh->GetNodes();

   // register the "mesh" topology in the blueprint.
   createMeshBlueprintTopologies(hasBP, "mesh");

   if (has_bnd_elts)
   {
      // Set the "boundary_topology" of "mesh" to "boundary".
      bp_grp->createViewString("topologies/mesh/boundary_topology", "boundary");

      // register the "boundary" topology in the blueprint.
      createMeshBlueprintTopologies(hasBP, "boundary");
   }

   if (nodes)
   {
      // See the comment at the definition of 'hasBP' above.
      if (hasBP)
      {
         // Get the bp mesh nodes name.
         sidre::DataView *v_bp_nodes_name =
            bp_grp->getView("topologies/mesh/grid_function");
         std::string bp_nodes_name(v_bp_nodes_name->getString());

         // Check that the names match, e.g. when loading the collection.
         MFEM_VERIFY(m_meshNodesGFName == bp_nodes_name,
                     "mismatch of requested and blueprint mesh nodes names");
         // Support renaming bp_nodes_name --> m_meshNodesGFName ?
      }

      if (m_owns_mesh_data)
      {
         // Make sure Sidre owns the data of the new_mesh's Nodes.
         if (!GetNamedBuffer(m_meshNodesGFName))
         {
            int sz = new_mesh->GetNodalFESpace()->GetVSize();
            double *gfData = AllocNamedBuffer(m_meshNodesGFName, sz)->getData();

            // See the comment at the definition of 'hasBP' above.
            if (!hasBP)
            {
               MFEM_ASSERT(nodes->Size() == sz, "");
               std::memcpy(gfData, nodes->GetData(), sizeof(double) * sz);
            }
         }
         // Since the named buffer for m_meshNodesGFName exists, the call to
         // RegisterField() below will replace the data of the nodes with the
         // data from the named buffer.
      }
      else
      {
         // Make sure Sidre does not have a named buffer for m_meshNodesGFName.
         MFEM_VERIFY(GetNamedBuffer(m_meshNodesGFName) == NULL, "");
      }

      RegisterField(m_meshNodesGFName, nodes);

      if (own_data)
      {
         // Avoid double delete calls (for the nodes gf) when (own_data == true)
         // and the new_mesh owns its Nodes --> take ownership from new_mesh.
         // When new_mesh does not own its Nodes and (own_data == true), we can
         // not take ownership --> verify that does not happen.
         MFEM_VERIFY(new_mesh->OwnsNodes(), "mesh does not own its nodes, "
                     "can not take ownership");
         new_mesh->SetNodesOwner(false);
      }
   }
}

void SidreDataCollection::
SetGroupPointers(axom::sidre::DataGroup *global_grp,
                 axom::sidre::DataGroup *domain_grp)
{
   MFEM_VERIFY(domain_grp->hasGroup("blueprint"),
               "Domain group does not contain a blueprint group.");
   MFEM_VERIFY(global_grp->hasGroup("blueprint_index/" + name),
               "Global group does not contain a blueprint index group.");

   bp_grp = domain_grp->getGroup("blueprint");
   bp_index_grp = global_grp->getGroup("blueprint_index/" + name);
   named_bufs_grp = domain_grp->getGroup("named_buffers");
}

void SidreDataCollection::Load(const std::string& path,
                               const std::string& protocol)
{
   DataCollection::DeleteAll();
   // Reset DataStore?

#ifdef MFEM_USE_MPI
   if (m_comm != MPI_COMM_NULL)
   {
      axom::spio::IOManager reader(m_comm);
      reader.read(bp_grp->getDataStore()->getRoot(), path);
   }
   else
#endif
   {
      bp_grp->load(path, protocol);
   }

   // If the data collection created the datastore, it knows the layout of where
   // the domain and global groups are, and can restore them after the Load().
   //
   // If the data collection did not create the datastore, the host code must
   // reset these pointers after the load operation and also reset the state
   // variables.
   if (m_owns_datastore)
   {
      SetGroupPointers(m_datastore_ptr->getRoot()->getGroup(name + "_global"),
                       m_datastore_ptr->getRoot()->getGroup(name));

      UpdateStateFromDS();
   }
}

void SidreDataCollection::LoadExternalData(const std::string& path)
{
#ifdef MFEM_USE_MPI
   if (m_comm != MPI_COMM_NULL)
   {
      axom::spio::IOManager reader(m_comm);
      reader.loadExternalData(bp_grp->getDataStore()->getRoot(), path);
   }
   else
#endif
   {
      bp_grp->loadExternalData(path);
   }
}

void SidreDataCollection::UpdateStateFromDS()
{
   SetTime( bp_grp->getView("state/time")->getData<double>() );
   SetCycle( bp_grp->getView("state/cycle")->getData<int>() );
   SetTimeStep( bp_grp->getView("state/time_step")->getData<double>() );
}

void SidreDataCollection::UpdateStateToDS()
{
   bp_grp->getView("state/cycle")->setScalar(GetCycle());
   bp_grp->getView("state/time")->setScalar(GetTime());
   bp_grp->getView("state/time_step")->setScalar(GetTimeStep());

   if (myid == 0)
   {
      bp_index_grp->getView("state/cycle")->setScalar(GetCycle());
      bp_index_grp->getView("state/time")->setScalar(time);
   }
}

void SidreDataCollection::PrepareToSave()
{
   verifyMeshBlueprint();
   UpdateStateToDS();
}

void SidreDataCollection::Save()
{
   std::string filename = name;
   std::string protocol = "sidre_hdf5";

   Save(filename, protocol);
}

void SidreDataCollection::Save(const std::string& filename,
                               const std::string& protocol)
{
   PrepareToSave();

   create_directory(prefix_path, mesh, myid);

   std::string file_path = get_file_path(filename);

   sidre::DataGroup * blueprint_indicies_grp = bp_index_grp->getParent();
#ifdef MFEM_USE_MPI
   if (m_comm != MPI_COMM_NULL)
   {
      axom::spio::IOManager writer(m_comm);
      sidre::DataStore *datastore = bp_grp->getDataStore();
      writer.write(datastore->getRoot(), num_procs, file_path, protocol);
      if (myid == 0)
      {
         if (protocol == "sidre_hdf5")
         {
            writer.writeGroupToRootFile(blueprint_indicies_grp,
                                        file_path + ".root");
         }
         // Root file support only available in hdf5.
         else
         {
            writer.write(blueprint_indicies_grp, 1,
                         file_path + ".root", protocol);
         }
      }
   }
   else
#endif
   {
      // If serial, use sidre group writer.
      bp_grp->save(file_path, protocol);

      blueprint_indicies_grp
      ->save(file_path + ".root", protocol);
   }
}

// private method
void SidreDataCollection::
addScalarBasedGridFunction(const std::string &field_name, GridFunction *gf,
                           const std::string &buffer_name,
                           axom::sidre::SidreLength offset)
{
   sidre::DataGroup* grp = bp_grp->getGroup("fields/" + field_name);
   MFEM_ASSERT(grp != NULL, "field " << field_name << " does not exist");

   const int numDofs = gf->FESpace()->GetVSize();

   if (gf->GetData() == NULL)
   {
      AllocNamedBuffer(buffer_name, offset + numDofs);
      // gf->data is set below.
   }

   /*
    *  Mesh blueprint for a scalar-based grid function is of the form
    *    /fields/field_name/basis
    *              -- string value is GridFunction's FEC::Name
    *    /fields/field_name/values
    *              -- array of size numDofs
    */

   // Make sure we have the DataView "values".
   sidre::DataView *vv = alloc_view(grp, "values");

   // Describe and apply the "values" DataView.
   // If the data store has buffer for field_name (e.g. AllocNamedBuffer was
   // called, or it was loaded from file), use that buffer.
   sidre::DataView *bv = named_buffers_grp()->getView(buffer_name);
   if (bv)
   {
      MFEM_ASSERT(bv->hasBuffer() && bv->isDescribed(), "");

      // named buffers always have offset 0
      MFEM_ASSERT(bv->getSchema().dtype().offset() == 0, "");
      MFEM_ASSERT(bv->getNumElements() >= offset + numDofs, "");

      if (vv->isEmpty())
      {
         vv->attachBuffer(bv->getBuffer())
         ->apply(sidre::DOUBLE_ID, numDofs, offset);
      }

      gf->NewDataAndSize(vv->getData(), numDofs);
   }
   else
   {
      // If we are not managing the grid function's data,
      // create a view with the external data
      vv->setExternalDataPtr(sidre::DOUBLE_ID, numDofs, gf->GetData());
   }
   MFEM_ASSERT((numDofs > 0 && vv->isApplied()) ||
               (numDofs == 0 && vv->isEmpty() && vv->isDescribed()),
               "invlid DataView state");
   MFEM_ASSERT(numDofs == 0 || vv->getData() == gf->GetData(),
               "DataView data is different from GridFunction data");
   MFEM_ASSERT(vv->getNumElements() == numDofs,
               "DataView size is different from GridFunction size");
}

// private method
void SidreDataCollection::
addVectorBasedGridFunction(const std::string& field_name, GridFunction *gf,
                           const std::string &buffer_name,
                           axom::sidre::SidreLength offset)
{
   sidre::DataGroup* grp = bp_grp->getGroup("fields/" + field_name);
   MFEM_ASSERT(grp != NULL, "field " << field_name << " does not exist");

   const int FLD_SZ = 20;
   char fidxName[FLD_SZ];

   int vdim = gf->FESpace()->GetVDim();
   int ndof = gf->FESpace()->GetNDofs();
   Ordering::Type ordering = gf->FESpace()->GetOrdering();

   if (gf->GetData() == NULL)
   {
      AllocNamedBuffer(buffer_name, offset + vdim*ndof);
      // gf->data is set below.
   }

   /*
    *  Mesh blueprint for a vector-based grid function is of the form
    *    /fields/field_name/basis
    *              -- string value is GridFunction's FEC::Name
    *    /fields/field_name/values/x0
    *    /fields/field_name/values/x1
    *    ...
    *    /fields/field_name/values/xn
    *              -- each coordinate is an array of size ndof
    */

   // Get/create the DataGroup "values".
   sidre::DataGroup *vg = alloc_group(grp, "values");

   // Create the DataViews "x0", "x1", etc inside the "values" DataGroup, vg.
   // If we have a named buffer for field_name, attach it to the DataViews;
   // otherwise set the DataViews to use gf->GetData() as external data.
   sidre::DataType dtype = sidre::DataType::c_double(ndof);
   const int entry_stride = (ordering == Ordering::byNODES ? 1 : vdim);
   const int vdim_stride  = (ordering == Ordering::byNODES ? ndof : 1);
   dtype.set_stride(dtype.stride()*entry_stride);

   sidre::DataView *bv = named_buffers_grp()->getView(buffer_name);
   if (bv)
   {
      MFEM_ASSERT(bv->hasBuffer() && bv->isDescribed(), "");

      // named buffers always have offset 0
      MFEM_ASSERT(bv->getSchema().dtype().offset() == 0, "");
      dtype.set_offset(dtype.element_bytes()*offset);

      for (int d = 0;  d < vdim; d++)
      {
         std::snprintf(fidxName, FLD_SZ, "x%d", d);
         sidre::DataView *xv = alloc_view(vg, fidxName, dtype);
         xv->attachBuffer(bv->getBuffer());
         dtype.set_offset(dtype.offset() + dtype.element_bytes()*vdim_stride);
      }

      gf->NewDataAndSize(bv->getData<double*>() + offset, vdim*ndof);
   }
   else
   {
      for (int d = 0;  d < vdim; d++)
      {
         std::snprintf(fidxName, FLD_SZ, "x%d", d);
         sidre::DataView *xv = alloc_view(vg, fidxName, dtype);
         xv->setExternalDataPtr(gf->GetData());
         dtype.set_offset(dtype.offset() + dtype.element_bytes()*vdim_stride);
      }
   }

#ifdef MFEM_DEBUG
   for (int d = 0;  d < vdim; d++)
   {
      std::snprintf(fidxName, FLD_SZ, "x%d", d);
      sidre::DataView *xv = vg->getView(fidxName);
      MFEM_ASSERT((ndof > 0 && xv->isApplied()) ||
                  (ndof == 0 && xv->isEmpty() && xv->isDescribed()),
                  "invlid DataView state");
      MFEM_ASSERT(ndof == 0 || xv->getData() == gf->GetData() + d*vdim_stride,
                  "DataView data is different from GridFunction data");
      MFEM_ASSERT(xv->getNumElements() == ndof,
                  "DataView size is different from GridFunction size");
   }
#endif
}

// private method
// Should only be called on mpi rank 0 ( or if serial problem ).
void SidreDataCollection::
RegisterFieldInBPIndex(const std::string& field_name, GridFunction *gf)
{
   sidre::DataGroup *bp_field_grp = bp_grp->getGroup("fields/" + field_name);
   sidre::DataGroup *bp_index_field_grp =
      bp_index_grp->createGroup("fields/" + field_name);

   bp_index_field_grp->createViewString( "path", bp_field_grp->getPathName() );
   bp_index_field_grp->copyView( bp_field_grp->getView("topology") );
   bp_index_field_grp->copyView( bp_field_grp->getView("basis") );

   // Note: The bp index requires GridFunction::VectorDim()
   //       since the GF might be scalar valued and have a vector basis
   //       (e.g. hdiv and hcurl spaces)
   const int number_of_components = gf->VectorDim();
   bp_index_field_grp->createViewScalar("number_of_components",
                                        number_of_components);
}

// private method
// Should only be called on mpi rank 0 ( or if serial problem ).
void SidreDataCollection::
DeregisterFieldInBPIndex(const std::string& field_name)
{
   sidre::DataGroup * fields_grp = bp_index_grp->getGroup("fields");
   MFEM_VERIFY(fields_grp->hasGroup(field_name),
               "No field exists in blueprint index with name " << name);

   // Note: This will destroy all orphaned views or buffer classes under this
   // group also.  If sidre owns this field data, the memory will be deleted
   // unless it's referenced somewhere else in sidre.
   fields_grp->destroyGroup(field_name);
}

void SidreDataCollection::RegisterField(const std::string &field_name,
                                        GridFunction *gf,
                                        const std::string &buffer_name,
                                        axom::sidre::SidreLength offset)
{
   if ( field_name.empty() || buffer_name.empty() ||
        gf == NULL || gf->FESpace() == NULL )
   {
      return;
   }

   // Register field_name in the blueprint group.
   sidre::DataGroup* f = bp_grp->getGroup("fields");

   if (f->hasGroup( field_name ))
   {
      // There are two possibilities:
      // 1. If HasField(field_name) is true - we are overwriting a field that
      //    was previously registered.
      // 2. Otherwise, the field was loaded from a file, or defined outside of
      //    the data collection.
      if (HasField(field_name))
      {
         MFEM_DEBUG_DO(
            MFEM_WARNING("field with the name '" << field_name<< "' is already "
                         "registered, overwriting the old field"));
         DeregisterField(field_name);
      }
   }

   sidre::DataGroup* grp = f->createGroup( field_name );

   // Set the "basis" string using the gf's finite element space, overwrite if
   // necessary.
   sidre::DataView *v = alloc_view(grp, "basis");
   v->setString(gf->FESpace()->FEColl()->Name());

   // Set the topology of the GridFunction.
   // This is always 'mesh' except for a special case with the boundary material
   // attributes field.
   v = alloc_view(grp, "topology")->setString("mesh");

   MFEM_ASSERT(gf->Size() == gf->FESpace()->GetVSize(),
               "GridFunction size does not match FiniteElementSpace size");

   // Set the data views of the grid function
   // e.g. the number of coefficients per DoF -- either scalar-valued or
   // vector-valued
   bool const isScalarValued = (gf->FESpace()->GetVDim() == 1);
   if (isScalarValued)
   {
      // Set the DataView "<bp_grp>/fields/<field_name>/values"
      addScalarBasedGridFunction(field_name, gf, buffer_name, offset);
   }
   else // vector valued
   {
      // Set the DataGroup "<bp_grp>/fields/<field_name>/values"
      addVectorBasedGridFunction(field_name, gf, buffer_name, offset);
   }

   // Register field_name in the blueprint_index group.
   if (myid == 0)
   {
      RegisterFieldInBPIndex(field_name, gf);
   }

   // Register field_name + gf in field_map.
   DataCollection::RegisterField(field_name, gf);
}

void SidreDataCollection::DeregisterField(const std::string& field_name)
{
   // Deregister field_name from field_map.
   DataCollection::DeregisterField(field_name);

   sidre::DataGroup * fields_grp = bp_grp->getGroup("fields");
   MFEM_VERIFY(fields_grp->hasGroup(field_name),
               "No field exists in blueprint with name " << field_name);

   // Delete field_name from the blueprint group.

   // Note: This will destroy all orphaned views or buffer classes under this
   // group also.  If sidre owns this field data, the memory will be deleted
   // unless it's referenced somewhere else in sidre.
   fields_grp->destroyGroup(field_name);

   // Delete field_name from the blueprint_index group.
   if (myid == 0)
   {
      DeregisterFieldInBPIndex(field_name);
   }

   // Delete field_name from the named_buffers group, if allocated.
   FreeNamedBuffer(field_name);
}

// private method
std::string SidreDataCollection::getElementName(Element::Type elementEnum)
{
   // Note -- the mapping from Element::Type to string is based on
   //   enum Element::Type { POINT, SEGMENT, TRIANGLE, QUADRILATERAL,
   //                        TETRAHEDRON, HEXAHEDRON };
   // Note: -- the string names are from conduit's blueprint

   switch (elementEnum)
   {
      case Element::POINT:          return "point";
      case Element::SEGMENT:        return "line";
      case Element::TRIANGLE:       return "tri";
      case Element::QUADRILATERAL:  return "quad";
      case Element::TETRAHEDRON:    return "tet";
      case Element::HEXAHEDRON:     return "hex";
   }

   return "unknown";
}

} // end namespace mfem

#endif
