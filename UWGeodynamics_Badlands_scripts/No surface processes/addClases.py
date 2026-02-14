### The entire code below is sourced from the UWGeodynamics code. Beucher et al., 2018.

### We customized some of this functions to enable voronoi integration and FSSA

import UWGeodynamics as GEO
import numpy as np
from underworld import function as fn
import sys
from mpi4py import MPI as _MPI
comm = GEO.comm
rank = comm.rank
size = comm.size
u = GEO.UnitRegistry
rcParams=GEO.rcParams
import os
import h5py
import underworld as uw
from datetime import datetime
from underworld.utils import _swarmvarschema
from underworld.mesh import FeMesh_Cartesian
from underworld.swarm import Swarm
from underworld.mesh import MeshVariable
from underworld.swarm import SwarmVariable

dimensionalise=GEO.dimensionalise
nd=GEO.nd

def _adjust_time_units(val):
    """ Adjust the units used depending on the value """
    if isinstance(val, u.Quantity):
        mag = val.to(u.years).magnitude
    else:
        val = dimensionalise(val, u.years)
        mag = val.magnitude
    exponent = int("{0:.3E}".format(mag).split("E")[-1])

    if exponent >= 9:
        units = u.gigayear
    elif exponent >= 6:
        units = u.megayear
    elif exponent >= 0:
        units = u.years
    elif exponent > -3:
        units = u.days
    elif exponent > -5:
        units = u.hours
    elif exponent > -7:
        units = u.minutes
    else:
        units = u.seconds
    return val.to(units)


class _CheckpointFunction(object):
    """This Class is responsible for Checkpointing a Model"""

    def __init__(self, Model, duration=None, checkpoint_interval=None,
                 checkpoint_times=None, restart_checkpoint=None,
                 output_units=None):

        self.Model = Model
        self.output_units = output_units
        self.step_type = None

        if isinstance(checkpoint_interval, u.Quantity):
            self.step_type = "time"
            self.checkpoint_interval = nd(checkpoint_interval)
            self.next_checkpoint = Model._ndtime + self.checkpoint_interval

        elif checkpoint_interval:
            self.step_type = "step"
            self.checkpoint_interval = checkpoint_interval
            self.next_checkpoint = Model.stepDone + checkpoint_interval

        self.checkpoint_times = checkpoint_times
        self.restart_checkpoint = restart_checkpoint
        self.outputDir = Model.outputDir

        if checkpoint_interval or checkpoint_times:
            self.checkpoint_all()

    def checkpoint(self):

        Model = self.Model

        if (((self.step_type is "time") and
             (Model._ndtime == self.next_checkpoint)) or
            ((self.step_type is "step") and
             (Model.stepDone == self.next_checkpoint))):

            Model.checkpointID += 1
            # Save Mesh Variables
            self.checkpoint_fields(checkpointID=Model.checkpointID)
            # Save Tracers
            self.checkpoint_tracers(checkpointID=Model.checkpointID)
            self.next_checkpoint += self.checkpoint_interval

            comm.Barrier()

            # if it's time to checkpoint the swarm, do so.
            if Model.checkpointID % self.restart_checkpoint == 0:
                self.checkpoint_swarms(checkpointID=Model.checkpointID)

            comm.Barrier()

    def get_next_checkpoint_time(self):

        Model = self.Model

        dt1 = None
        dt2 = None

        if self.step_type is "time":
            dt1 = self.next_checkpoint - Model._ndtime

        if self.checkpoint_times:
            tcheck = [val - Model._ndtime for val in self.checkpoint_times]
            tcheck = [val for val in tcheck if val >= 0]
            tcheck.sort()
            dt2 = tcheck[0]

        if dt1 and dt2:
            return min(dt1, dt2)
        elif dt1:
            return dt1
        elif dt2:
            return dt2
        else:
            return

    def create_output_directory(self, outputDir=None):

        Model = self.Model

        if not outputDir:
            outputDir = Model.outputDir

        if not os.path.exists(outputDir):
            if rank == 0:
                os.makedirs(outputDir)
        comm.Barrier()

        return outputDir

    def checkpoint_all(self, checkpointID=None, variables=None,
                       tracers=None, time=None, outputDir=None):
        """ Do a checkpoint (Save fields)
        Parameters:
        -----------
            variables:
                list of fields/variables to save
            checkpointID:
                checkpoint ID.
            outpuDir:
                output directory
        """
        self.checkpoint_fields(variables, checkpointID, time, outputDir)
        self.checkpoint_swarms(variables, checkpointID, time, outputDir)
        self.checkpoint_tracers(tracers, checkpointID, time, outputDir)
        comm.Barrier()

    def checkpoint_fields(self, fields=None, checkpointID=None,
                          time=None, outputDir=None):
        """ Save the mesh and the mesh variables to outputDir
        Parameters
        ----------
        fields : A list of mesh/field variables to be saved.
        checkpointID : Checkpoint ID
        time : Model time at checkpoint
        outputDir : output directory
        """

        Model = self.Model

        if not fields:
            fields = rcParams["default.outputs"]

        if not checkpointID:
            checkpointID = Model.checkpointID

        outputDir = self.create_output_directory(outputDir)

        time = time if time else Model.time
        if isinstance(time, u.Quantity) and self.output_units:
            time = time.to(self.output_units)

        if Model._advector or Model._freeSurface:
            mesh_name = 'mesh-%s' % checkpointID
            mesh_prefix = os.path.join(outputDir, mesh_name)
            mH = Model.mesh.save('%s.h5' % mesh_prefix,
                                 units=u.kilometers,
                                 time=time)
        elif not Model._mesh_saved:
            mesh_name = 'mesh'
            mesh_prefix = os.path.join(outputDir, mesh_name)
            mH = Model.mesh.save('%s.h5' % mesh_prefix,
                                 units=u.kilometers,
                                 time=time)
            Model._mesh_saved = True
        else:
            mesh_name = 'mesh'
            mesh_prefix = os.path.join(outputDir, mesh_name)
            mH = uw.utils.SavedFileData(Model.mesh, '%s.h5' % mesh_prefix)

        if rank == 0:
            filename = "XDMF.fields." + str(checkpointID).zfill(5) + ".xmf"
            filename = os.path.join(outputDir, filename)

            # First write the XDMF header
            string = uw.utils._xdmfheader()
            string += uw.utils._spacetimeschema(mH, mesh_name,
                                                time)

        comm.Barrier()

        for field in fields:
            if field == "temperature" and not Model.temperature:
                continue
            if field in Model.mesh_variables.keys():
                field = str(field)

                try:
                    units = rcParams[field + ".SIunits"]
                except KeyError:
                    units = None

                # Save the h5 file and write the field schema for
                # each one of the field variables
                obj = getattr(Model, field)
                file_prefix = os.path.join(outputDir, field + '-%s' % checkpointID)
                handle = obj.save('%s.h5' % file_prefix, units=units,
                                  time=time)
                if rank == 0:
                    string += uw.utils._fieldschema(handle, field)
            comm.Barrier()

        if rank == 0:
            # Write the footer to the xmf
            string += uw.utils._xdmffooter()

            # Write the string to file - only proc 0
            with open(filename, "w") as xdmfFH:
                xdmfFH.write(string)
        comm.Barrier()

    def checkpoint_swarms(self, fields=None, checkpointID=None, time=None,
                          outputDir=None):
        """ Save the swarm and the swarm variables to outputDir
        Parameters
        ----------
        fields : A list of swarm/field variables to be saved.
        checkpointID : Checkpoint ID
        time : Model time at checkpoint
        outputDir : output directory
        """
        Model = self.Model

        if not fields:
            fields = Model.restart_variables

        if not checkpointID:
            checkpointID = Model.checkpointID

        outputDir = self.create_output_directory(outputDir)

        time = time if time else Model.time
        if isinstance(time, u.Quantity) and self.output_units:
            time = time.to(self.output_units)

        swarm_name = 'swarm-%s.h5' % checkpointID

        sH = Model.swarm.save(os.path.join(outputDir,
                              swarm_name),
                              units=u.kilometers,
                              time=time)

        if rank == 0:
            filename = "XDMF.swarms." + str(checkpointID).zfill(5) + ".xmf"
            filename = os.path.join(outputDir, filename)

            # First write the XDMF header
            string = uw.utils._xdmfheader()
            string += uw.utils._swarmspacetimeschema(sH, swarm_name,
                                                     time)
        comm.Barrier()

        for field in fields:
            if field in Model.swarm_variables.keys():
                field = str(field)
                try:
                    units = rcParams[field + ".SIunits"]
                except KeyError:
                    units = None

                # Save the h5 file and write the field schema for
                # each one of the field variables
                obj = getattr(Model, field)
                file_prefix = os.path.join(outputDir,
                                           field + '-%s' % checkpointID)
                handle = obj.save('%s.h5' % file_prefix,
                                  units=units, time=time)
                if rank == 0:
                    string += _swarmvarschema(handle, field)
                comm.Barrier()

        if rank == 0:
            # Write the footer to the xmf
            string += uw.utils._xdmffooter()

            # Write the string to file - only proc 0
            with open(filename, "w") as xdmfFH:
                xdmfFH.write(string)

        comm.Barrier()

    def checkpoint_tracers(self, tracers=None, checkpointID=None,
                           time=None, outputDir=None):
        """ Checkpoint the tracers
        Parameters
        ----------
        tracers : List of tracers to checkpoint.
        checkpointID : Checkpoint ID.
        time : Model time at checkpoint.
        outputDir : output directory
        """

        Model = self.Model

        if not checkpointID:
            checkpointID = Model.checkpointID

        time = time if time else Model.time
        if isinstance(time, u.Quantity) and self.output_units:
            time = time.to(self.output_units)

        if not outputDir:
            outputDir = Model.outputDir

        if rank == 0 and not os.path.exists(outputDir):
            os.makedirs(outputDir)
        comm.Barrier()

        # Checkpoint passive tracers and associated tracked fields
        if Model.passive_tracers:
            for (dump, item) in Model.passive_tracers.items():
                item.save(outputDir, checkpointID, time)

        comm.Barrier()


class _RestartFunction(object):

    def __init__(self, Model, restartDir):

        self.Model = Model
        self.restartDir = restartDir

        comm.Barrier()

    def restart(self, step):
        """restart
        Parameters
        ----------
        step : int
            Step from which you want to restart the model.
            Must be an int (step number either absolute or relative)
            if step == -1, run the last available step
            if step == -2, run the second last etc.
        Returns
        -------
        This function returns None
        """
        Model = self.Model


        indices = self.find_available_steps()
        step = indices[step] if step < 0 else step
        Model.checkpointID = step

        if step not in indices:
            raise ValueError("Cannot find step in specified folder")

        # Get time from swarm-%.h5 file
        if rank == 0:
            swarm_file = os.path.join(self.restartDir, "swarm-%s.h5" % step)
            with h5py.File(swarm_file, "r") as h5f:
                Model._ndtime = nd(u.Quantity(h5f.attrs.get("time")))
        else:
            Model._ndtime = None

        Model._ndtime = comm.bcast(Model._ndtime, root=0)

        if rank == 0:
            print(80 * "=" + "\n")
            print("Restarting Model from Step {0} at Time = {1}\n".format(step, Model.time))
            print('(' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ')')
            print(80 * "=" + "\n")
            sys.stdout.flush()
        comm.Barrier()

        self.reload_mesh(step)
        self.reload_swarm(step)
        Model._initialize()
        self.reload_restart_variables(step)
        self.reload_passive_tracers(step)

        if Model._solver:
            solver_options = Model._solver.options
            Model._solver = None
            Model.solver.options = solver_options

        if isinstance(Model.surfaceProcesses,
                      (surfaceProcesses.SedimentationThreshold,
                       surfaceProcesses.ErosionThreshold,
                       surfaceProcesses.ErosionAndSedimentationThreshold)):

            obj = Model.surfaceProcesses
            obj.Model = Model
            obj.timeField = Model.timeField

        # Restart Badlands if we are running a coupled model
        if isinstance(Model.surfaceProcesses, surfaceProcesses.Badlands):
            self.restart_badlands(step)

        return

    def find_available_steps(self):

        # Look for step with swarm available
        indices = [int(os.path.splitext(filename)[0].split("-")[-1])
                   for filename in os.listdir(self.restartDir) if "-" in
                   filename]
        indices.sort()
        return indices

    def reload_mesh(self, step):

        Model = self.Model

        if Model._advector:
            Model.mesh.load(os.path.join(self.restartDir, 'mesh-%s.h5' % step))
        else:
            Model.mesh.load(os.path.join(self.restartDir, "mesh.h5"))

        if rank == 0:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("Mesh loaded" + '(' + now + ')')
            sys.stdout.flush()

    def reload_swarm(self, step):

        Model = self.Model
        Model.swarm = Swarm(mesh=Model.mesh, particleEscape=True)
        Model.swarm.load(os.path.join(self.restartDir, 'swarm-%s.h5' % step))

        if rank == 0:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("Swarm loaded" + '(' + now + ')')
            sys.stdout.flush()

    def reload_restart_variables(self, step):

        Model = self.Model

        for field in Model.restart_variables:
            obj = getattr(Model, field)
            path = os.path.join(self.restartDir, field + "-%s.h5" % step)
            obj.load(str(path))
            if rank == 0:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{0} loaded".format(field) + '(' + now + ')')
                sys.stdout.flush()

    def reload_passive_tracers(self, step):

        Model = self.Model

        for key, tracer in Model.passive_tracers.items():

            fname = tracer.name + '-%s.h5' % step
            fpath = os.path.join(self.restartDir, fname)
            tracked_fields = tracer.tracked_fields

            obj = PassiveTracers(Model.mesh,
                                 tracer.name,
                                 particleEscape=tracer.particleEscape)
            obj.load(fpath)
            if tracer.advector:
                obj.advector = uw.systems.SwarmAdvector(Model.velocityField, obj, order=2)

            # Reload global indices
            fpath = os.path.join(self.restartDir, tracer.name + '_global_index-%s.h5' % step)
            obj.global_index.load(fpath)

            # Create and Reload all tracked fields
            for name, kwargs in tracked_fields.items():
                field = obj.add_tracked_field(name=name, overwrite=True, **kwargs)
                svar_fname = tracer.name +"_" + name + '-%s.h5' % step
                svar_fpath = os.path.join(self.restartDir, svar_fname)
                field.load(svar_fpath)

            attr_name = tracer.name.lower() + "_tracers"
            setattr(Model, attr_name, obj)
            Model.passive_tracers[key] = obj

            if rank == 0:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("{0} loaded".format(tracer.name) + '(' + now  + ')')
                sys.stdout.flush()

    def restart_badlands(self, step):
        """ Restart Badlands from step
            Note that step is only used if no restartStep has been
            defined on the badlands_model object """

        Model = self.Model

        badlands_model = Model.surfaceProcesses
        restartFolder = badlands_model.restartFolder
        restartStep = badlands_model.restartStep

        if badlands_model.restartFolder:
            restartFolder = badlands_model.restartFolder
        else:
            restartFolder = badlands_model.outputDir

        if badlands_model.restartStep:
            restartStep = badlands_model.restartStep
        else:
            restartStep = step

        # Parse xmf for the last timestep time
        import xml.etree.ElementTree as etree
        xmf = restartFolder + "/xmf/tin.time" + str(restartStep) + ".xmf"
        tree = etree.parse(xmf)
        root = tree.getroot()
        badlands_time = float(root[0][0][0].attrib["Value"])
        uw_time = Model.time.to(u.years).magnitude

        if np.abs(badlands_time - uw_time) > 1:
            raise ValueError("""Time in Underworld and Badlands outputs
                             differs:\n
                             Badlands: {0}\n
                             Underworld: {1}""".format(badlands_time,
                                                       uw_time))

        airIndex = badlands_model.airIndex
        sedimentIndex = badlands_model.sedimentIndex
        XML = badlands_model.XML
        resolution = badlands_model.resolution
        checkpoint_interval = badlands_model.checkpoint_interval

        Model.surfaceProcesses = surfaceProcesses.Badlands(
            airIndex, sedimentIndex,
            XML, resolution,
            checkpoint_interval,
            restartFolder=restartFolder,
            restartStep=restartStep)

        if rank == 0:
            print("Badlands restarted" + '(' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ')')
            sys.stdout.flush()


def _get_output_units(*args):
    from pint import UndefinedUnitError
    for arg in args:
        try:
            return u.Unit(arg)
        except (TypeError, UndefinedUnitError):
            pass
        if isinstance(arg, u.Quantity):
            return arg.units

    return rcParams["time.SIunits"]


def _solver_options_dictionary(solver):
    """Return a dictionary of all the solver options"""
    dd = {}
    for key, val in solver.options.__dict__.items():
        if isinstance(val, dict):
            dd2 = {}
            for key2, val2, in val.__dict__.items():
                dd2[key2] = val2
        else:
            dd2 = val
        dd[key] = dd2
    return dd


def _apply_saved_options_on_solver(solver, options):
    """Apply options on a solver
    solver: solver
    options: python dictionary
    """
    for key, val in options.items():
        if isinstance(val, dict):
            for key2, val2 in val.items():
                solver.options.__dict__[key].__dict__[key2] = val2
        else:
            solver.options.__dict__[key] = val
    return solver


def _adjust_time_units(val):
    """ Adjust the units used depending on the value """
    if isinstance(val, u.Quantity):
        mag = val.to(u.years).magnitude
    else:
        val = dimensionalise(val, u.years)
        mag = val.magnitude
    exponent = int("{0:.3E}".format(mag).split("E")[-1])

    if exponent >= 9:
        units = u.gigayear
    elif exponent >= 6:
        units = u.megayear
    elif exponent >= 0:
        units = u.years
    elif exponent > -3:
        units = u.days
    elif exponent > -5:
        units = u.hours
    elif exponent > -7:
        units = u.minutes
    else:
        units = u.seconds
    return val.to(units)
