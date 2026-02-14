#!/usr/bin/env python
# coding: utf-8

### Arc-continent collision Bdls - script - 10-03-22

import UWGeodynamics as GEO
import numpy as np
from underworld import function as fn
from UWGeodynamics import visualisation as glucifer
from MechanicalProperties import PlateProperties
from ModelGeometry import SubductionCreator,interpolateTracer,rmRepeated,fuseListM,ListToNd,generateWeakzone
import os
import underworld as uw

# %%
# shortcuts for parallel wrappers
barrier = GEO.uw.mpi.barrier
rank    = GEO.rank

#Units
u = GEO.UnitRegistry

########################################
#Rebecca et al., Scaling
dRho =   80. * u.kilogram / u.meter**3 # matprop.ref_density
g    =   9.8 * u.meter / u.second**2   # modprop.gravity
#H    = 800. * u.kilometer #  modprop.boxHeight-- OR 840 dependig if I include the sticky air layer
H= 900. * u.kilometer

# lithostatic pressure for mass-time-length
ref_stress = dRho * g * H
# viscosity of upper mante for mass-time-length
ref_viscosity = 1e20 * u.pascal * u.seconds
#References
ref_time        = ref_viscosity/ref_stress
ref_length      = H
ref_mass        = (ref_viscosity*ref_length*ref_time)
#ref_temperature = modprop.Tint - modprop.Tsurf

KL = ref_length
KM = ref_mass
Kt = ref_time
#KT = ref_temperature

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM
#GEO.scaling_coefficients["[temperature]"] = KT
##########################################

#########################################################
## These are the C parameters of Crameri et al., 2011 for choosing a proper sticky air layer thickness and viscosity

H_sticky=200 #km

# %%
# #Sticky air calcualtion - Stokes
L=900#980
c=(L/H_sticky)**3*(1e20/1e20)*(1/16)*(80/3400)
c

# %%
# #Sticky air calcualtion - Isostacy
L=2600#980
c=(L/H_sticky)**3*(1e20/1e20)*(3/(16*(np.pi**3)))
c
#####################################################

#Model creation
res=(650,300) ## Resolution. This gives 5 km in x and 3 km in Y
#res=(300,250)
#Model Dimensions
Model = GEO.Model(elementRes=res, #296,160 #296,128, #120,64
                  minCoord=(-1300.0 * u.kilometer, -700.0 * u.kilometer),
                  maxCoord=(1300.0* u.kilometer, 200.0 * u.kilometer),
                  gravity=(0.0, -9.81 * u.meter / u.second**2))

#Output folder definition
outputPath="Arc_collision_bdls_2e-5"
Model.outputDir =outputPath


# %%

outputPath = os.path.join(os.path.abspath("."), outputPath)
if rank==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
barrier()


#Model Geometry
#Decoupling Layer- 70 Km for 296, for 120, 110
DipAngle=30
DipLen=300
#original SP length=2400
#SubductionCreator(Model,y0,thickness, dipAngle, dipLength, maxLength, orientation, SLayers,OLayers, ExLens,bStrips,tD=False)
geometry=SubductionCreator(Model,0,100,DipAngle,DipLen,1700,-1,4,2,(210,210,210,210),(40,40,40,40),(True,True, True, True),0)
xcoords = GEO.uw.function.input()[0]
ycoords = GEO.uw.function.input()[1]
#Model limits
orientation=geometry[6][0]
xlimit=GEO.nd(geometry[6][1])
#cratLim=GEO.nd(geometry[8][2])



## Calculating material properties for all materials in the model.

#PlateProperties(Nlayers,crustThickness,crustDensity,mantleDensity,plateThickness,oceanic,age, cohesion,friction,arc,arcDensity,arcThickness,depthToMantle)
#Oceanic Plate
data1=PlateProperties(4,7.,2900.,3400.,100.,True,80.,12.5,12.5,0.066,0.033,False,0.,0.,100.,2.)
#Cratonic continental - Values can be +5MPa!!
data2=PlateProperties(2,40.,2700.,3400.,150.,False,80.,15.,10.,0.15,0.08,False,0.,0.,150.,.2)
#Back-arc
data3=PlateProperties(2,20.,2800.,3400.,100.,False,80.,10.,7.5,0.005,0.0035,False,0.,0.,80.,2.) #the last specifies how much was extended the lithosphere
#Arc Crust
data4=PlateProperties(4,7.,2900.,3400.,100.,True,80.,12.5,12.5,0.066,0.033,True,2838.,25.,100.,2.)
#eclogite-Properties
data5=PlateProperties(4,2.,3450.,3400.,100.,True,80.,12.5,12.5,0.066,0.033,False,0.,0.,100.,2.)
#Transitional crust
data6=PlateProperties(2,25.,2700.,3400.,100.,False,80.,12.5,12.5,0.066,0.033,False,0.,0.,100.,2.)
# %%


#Densities without units
refD=3400.* u.kilogram / u.metre**3 #Mantle Density
#(Subducting plate)
l1d=data1[2][0]* u.kilogram / u.metre**3
l2d=data1[2][1]* u.kilogram / u.metre**3
l3d=data1[2][2]* u.kilogram / u.metre**3
l4d=data1[2][3]* u.kilogram / u.metre**3
#Cratonic overriding plate
cl1=data2[2][0]* u.kilogram / u.metre**3
cl2=data2[2][1]* u.kilogram / u.metre**3
#Back-Arc overriding plate
bl1=data3[2][0]* u.kilogram / u.metre**3
bl2=data3[2][1]* u.kilogram / u.metre**3
#Arc density
al1=data4[2][0]* u.kilogram / u.metre**3
al2=data4[2][1]* u.kilogram / u.metre**3
#Oceanic crust to Eclogite
e1d=data5[2][0]* u.kilogram / u.metre**3
#e1d= 3450* u.kilogram / u.metre**3
#e1d=3363.51575* u.kilogram / u.metre**3
#Lower Mantle density - 1.6 +-1 to 5+-2 percentage the upper mantle density - (1.6+7)/2/100 - 3400.+(3400*0.043)
#LMdensity=3546.2* u.kilogram / u.metre**3
LMdensity=3488.4* u.kilogram / u.metre**3
#transition_crust
tra1=data6[2][0]* u.kilogram / u.metre**3
tra2=data6[2][1]* u.kilogram / u.metre**3

## Generate a weak zone that will enable subduction initiation
weakShape=generateWeakzone(geometry[6][1],0*u.kilometer, 30,DipAngle, DipLen) #This cointains an initial plastic damage of 0.2
weakShape2=generateWeakzone(geometry[6][1],0*u.kilometer, 250,DipAngle, DipLen)  #This allows plastic strain around the trench

## Calculate Subduction geometry Geometry
#SubductionCreator(Model,y0,thickness, dipAngle, dipLength, maxLength, orientation, SLayers,OLayers, ExLens,bStrips,tD=False)
#return subducting,overriding, weak,arc,bstop

#Add materials to the model
Air_shape=GEO.shapes.Layer(top=200.*u.kilometer, bottom=0.*u.kilometer)
Air=Model.add_material(name="Air", shape=Air_shape)
UMantle =Model.add_material(name="UpperMantle", shape=GEO.shapes.Layer(top=0.*u.kilometer, bottom=-700.*u.kilometer))
OLithosphere1F = Model.add_material(name="SubductingPlateL1", shape=geometry[0][2][0])
OLithosphere2F=Model.add_material(name="SubductingPlateL2", shape=geometry[0][2][1])
OLithosphere3F=Model.add_material(name="SubductingPlateL3", shape=geometry[0][2][2])
OLithosphere4F=Model.add_material(name="SubductingPlateL4", shape=geometry[0][2][3])
Clithosphere1= Model.add_material(name="Overriding plate Crust", shape=geometry[1][0])
Clithosphere2= Model.add_material(name="Overriding plate Lithosphere", shape=geometry[1][1])
Clithosphereweak1=Model.add_material(name="Overriding plate Weak Crust", shape=geometry[2][0])
Clithosphereweak2=Model.add_material(name="Overriding plate Weak Lithosphere", shape=geometry[2][1])
OArc1=Model.add_material(name="IntraOceanicArc", shape=geometry[3][0])
Ctransitional1=Model.add_material(name="OP_transitional1", shape=geometry[12][0])
Ctransitional2=Model.add_material(name="OP_transitional2", shape=geometry[12][1])
sediment   = Model.add_material(name="Sediment")

# The shape objetcs below will be used to remove plastic strain from model walls. This causes plastic strain to localize in the collisional zone, where it should
#Strain-backstop OP plate shapes
back1=geometry[11][0]
back2=geometry[11][1]
back3=geometry[11][2]
back4=geometry[11][3]

#Transitional crust shapes
ctra1=geometry[12][0]
ctra2=geometry[12][1]
#cratonic continental plate shapes
cratS1=geometry[1][0]
cratS2=geometry[1][1]

# %%


#Preview of 2D materials-Materials Field (from swarm) - Glucifer plot
Fig = glucifer.Figure(figsize=(1200,400))
Fig.Points(Model.swarm, Model.materialField,fn_size=2.0, discrete=True)
#Fig.Surface(Model.mesh,Model.projMaterialField,fn_size=2.0)
Fig.show()


# %%


#eclogite density fn
conditions=[(Model.y > GEO.nd(-150.*u.kilometer),GEO.nd(l1d)),
            (True,GEO.nd(e1d)),
    
]
eclogiteFn=fn.branching.conditional(conditions)

#avoid continental crust reaching --400 km to come back surface

conditionsSubCrust=[(Model.y > GEO.nd(-400.*u.kilometer),GEO.nd(bl1)),
            (True,GEO.nd(refD)),
    
]

#avoid air-mantle contact - All mantle material reaching the surface becomes basalt
conditionsAvoid=[(((Model.x > GEO.nd(0.*u.kilometer)) & (Model.y > GEO.nd(-5.*u.kilometer))),GEO.nd(l1d)),

            (True,GEO.nd(refD)),
    
]
AvoidFn=fn.branching.conditional(conditionsAvoid)


#all continental crust below 400 becmes mantle material to avoid chucks of continental crust to ascend through the mantle
CrustFn=fn.branching.conditional(conditionsSubCrust)

#initial plastic Strain - This function "damages" the SP-OP interface to facilitate subduction initiation
plasticCondition=[(Model.y<GEO.nd(-100.*u.kilometer),GEO.nd(0.)),(weakShape,GEO.nd(0.2)),(True,GEO.nd(0.))]
plasticFn=fn.branching.conditional(plasticCondition)
fact=plasticFn.evaluate(Model.swarm)
Model.plasticStrain.data[:] += fact


#Plastic strain healing functions

#no-pin SP - if SP is allowed to move and is not attached from the boundary
coordianteSP=-1050*u.kilometer

SPheal=[(back1.vertices[0][0],back1.vertices[0][1]+10*u.kilometer),
        (coordianteSP,back1.vertices[0][1]+10*u.kilometer),
        (coordianteSP,back4.vertices[3][1]),
        (back4.vertices[3][0],back4.vertices[3][1])
]

SPheal=GEO.shapes.Polygon(SPheal)

#This heal the plastic damage in the overriding plate at the right wall every time step

coordinateOP=920*u.kilometer

OPheal=[(Model.maxCoord[0],ctra1.vertices[0][1]+10*u.kilometer),
        (coordinateOP,ctra1.vertices[0][1]+10*u.kilometer),
        (coordinateOP,-160.*u.kilometer),
        (Model.maxCoord[0],-160.*u.kilometer)
       ]
OPheal=GEO.shapes.Polygon(OPheal)


#To keep plastic strain as cero at the model walls. The rear of the SP and OP

InfCondPlastic=[(back1,GEO.nd(0.)),
               (back2,GEO.nd(0.)),
               (back3,GEO.nd(0.)),
               (back4,GEO.nd(0.)),
               (OPheal,GEO.nd(0.)),
               (SPheal,GEO.nd(0.)),
              # (cratS1,GEO.nd(0.)),
              # (cratS2,GEO.nd(0.)),
               (True,GEO.nd(1.)),
               ]

InfplasticFn=fn.branching.conditional(InfCondPlastic)

#This created a log file with all calculated UW time steps
import underworld.function as fn

fout = outputPath+'/FrequentOutput.dat'
if rank == 0:
    with open(fout,'a') as f:
         f.write('#step\t time(yr)\t Vrms(cm/yr)\n') #

   
    
#Increased distance by 10 km

## Phases transitions of mantle to oceanic lithosphere. This makes the Subducting plate "infinite"
UMantle.phase_changes = GEO.PhaseChange(back1,
                                          OLithosphere1F.index)

UMantle.phase_changes = GEO.PhaseChange(back2,
                                          OLithosphere2F.index)

UMantle.phase_changes = GEO.PhaseChange(back3,
                                          OLithosphere3F.index)

UMantle.phase_changes = GEO.PhaseChange(back4,
                                          OLithosphere4F.index)


## Asigning Density to Materials - Includes relative density calculation for arc
Air.density = 1000. * u.kilogram / u.metre**3 #air
UMantle.density = AvoidFn   #Upper Mantle
OLithosphere1F.density =eclogiteFn  #Subducting plate layer 1
OLithosphere2F.density =l2d   #Subducting plate layer 2
OLithosphere3F.density =l3d   #Subducting plate layer 3
OLithosphere4F.density =l4d   #Subducting plate layer 4
Clithosphere1.density=cl1      #Continental Craton layer 1
Clithosphere2.density=cl2       #Continental Craton layer 2
Clithosphereweak1.density=CrustFn   #Continental plate layer 1
Clithosphereweak2.density=bl2       #Continental plate layer 2
Ctransitional1.density=tra1      #Transitional Continental plate layer 1
Ctransitional2.density=tra2      #Transitional Continental plate layer 2
OArc1.density=al1              #Intra-oceanic arc
sediment.density=CrustFn      #Sediment


#Density Field - Glucifer plot
Fig = glucifer.Figure(figsize=(1200,400))
#Fig.Points(Model.swarm, GEO.Dimensionalize(Model.densityField, u.kilogram / u.metre**3))
Fig.Surface(Model.mesh, GEO.Dimensionalize(Model.projDensityField, u.kilogram / u.metre**3))
Fig.show()


#Viscosities without units to asign to materials later
refV=1e20 #UMantle as reference Viscosity
#(Subducting plate)
vl1d=data1[3][0] #layer 1
vl2d=data1[3][1] #layer 2
vl3d=data1[3][2] #layer 3
vl4d=data1[3][3] #layer 4
#Cratonic overriding plate
vcl1=data2[3][0] #layer 1
vcl2=data2[3][1] #layer 2
#Back-Arc overriding plate
vbl1=data3[3][0] #layer 1
vbl2=data3[3][1] #layer 2
#Arc viscosity -- Consider what discussed in Len & Gurnis, 2015 (lower-middle crust is very weak)
val1=data4[3][0] #layer 1
val2=data4[3][1] #layer 2
# #Eclogite viscosity
# ve1=data5[3][0]
#Vicosity transition continental plate
tra1V=data6[3][0] #layer 1
tra2V=data6[3][1] #layer 2


# Maximum and minimum viscosity in the model. Also, maximum stress allowed in the model
Model.maxViscosity  = 1e23 * u.pascal * u.second # Maximum viscosity of materials allowed in the model domain
Model.minViscosity=   1e18 * u.pascal * u.second # Minimum viscosity of materials allowed in the model domain
Model.stressLimiter = 300. * u.megapascal # Maximum stress allowed in the model domain

# Asignation of viscosity to Materials
Air.viscosity=1e18 * u.pascal * u.second #air
UMantle.viscosity =  refV * u.pascal * u.second  #Upper mantle as reference Viscosity
OLithosphere1F.viscosity = vl1d* u.pascal * u.second #Subducting plate layer 1
OLithosphere2F.viscosity = vl2d* u.pascal * u.second #Subducting plate layer 2
OLithosphere3F.viscosity = vl3d* u.pascal * u.second #Subducting plate layer 3
OLithosphere4F.viscosity = vl4d* u.pascal * u.second  #Subducting plate layer 4
Clithosphere1.viscosity= vcl1 * u.pascal * u.second  #Continental Craton layer 1
Clithosphere2.viscosity= vcl2 *u.pascal * u.second   #Continental Craton layer 2
Clithosphereweak1.viscosity= vbl1 * u.pascal * u.second   #Continental plate layer 1
Clithosphereweak2.viscosity= vbl2* u.pascal * u.second# #Continental plate layer 2
OArc1.viscosity= val1 * u.pascal * u.second    #Intra-oceanic arc
Ctransitional1.viscosity=tra1V * u.pascal * u.second #Transitional Continental plate layer 1
Ctransitional2.viscosity=tra2V * u.pascal * u.second  #Transitional Continental plate layer 2
sediment.viscosity=vbl1 * u.pascal * u.second  #Sediment


# %%


#Minimum Viscosity for materials
OLithosphere1F.minViscosity =10**(21) * u.pascal * u.second
OLithosphere2F.minViscosity = 10**(21) * u.pascal * u.second
OLithosphere3F.minViscosity = 10**(21) * u.pascal * u.second
OLithosphere4F.minViscosity = 10**(21) * u.pascal * u.second
Clithosphereweak1.minViscosity= 10**(21) * u.pascal * u.second
Clithosphereweak2.minViscosity=10**(21) * u.pascal * u.second
OArc1.minViscosity=10**(20) * u.pascal * u.second
sediment.minViscosity=10**(20) * u.pascal * u.second


# %%


#Viscosity Field - Glucifer plot
Fig = glucifer.Figure(figsize=(1200,400))
Fig.Points(Model.swarm, GEO.Dimensionalize(Model.viscosityField, u.pascal * u.second),logScale=True)
Fig.show()


# %%

# %%
pl=GEO.PlasticityRegistry()

# %%

dir(pl)

# %%
pl.Rey_and_Muller_2010_UpperCrust

# %%
pl.Rey_et_al_2014_UpperCrust

#Plasticty - we use drucker-pragger to implement the plasticity of all materials

# Drucker-Prager-
OLithosphere1F.plasticity = GEO.DruckerPrager(cohesion=12.5 * u.megapascal,
                                            frictionCoefficient=0.008,
                                              #cohesionAfterSoftening=12.5/2. * u.megapascal,
                                              frictionAfterSoftening=0.00001, #Serpentinization (shallow 25 Km)
                                             epsilon1=0.0,
                                             epsilon2=0.2)
OLithosphere2F.plasticity = GEO.DruckerPrager(cohesion=12.5 * u.megapascal,
                                              frictionCoefficient=0.05,
                                              #cohesionAfterSoftening=12.5/2. * u.megapascal,
                                              frictionAfterSoftening=0.04, #to 0.0035 ? from 0.0027
                                             epsilon1=0.0,
                                             epsilon2=0.2)
OLithosphere3F.plasticity =GEO.DruckerPrager(cohesion=12.5 * u.megapascal,
                                             frictionCoefficient= 0.01,
                                             cohesionAfterSoftening=12.5/2. * u.megapascal,
                                             frictionAfterSoftening=0.0065,
                                            epsilon1=0.0,
                                            epsilon2=0.2)
Clithosphere1.plasticity =GEO.DruckerPrager(cohesion=15. * u.megapascal,  #Mean from Lower and Upper from Rey et al., 2014
                                            frictionCoefficient=0.07,
                                       cohesionAfterSoftening=15./2. * u.megapascal,
                                       frictionAfterSoftening=0.05,
                                            epsilon1=0.0, epsilon2=0.2)
Clithosphere2.plasticity = GEO.DruckerPrager(cohesion=10. * u.megapascal,
                                             frictionCoefficient=0.04, #Mean from Lower and Upper from Rey et al., 2014
                                       cohesionAfterSoftening=5. * u.megapascal,
                                       frictionAfterSoftening=0.02,
                                             epsilon1=0.0, epsilon2=0.2)
Clithosphereweak1.plasticity = GEO.DruckerPrager(cohesion=15. * u.megapascal,
                                                 frictionCoefficient=0.0065,#Mean from Lower and Upper from Rey et al., 2014
                                       cohesionAfterSoftening=4. * u.megapascal,
                                                 frictionAfterSoftening=0.0001,
                                                 epsilon1=0.0, epsilon2=0.2)
Clithosphereweak2.plasticity = GEO.DruckerPrager(cohesion=10. * u.megapascal,
                                                 frictionCoefficient=0.0075,#Mean from Lower and Upper from Rey et al., 2014
                                       cohesionAfterSoftening=3.5 * u.megapascal,
                                                 frictionAfterSoftening=0.0045,
                                                epsilon1=0.0, epsilon2=0.2)   #Friction to zero due to weakening (Patrice personal and Len & Gurnis, 2015)
OArc1.plasticity = GEO.DruckerPrager(cohesion=12.5 * u.megapascal,
                                            frictionCoefficient=0.008,
                                              #cohesionAfterSoftening=12.5/2. * u.megapascal,
                                              frictionAfterSoftening=0.00001, #Serpentinization (shallow 25 Km)
                                             epsilon1=0.0,
                                             epsilon2=0.20)
                                             
Ctransitional1.plasticity =GEO.DruckerPrager(cohesion=15. * u.megapascal,  #Mean from Lower and Upper from Rey et al., 2014
                                            frictionCoefficient=0.05,
                                       cohesionAfterSoftening=15./2. * u.megapascal,
                                       frictionAfterSoftening=0.009,
                                            epsilon1=0.0, epsilon2=0.2)

Ctransitional2.plasticity =GEO.DruckerPrager(cohesion=10. * u.megapascal,
                                             frictionCoefficient=0.04, #Mean from Lower and Upper from Rey et al., 2014
                                       cohesionAfterSoftening=5. * u.megapascal,
                                       frictionAfterSoftening=0.008,
                                             epsilon1=0.0, epsilon2=0.2)
                                             
sediment.plasticity = GEO.DruckerPrager(cohesion=15. * u.megapascal,
                                                 frictionCoefficient=0.0055,#Mean from Lower and Upper from Rey et al., 2014
                                       cohesionAfterSoftening=4. * u.megapascal,
                                                 frictionAfterSoftening=0.00001,
                                                 epsilon1=0.0, epsilon2=0.2)


 #Passive Tracers- To track plate convergence/retreat rates
SPTracersC=rmRepeated(geometry[7])
OPTracersC=rmRepeated(geometry[8])
ArcTracersC=rmRepeated(geometry[9])
CratonTracersC=rmRepeated(geometry[10])
#Map for vertexes
SPMap=[(0,1),(1,2),(5,4),(4,3),(8,7),(7,6),(11,10),(10,9),(14,13),(13,12)]
# FlatMap=[(0,1),(5,4),(8,7),(11,10),(14,13)]
# DipMap=[(1,2),(4,3),(7,6),(10,9),(13,12)]
OPMap=[(0,1),(2,3),(4,5)]
ArcMap=[(0,1),(2,3),(4,5)]
CratMap=[(0,1),(2,3),(4,5)]

SP=[]
wise=0
for i in SPMap:
    if wise==0:
        npoints=500
        wise=1
    else:
        npoints=100
    aux=i
    SP.append(interpolateTracer([SPTracersC[0][aux[0]],SPTracersC[1][aux[0]]],

                         [SPTracersC[0][aux[1]],SPTracersC[1][aux[1]]],1000))
OP=[]
aux=OPMap[0]
OP.append(interpolateTracer([OPTracersC[0][aux[0]],OPTracersC[1][aux[0]]],

                         [OPTracersC[0][aux[1]],OPTracersC[1][aux[1]]],1000))


#Tracers at -3km depth

p=interpolateTracer([OPTracersC[0][aux[0]],OPTracersC[1][aux[0]]],

                         [OPTracersC[0][aux[1]],OPTracersC[1][aux[1]]],1000)
pp=np.asarray(p)
pp[1,:]=pp[1,:]-3

OP.append(pp* u.kilometer)

#Tracers at -6km depth
p=interpolateTracer([OPTracersC[0][aux[0]],OPTracersC[1][aux[0]]],

                         [OPTracersC[0][aux[1]],OPTracersC[1][aux[1]]],1000)
pp=np.asarray(p)
pp[1,:]=pp[1,:]-6

OP.append(pp* u.kilometer)

#Tracers at -9km depth
p=interpolateTracer([OPTracersC[0][aux[0]],OPTracersC[1][aux[0]]],

                         [OPTracersC[0][aux[1]],OPTracersC[1][aux[1]]],1000)
pp=np.asarray(p)
pp[1,:]=pp[1,:]-9

OP.append(pp* u.kilometer)

######-50 km
aux=OPMap[1]
OP.append(interpolateTracer([OPTracersC[0][aux[0]],OPTracersC[1][aux[0]]],

                         [OPTracersC[0][aux[1]],OPTracersC[1][aux[1]]],1000))
aux=OPMap[2]
OP.append(interpolateTracer([OPTracersC[0][aux[0]],OPTracersC[1][aux[0]]],

                         [OPTracersC[0][aux[1]],OPTracersC[1][aux[1]]],1000))
OP_Crat=[]
aux=CratMap[0]
OP_Crat.append(interpolateTracer([CratonTracersC[0][aux[0]],CratonTracersC[1][aux[0]]],

                         [CratonTracersC[0][aux[1]],CratonTracersC[1][aux[1]]],400))
aux=CratMap[1]
OP_Crat.append(interpolateTracer([CratonTracersC[0][aux[0]],CratonTracersC[1][aux[0]]],

                         [CratonTracersC[0][aux[1]],CratonTracersC[1][aux[1]]],400))
aux=CratMap[2]
OP_Crat.append(interpolateTracer([CratonTracersC[0][aux[0]],CratonTracersC[1][aux[0]]],

                         [CratonTracersC[0][aux[1]],CratonTracersC[1][aux[1]]],400))

ArcT=[]
aux=ArcMap[0]
ArcT.append(interpolateTracer([ArcTracersC[0][aux[0]],ArcTracersC[1][aux[0]]],

                         [ArcTracersC[0][aux[1]],ArcTracersC[1][aux[1]]],500))
#####
#Tracers at -3km depth

p=interpolateTracer([ArcTracersC[0][aux[0]],ArcTracersC[1][aux[0]]],

                         [ArcTracersC[0][aux[1]],ArcTracersC[1][aux[1]]],500)
pp=np.asarray(p)
pp[1,:]=pp[1,:]-3

ArcT.append(pp* u.kilometer)

#Tracers at -6km depth
p=interpolateTracer([ArcTracersC[0][aux[0]],ArcTracersC[1][aux[0]]],

                         [ArcTracersC[0][aux[1]],ArcTracersC[1][aux[1]]],500)
pp=np.asarray(p)
pp[1,:]=pp[1,:]-6

ArcT.append(pp* u.kilometer)

#Tracers at -9km depth
p=interpolateTracer([ArcTracersC[0][aux[0]],ArcTracersC[1][aux[0]]],

                         [ArcTracersC[0][aux[1]],ArcTracersC[1][aux[1]]],500)
pp=np.asarray(p)
pp[1,:]=pp[1,:]-9

ArcT.append(pp* u.kilometer)
#####

aux=ArcMap[1]
ArcT.append(interpolateTracer([ArcTracersC[0][aux[0]],ArcTracersC[1][aux[0]]],

                         [ArcTracersC[0][aux[1]],ArcTracersC[1][aux[1]]],500))

aux=ArcMap[2]
ArcT.append(interpolateTracer([ArcTracersC[0][aux[0]],ArcTracersC[1][aux[0]]],

                         [ArcTracersC[0][aux[1]],ArcTracersC[1][aux[1]]],500))
#Fusing Data
# SP_F=fuseListM([SP_F[0],SP_F[1],SP_F[2],SP_F[3],SP_F[4]])
SP=fuseListM([SP[0],SP[1],SP[2],SP[3],SP[4],SP[5],SP[6],SP[7]])
OP=fuseListM([OP[0],OP[1],OP[2],OP[2],OP[3],OP[4],OP[5]])
ArcT=fuseListM([ArcT[0],ArcT[1],ArcT[2],ArcT[3],ArcT[4],ArcT[5]])
CratonT=fuseListM([OP_Crat[0],OP_Crat[1],OP_Crat[2]])


def to2Darray(array_x,array_y):
    tracers=np.zeros((len(array_x),2))
    counter=0
    for i,j in zip(array_x,array_y):
        #print (counter)
        tracers[counter][0]=GEO.nd(i.magnitude* u.kilometer)
        tracers[counter][1]=GEO.nd(j.magnitude* u.kilometer)
        #print(i,j, len(tracers))
        counter=counter+1
    return tracers

#Adding passive tracers - Working FSSA version
Model.add_passive_tracers(name="SPTracers",vertices=to2Darray(SP[0],SP[1]))
Model.add_passive_tracers(name="OPTracers",vertices=to2Darray((OP[0]),(OP[1])))
Model.add_passive_tracers(name="ArcTracers",vertices=to2Darray(((ArcT[0])),((ArcT[1]))))
Model.add_passive_tracers(name="CratonTracers",vertices=to2Darray(((CratonT[0])),((CratonT[1]))))

#Fields to Record

Model.SPTracers_tracers.add_tracked_field(Model.velocityField[0],
                              name="Subducting plate velocity_X",
                              units=u.centimeter/ u.year,
                              dataType="float")
Model.SPTracers_tracers.add_tracked_field(Model.velocityField[1],
                              name="Subducting plate velocity_Y",
                              units=u.centimeter/ u.year,
                              dataType="float")
Model.ArcTracers_tracers.add_tracked_field(Model.velocityField[0],
                              name="Arc nodes velocity_X",
                              units=u.centimeter/ u.year,
                              dataType="float")
Model.ArcTracers_tracers.add_tracked_field(Model.plasticStrain,
                              name="arc plastic strain",
                              units=u.meter/u.meter,
                              dataType="float")
Model.ArcTracers_tracers.add_tracked_field(Model.velocityField[1],
                              name="Arc nodes velocity_Y",
                              units=u.centimeter/ u.year,
                              dataType="float")
#Model.ArcTracers_tracers.add_tracked_field(Model._stressTensor[0],
#                              name="Arc stress tensor_X",
#                              units=u.megapascal,
#                              dataType="float")
#Model.ArcTracers_tracers.add_tracked_field(Model._stressTensor[1],
#                              name="Arc stress tensor_Y",
#                              units=u.megapascal,
#                              dataType="float")
#Model.ArcTracers_tracers.add_tracked_field(Model._stressTensor[2],
#                              name="Arc stress tensor_XY",
#                              units=u.megapascal,
#                              dataType="float")
Model.ArcTracers_tracers.add_tracked_field(Model.strainRateField,
                              name="Arc strain rate",
                              units=1./ u.second,
                              dataType="float")
#Model.ArcTracers_tracers.add_tracked_field(Model.projStressField[0],
#                              name="Arc stress Field",
#                              units=u.megapascal,
#                              dataType="float")
Model.ArcTracers_tracers.add_tracked_field(Model.viscosityField,
                              name="Arc Viscosity",
                              units=u.pascal * u.second,
                              dataType="float")
Model.OPTracers_tracers.add_tracked_field(Model.velocityField[0],
                              name="Weak overriding plate velocity_X",
                              units=u.centimeter/ u.year,
                              dataType="float")
Model.OPTracers_tracers.add_tracked_field(Model.velocityField[1],
                              name="Weak overriding plate velocity_Y",
                              units=u.centimeter/ u.year,
                              dataType="float")
Model.OPTracers_tracers.add_tracked_field(Model.strainRateField,
                              name="Weak overriding plate strain rate",
                              units=1./ u.second,
                              dataType="float")
#Model.OPTracers_tracers.add_tracked_field(Model.projStressField[0],
#                              name="Weak overriding plate stress Field",
#                              units=u.megapascal,
#                              dataType="float")
Model.OPTracers_tracers.add_tracked_field(Model.plasticStrain,
                              name="Weak overriding plate plastic strain",
                              units=u.meter/u.meter,
                              dataType="float")
#Model.OPTracers_tracers.add_tracked_field(Model.projStressTensor[0],
#                              name="Weak overriding plate stress tensor_X",
#                              units=u.megapascal,
#                              dataType="float")
Model.OPTracers_tracers.add_tracked_field(Model.viscosityField,
                              name="Weak overriding plate Viscosity",
                              units=u.pascal * u.second,
                              dataType="float")
#Model.OPTracers_tracers.add_tracked_field(Model._stressTensor[1],
#                              name="Weak overriding plate stress tensor_Y",
#                              units=u.megapascal,
#                              dataType="float")
#Model.OPTracers_tracers.add_tracked_field(Model._stressTensor[2],
#                              name="Weak overriding plate stress tensor_XY",
#                              units=u.megapascal,
#                              dataType="float")
#Model.SPTracers_tracers.add_tracked_field(Model.timeField,
#                              name="Time_SP",
#                              units=u.megayear,
#                              dataType="float")
#Model.OPTracers_tracers.add_tracked_field(Model.timeField,
#                              name="Time_OP",
#                              units=u.megayear,
#                              dataType="float")
#Model.CratonTracers_tracers.add_tracked_field(Model.timeField,
#                              name="Time_SP",
#                              units=u.megayear,
#                              dataType="float")
Model.CratonTracers_tracers.add_tracked_field(Model.velocityField[0],
                              name="Cratonic overriding plate velocity_X",
                              units=u.centimeter/ u.year,
                              dataType="float")
Model.CratonTracers_tracers.add_tracked_field(Model.velocityField[1],
                              name="Cratonic overriding plate velocity_Y",
                              units=u.centimeter/ u.year,
                              dataType="float")
#Model.CratonTracers_tracers.add_tracked_field(Model._stressTensor[0],
#                              name="Craton stress tensor_X",
#                              units=u.megapascal,
#                              dataType="float")
#Model.CratonTracers_tracers.add_tracked_field(Model._stressTensor[1],
#                              name="Craton stress tensor_Y",
#                              units=u.megapascal,
#                              dataType="float")
#Model.CratonTracers_tracers.add_tracked_field(Model._stressTensor[2],
#                              name="Craton stress tensor_XY",
#                              units=u.megapascal,
#                              dataType="float")
#Model.ArcTracers_tracers.add_tracked_field(Model.timeField,
#                              name="Time_Arc",
#                              units=u.megayear,
#                              dataType="float")


#passive tracers visualization

Fig = glucifer.Figure(figsize=(1200,400))
Fig.Points(Model.SPTracers_tracers, pointSize=5.0)
Fig.Points(Model.OPTracers_tracers, pointSize=5.0)
Fig.Points(Model.ArcTracers_tracers, pointSize=5.0)
Fig.Points(Model.CratonTracers_tracers, pointSize=5.0)
Fig.Points(Model.swarm, Model.materialField, fn_size=3.0)
Fig.show()

# %%



#Free-slip Boundary Conditions (Kinematic BCs)
Model.set_velocityBCs(left=[0., None],
                     right=[0.,None],
                     bottom=[0., 0.],
                     top=[None, 0.],
                    nodeSets = [(SPheal, [None,GEO.nd(0. * u.centimetre/u.year)])
                    ])




#FSSA ALgorithm activation
Model.fssa_factor=0.5

if rank == 0: print("Calling init_model()...")
Model.init_model()

######### Customized functions from UWGeo to make work Voronoi integration. Also to be able to re-start a model using voronoi integration.
### Without Voronoi integration the velocity filed becames very unestable
#####From UW Source Code. Beucher et al., 2018

def _get_output_units(*args):
    from pint import UndefinedUnitError
    for arg in args:
        try:
            return u.Unit(arg)
        except (TypeError, UndefinedUnitError):
            pass
        if isinstance(arg, u.Quantity):
            return arg.units

    return GEO.rcParams["time.SIunits"]


from addClases import _CheckpointFunction
from addClases import _adjust_time_units
    

import sys
from mpi4py import MPI as _MPI
comm = GEO.comm
rank = comm.rank
size = comm.size
from underworld.utils import _swarmvarschema
#from .version import full_version
from datetime import datetime

full_version=GEO.version.full_version

def _initializeS(self):
    """_initialize
    Model Initialisation
    """
    rcParams=GEO.rcParams
    self.add_mesh_variable("tractionField", nodeDofCount=self.mesh.dim)

    self.swarm_advector = uw.systems.SwarmAdvector(
        swarm=self.swarm,
        velocityField=self.velocityField,
        order=2
    )
    
    print(rcParams["popcontrol.particles.per.cell.2D"])
    
    if self.mesh.dim == 2:
        #particlesPerCell = rcParams["popcontrol.particles.per.cell.2D"]
        particlesPerCell = 100
    else:
        particlesPerCell = rcParams["popcontrol.particles.per.cell.3D"]

    self.population_control = uw.swarm.PopulationControl(
        self.swarm,
        aggressive=rcParams["popcontrol.aggressive"],
        splitThreshold=rcParams["popcontrol.split.threshold"],
        maxSplits=rcParams["popcontrol.max.splits"],
        particlesPerCell=particlesPerCell)

    # Add Common Swarm Variables
    self.add_swarm_variable("materialField", dataType="int", count=1,
                            restart_variable=True, init_value=self.index)
    self.add_swarm_variable("plasticStrain", dataType="double", count=1,
                            restart_variable=True)
    self.add_swarm_variable("_viscosityField", dataType="double", count=1)
    self.add_swarm_variable("_densityField", dataType="double", count=1)
    self.add_swarm_variable("meltField", dataType="double", count=1)
    self.add_swarm_variable("timeField", dataType="double", count=1,
                            restart_variable=True)
    self.timeField.data[...] = 0.0
    self.materialField.data[...] = self.index

    if self.mesh.dim == 3:
        stress_dim = 6
    else:
        stress_dim = 3

    self.add_swarm_variable("_previousStressField", dataType="double",
                            count=stress_dim)
    self.add_swarm_variable("_stressTensor", dataType="double",
                            count=stress_dim, projected="submesh")
    self.add_swarm_variable("_stressField", dataType="double",
                            count=1, projected="submesh")


from addClases import _CheckpointFunction
from addClases import _adjust_time_units

def _update(self):
    """ Update Function
    The following function processes the mesh and swarm variables
    between two solves. It takes care of mesh, swarm advection and
    update the fields according to the Model state.
    """

    dt = self._dt

    # Heal plastic strain
    if any([material.healingRate for material in self.materials]):
        healingRates = {}
        for material in self.materials:
            healingRates[material.index] = nd(material.healingRate)
        HealingRateFn = fn.branching.map(fn_key=self.materialField,
                                         mapping=healingRates)

        plasticStrainIncHealing = dt * HealingRateFn.evaluate(self.swarm)
        self.plasticStrain.data[:] -= plasticStrainIncHealing
        self.plasticStrain.data[self.plasticStrain.data < 0.] = 0.

    # Increment plastic strain
    _isYielding = self._viscosity_processor._isYielding
    plasticStrainIncrement = dt * _isYielding.evaluate(self.swarm)
    self.plasticStrain.data[:] += plasticStrainIncrement

    if any([material.melt for material in self.materials]):
        # Calculate New meltField
        self.update_melt_fraction()

    # Solve for temperature
    try:
        if self.temperature:
            self._advdiffSystem.integrate(dt)
    except ValueError:
        pass

    if self._advector:
        self.swarm_advector.integrate(dt)
        self._advector.advect_mesh(dt)
    elif self._freeSurface:
        self.swarm_advector.integrate(dt, update_owners=False)
        self._freeSurface.solve(dt)
        self.swarm.update_particle_owners()
    else:
        # Integrate Swarms in time
        self.swarm_advector.integrate(dt, update_owners=True)

    # Update stress
    if any([material.elasticity for material in self.materials]):
        self._update_stress_history(dt)

    if self.passive_tracers:
        for key, val in self.passive_tracers.items():
            if val.advector:
                val.advector.integrate(dt)

    # Do pop control
    self.population_control.repopulate()
    self.swarm.update_particle_owners()

    if self.surfaceProcesses:
        self.surfaceProcesses.solve(dt)

    # Update Time Field
    self.timeField.data[...] += dt

    if self._visugrid:
        self._visugrid.advect(dt)

    self._phaseChangeFn()


import sys
from mpi4py import MPI as _MPI
comm = GEO.comm
rank = comm.rank
size = comm.size
from underworld.utils import _swarmvarschema
#from .version import full_version
from datetime import datetime

full_version=GEO.version.full_version

#####From UW Source Code. Beucher et al., 2018
### For using voronoi integration, we also modified a little bit the run for function.

def run_for_adaptative(Model, duration=None, checkpoint_interval=None, nstep=None,
            checkpoint_times=None, restart_checkpoint=1, dt=None,
            restartStep=None, restartDir=None, output_units=None):
    
    self=Model
    
    nd=GEO.nd
    """ Run the Model
    Parameters
    ----------
    duration :
        Model time in units of time.
    checkpoint_interval :
        Checkpoint interval time.
    nstep :
        Number of steps to run.`
    checkpoint_times :
        Specify a list of additional Checkpoint times ([Time])
    restart_checkpoint :
        This parameter specify how often the swarm and swarm variables
        are checkpointed. A value of 1 means that the swarm and its
        associated variables are saved at every checkpoint.
        A value of 2 results in saving only every second checkpoint.
    dt :
        Specify the time interval (dt) to be used in
        units of time.
    restartStep :
        Restart Model. int (step number)
    restartDir :
        Restart Directory.
    output_units:
        Units used in output. If None, the units of the checkpoint_interval
        are used, if the latter does not have units, defaults to
        rcParams["time.SIunits"]
    """

    self.stepDone = 0
    self.restart(restartStep, restartDir)

    ndduration = self._ndtime + nd(duration) if duration else None

    output_dt_units = _get_output_units(
        output_units, checkpoint_interval, duration)

    checkpointer = _CheckpointFunction(
        self, duration, checkpoint_interval,
        checkpoint_times, restart_checkpoint, output_dt_units)
        
    solver_options = Model._solver.options
    Model._solver = uw.systems.Solver(stokes_SLE(Model))
    Model.solver.options = solver_options

    if not nstep:
        nstep = self.stepDone

    if dt:
        user_dt = nd(dt)
    else:
        user_dt = None

    if rank == 0:
        print("""Running with UWGeodynamics version {0}""".format(full_version))
        sys.stdout.flush()

    try:
        if self.solver.print_petsc_options():
            print("""Petsc {0}""".format(self.solver.print_petsc_options()))
            sys.stdout.flush()
    except AttributeError:
        pass

    while (ndduration and self._ndtime < ndduration) or self.stepDone < nstep:

        self._pre_solve()

        self.solve()

        self._dt = 2.0 * GEO.rcParams["CFL"] * self.swarm_advector.get_max_dt()
        
        try:
            if self.temperature:
                # Only get a condition if using SUPG
                if GEO.rcParams["advection.diffusion.method"] == "SUPG":
                    supg_dt = self._advdiffSystem.get_max_dt()
                    supg_dt *= 2.0 * GEO.rcParams["CFL"]
                    self._dt = min(self._dt, supg_dt)
        except ValueError:
            pass

        if duration:
            self._dt = min(self._dt, ndduration - self._ndtime)

        if user_dt:
            self._dt = min(self._dt, user_dt)

        check_dt = checkpointer.get_next_checkpoint_time()
        if check_dt:
            self._dt = min(self._dt, check_dt)

        dte = []
        for material in self.materials:
            if material.elasticity:
                dte.append(nd(material.elasticity.observation_time))

        if dte:
            dte = np.array(dte).min()
            # Cap dt for observation time, dte / 3.
            if dte and self._dt > (dte / 3.):
                self._dt = dte / 3.

        comm.Barrier()
        
        _update(Model)
        #self._update()

        self.step += 1
        self.stepDone += 1
        self._ndtime += self._dt

        checkpointer.checkpoint()

        if rank == 0:
            string = """Step: {0:5d} Model Time: {1:6.1f} dt: {2:6.1f} ({3})\n""".format(
                self.stepDone, _adjust_time_units(self.time),
                _adjust_time_units(self._dt),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            sys.stdout.write(string)
            sys.stdout.flush()

        self._post_solve()

    return 1

#####From UW Source Code. Beucher et al., 2022- this allows to run the model using voronoi swarm
def stokes_SLE(Model):

    """ Stokes SLE """
    self=Model
    if any([material.viscosity for material in self.materials]):

        conditions = list()
        conditions.append(self.velocityBCs)

        if self._stressBCs:
            conditions.append(self.stressBCs)

        fssa = None
        if self._fssa_factor:
            fssa = self._fssa

        self._stokes_SLE = uw.systems.Stokes(
            velocityField=self.velocityField,
            pressureField=self.pressureField,
            voronoi_swarm = self.swarm,
            conditions=conditions,
            _fn_fssa = fssa,
            fn_viscosity=self._viscosityFn,
            fn_bodyforce=self._buoyancyFn,
            fn_stresshistory=self._elastic_stressFn,
            fn_one_on_lambda=self._lambdaFn)

    return self._stokes_SLE

    
Model._solver = uw.systems.Solver(stokes_SLE(Model))


#Solver Parameters
Model.solver.set_inner_method("mumps")
Model.solver.set_penalty(1e6)
#GEO.rcParams["initial.nonlinear.tolerance"] = 1e-4


# %%


#Fields to Save
outputss=['temperature',
        'pressureField',
         'strainRateField',
         'velocityField',
          'projStressField',
          'projTimeField',
           'projMaterialField',
         'projViscosityField',
         'projStressField',
         'projMeltField',
          'projPlasticStrain',
         'projDensityField',
         'projStressTensor',
         ]
GEO.rcParams['default.outputs']=outputss


#This is needed just in case the model do not converges with the default configuration of the solver

#solver = Model.solver
#scr_rtol = 1e-6
## Schur complement solver options
#solver.options.scr.ksp_rtol = scr_rtol
#solver.options.scr.ksp_type = "fgmres"
##solver.options.main.list()
#
## Inner solve (velocity), A11 options
#solver.options.A11.ksp_rtol = 1e-1 * scr_rtol
#solver.options.A11.ksp_type = "fgmres"



#Enables the coupling with badlands
Model.surfaceProcesses = GEO.surfaceProcesses.Badlands(airIndex=[Air.index],sedimentIndex=sediment.index, XML="./ressources/badlands3.xml", resolution=4. * u.kilometer,restartFolder="outbdls", checkpoint_interval=0.2 * u.megayears,restartStep=165)



def post_hook():
    vrms = Model.stokes_SLE.velocity_rms()
    step = Model.step
    time = Model.time.m_as(u.year)

    if rank == 0:
        with open(fout,'a') as f:
             f.write(f"{step}\t{time:5e}\t{vrms:5e}\n")

     #This removes plastic damage from model left and right walls
    fact1=InfplasticFn.evaluate(Model.swarm)
    Model.plasticStrain.data[:] =Model.plasticStrain.data[:] * fact1


Model.post_solve_functions["B"] = post_hook

#### This handles the initial instabilities in the model. We aim to "equilibrate the model" by removing all plastic strain during the first 1-3 Myr.
### We only allow plastic strain to localize in the trench during this first 1-3 Myr.
## After that the Equilibrium variable becomes True and the time step increases.

MaUnit=1000000#*u.years

#Equilibrium=False
#Equilibrium=True
restart=True
if restart==False:
    #Small time step at the beginning - Because isostatic compesation occurs here
    run_for_adaptative(Model,duration=(0.2*MaUnit)*u.year,checkpoint_interval=(0.2*MaUnit)*u.year,dt=500*u.years,restartDir=outputPath)

    # All plastic damage duirng equilibration is removed
    Model.plasticStrain.data[:] = 0.

    #It gives some time the model to take "impulse" again as plastic damage was removed everywhere in the model
    run_for_adaptative(Model,duration=(0.2*MaUnit)*u.year,checkpoint_interval=(0.2*MaUnit)*u.year,dt=1500*u.years,restartDir=outputPath)

    #A little bigger time step during subduction initiation
    run_for_adaptative(Model,duration=(6.*MaUnit)*u.year,checkpoint_interval=(0.2*MaUnit)*u.year,dt=6000*u.years,restartDir=outputPath)

    #The maximum stable time step allowed by FSSA.
    run_for_adaptative(Model,duration=(25.*MaUnit)*u.year,checkpoint_interval=(0.2*MaUnit)*u.year,dt=7500*u.years,restartDir=outputPath)

    #Close to break-up the time step must decrease

    run_for_adaptative(Model,duration=(15.*MaUnit)*u.year,checkpoint_interval=(0.2*MaUnit)*u.year,dt=5000*u.years,restartDir=outputPath)
else:
    run_for_adaptative(Model,duration=(20.*MaUnit)*u.year,checkpoint_interval=(0.2*MaUnit)*u.year,dt=5000*u.years,restartDir=outputPath,restartStep=164)
    run_for_adaptative(Model,duration=(10.*MaUnit)*u.year,checkpoint_interval=(0.2*MaUnit)*u.year,dt=5000*u.years,restartDir=outputPath)

