# import sys
# sys.path.append('/usr/lib')
import sys
import csv
import copy
import numpy as np
from collections import deque

import gmsh


MULTIPLIER_MM = 1e3
MULTIPLIER_UM = 1e6
MULTIPLIER_NM = 1e9

lc = 10

RECOMBINE = True
EPSILON = 1e-6

EXTRUDE_HELMHOLTZ = True
EXTRUDE_SUBMEMBRANE = True
EXTRUDE_MEMBRANE = True
EXTRUDE_SOLUTION = True
EXTRUDE_DEFECTS = True

SOLUTION_LAYERS = 3

DEF_LC_MIN = lambda defect: defect.radius / 2.5
DEF_LC_MAX = lambda defect: 100
# DEF_R_MIN = lambda defect: defect.radius * 0.1
DEF_R_MIN = lambda defect: 1
# DEF_R_MAX = lambda defect: defect.radius * 30
DEF_R_MAX = lambda defect: 300
DEF_THRESHOLD_FIELD_SIGMOID = 0


class Defect:
    def __init__(self, id_, x, y, radius):
        self.id_ = id_
        self.x = x
        self.y = y
        self.radius = radius

    def intersects(self, other):
        dist_centers = np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        return dist_centers <= (self.radius + other.radius)


class DefectCluster:
    def __init__(self, defects):
        self.defects = defects

    @staticmethod
    def make_all(defects):
        intersections = np.zeros((len(defects), len(defects)))
        for idx1, def1 in enumerate(defects):
            for idx2, def2 in enumerate(defects):
                if def1.intersects(def2):
                    intersections[idx1, idx2] = 1

        # Zero out upper triangle
        intersections[np.triu_indices(len(defects))] = 0

        clusters = []
        for idx, row in enumerate(intersections):
            if np.count_nonzero(row) > 0:
                cluster_defects = [defects[idx]]
                cluster_defects += [
                    defects[idx2] for idx2, val in enumerate(row)
                    if val != 0]
                cluster_defects.sort(key=lambda c: c.id_)
                cluster = DefectCluster(cluster_defects)
                clusters.append(cluster)
        return clusters


class Hexagon:
    def __init__(self, center, vertices, side_length):
        self.center = center
        self.vertices = vertices
        self.side_length = side_length

    @property
    def x_min(self):
        return min([v[0] for v in self.vertices])

    @property
    def x_max(self):
        return max([v[0] for v in self.vertices])

    @property
    def y_min(self):
        return min([v[1] for v in self.vertices])

    @property
    def y_max(self):
        return max([v[1] for v in self.vertices])

    @property
    def area(self):
        return 3 * np.sqrt(3) * self.side_length ** 2 / 2

    def edges(self):
        indices_start = list(range(len(self.vertices)))
        indices_end = deque(indices_start)
        indices_end.rotate(-1)
        for index_start, index_end in zip(indices_start, indices_end):
            yield self.vertices[index_start], self.vertices[index_end]


class Experiment:
    # 0-indexed
    SIDE_LENGTH_LINE = 2
    FIRST_EDGE_LINE = 6
    CENTER_LINE = 13
    FIRST_DEFECT_LINE = 16

    def __init__(self, hexagon, defects):
        self.hexagon = hexagon
        self.defects = defects

    @property
    def d_helmholtz(self):
        return 0.66 * 1e-9 * MULTIPLIER_NM

    @property
    def d_submembrane(self):
        return 1.8 * 1e-9 * MULTIPLIER_NM

    @property
    def d_membrane(self):
        return 3.0 * 1e-9 * MULTIPLIER_NM

    @property
    def d_solution(self):
        return 50.0 * 1e-9 * MULTIPLIER_NM

    @property
    def z_helmholtz_bottom(self):
        return 0

    @property
    def z_helmholtz_top(self):
        return self.d_helmholtz

    @property
    def z_submembrane_bottom(self):
        return self.z_helmholtz_top
    
    @property
    def z_submembrane_top(self):
        return self.z_submembrane_bottom + self.d_submembrane

    @property
    def z_membrane_bottom(self):
        return self.z_submembrane_top

    @property
    def z_membrane_top(self):
        return self.z_membrane_bottom + self.d_membrane

    @property
    def z_solution_bottom(self):
        return self.z_membrane_top

    @property
    def z_solution_top(self):
        return self.z_solution_bottom + self.d_solution

    def create_hexagon_surface(self, factory, z):
        # Hexagon points
        tags_points = []
        for x, y in self.hexagon.vertices:
            pt = factory.addPoint(x, y, z, lc)
            tags_points.append(pt)

        # Hexagon edges
        tags_edges = []
        points_start = tags_points
        points_end = deque(points_start)
        points_end.rotate(-1)
        for idx, (point_start, point_end) in enumerate(zip(points_start, points_end)):
            edge_tag = factory.addLine(point_start, point_end)
            tags_edges.append(edge_tag)

        # Hexagon surface
        cl = factory.addCurveLoop(tags_edges)
        surface = factory.addPlaneSurface([cl])
        return surface

    @classmethod
    def read(cls, filename):
        with open(filename, 'r') as fp:
            lines = [line for line in csv.reader(fp)]
        
        # Hexagon properties
        side_length = float(lines[cls.SIDE_LENGTH_LINE][1])
        center = (
            float(lines[cls.CENTER_LINE][1]),
            float(lines[cls.CENTER_LINE][2]),)

        vertices = []
        for vertex_line in lines[cls.FIRST_EDGE_LINE:cls.FIRST_EDGE_LINE + 6]:
            vertex = (
                float(vertex_line[1]),
                float(vertex_line[2]),)
            vertices.append(vertex)

        hexagon = Hexagon(center, vertices, side_length)

        # Defects with coordinates and radiuses
        defects = []
        for idx, def_line in enumerate(lines[cls.FIRST_DEFECT_LINE:]):
            x = float(def_line[0]) * MULTIPLIER_NM
            y = float(def_line[1]) * MULTIPLIER_NM
            radius = float(def_line[2]) * MULTIPLIER_NM
            defect = Defect(idx, x, y, radius)
            defects.append(defect)

        return Experiment(hexagon, defects)


def get_horizontal_surfaces(exp, z):
    return model.getEntitiesInBoundingBox(
        exp.hexagon.x_min - EPSILON,
        exp.hexagon.y_min - EPSILON,
        z - EPSILON,
        exp.hexagon.x_max + EPSILON,
        exp.hexagon.y_max + EPSILON,
        z + EPSILON,
        dim=2,)



model = gmsh.model
factory = model.occ

# gmsh.initialize()
gmsh.initialize('', False)
gmsh.option.setNumber("General.Terminal", 1)
# gmsh.option.setNumber("Geometry.AutoCoherence", 0)
# gmsh.option.setNumber("Geometry.Tolerance", 1e-1)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.25)
# gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 100)
# gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
# gmsh.option.setNumber("Mesh.MinimumCirclePoints", 15)


# gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
# gmsh.option.setNumber("Mesh.Recombine3DLevel", 2)
# gmsh.option.setNumber("Mesh.Recombine3DConformity", 4)


gmsh.option.setNumber("Mesh.RecombineAll", 1)
# gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 10)
# gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)

gmsh.option.setNumber("Mesh.Algorithm", 8)
# gmsh.option.setNumber("Mesh.Algorithm3D", 1)
# gmsh.option.setNumber("Mesh.FlexibleTransfinite", 1)
# gmsh.option.setNumber("Mesh.Smoothing", 100)
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)


gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
# gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1e-6)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)



'''
Threading
'''
gmsh.option.setNumber("General.NumThreads", 8)
gmsh.option.setNumber("Geometry.OCCParallel", 1)
gmsh.option.setNumber("Mesh.MaxNumThreads1D", 8)
gmsh.option.setNumber("Mesh.MaxNumThreads2D", 8)
gmsh.option.setNumber("Mesh.MaxNumThreads3D", 8)





model.add("membrane-model")


# exp = Experiment.read('/data/Dropbox/Git/tblm-defects/modeling/experiments/MeshTests_N=10_Ndef=100_r=10nm/coord/Coordinates_9.csv')
# exp = Experiment.read('/data/Dropbox/Git/tblm-defects/modeling/sandbox/coordinates/Tiny.csv')
# exp = Experiment.read('/data/Dropbox/Git/tblm-defects/modeling/sandbox/coordinates/Coordinates_1.csv')

# exp = Experiment.read('/data/Dropbox/Git/tblm-defects/modeling/experiments/Sintetiniai_Duomenys/20190307_SulipimasNanometrais_51_ir_195/Sulipe_Klasterizuoti_Atsitiktinai_Issideste_Defektu_Centrai_-_FiksuotasSulipimoSlenkstis=195nm_-_DefektuTankis_Ndef=270_IKvadratiniMikrometra_DefektuSkaicius_N=505_HeksagonoKrastine_a=848.47nm/1_Scenarijus_-_Visi_Defektu_Spinduliai_Vienodi_rdef=0.5nm/Defektu_Centru_Koordinates/DefektuCentruKoordinates_-_AtvejoNr_001_DefektuTankis_Ndef=270_IKvadratiniMikrometra_DefektuSkaicius_N=505_HeksagonoKrastine_a=848.47nm_BuferioZonosPlotis=30nm.csv')
exp = Experiment.read('/data/Dropbox/Git/tblm-defects/modeling/experiments/Sintetiniai_Duomenys/20190502_DefektuDydziai/Klasterizuoti_Atsitiktinai_Issideste_Defektu_Centrai_-_MinimaliTikimybe=0.11_KlasteriuDydis=0.5_DefektuTankis_Ndef=0.1_IKvadratiniMikrometra_DefektuSkaicius_N=505_HeksagonoKrastine_a=44087.9nm/2_Scenarijus_-_Visi_Defektu_Spinduliai_Vienodi_rdef=1nm/Defektu_Centru_Koordinates/DefektuCentruKoordinates_-_AtvejoNr_001_DefektuTankis_Ndef=0.1_IKvadratiniMikrometra_DefektuSkaicius_N=505_HeksagonoKrastine_a=44087.9nm_BuferioZonosPlotis=12nm.csv')

# exp = Experiment.read('/data/Dropbox/Git/tblm-defects/modeling/experiments/Sintetiniai_Duomenys/20180416_SkirtingiDefektuDydziai_N=26/9nm/DefektuSpindulys_r0=9nm_DefektuTankis_Ndef=1_IKvadratiniMikrometra_DefektuSkaicius_N=26_HeksagonoKrastine_a=3163.45nm_BuferioZonosPlotis=16nm/Defektu_Centru_Koordinates/DefektuCentruKoordinates_-_DefektuSpindulys_r0=9nm_DefektuTankis_Ndef=1_IKvadratiniMikrometra_DefektuSkaicius_N=26_HeksagonoKrastine_a=3163.45nm_BuferioZonosPlotis=16nm_AtvejoNr_001.csv')




'''
Helmholtz layer
'''
# base_helmholtz = exp.create_hexagon_surface(factory, 0)
# ext_helmholtz = factory.extrude([(2, base_helmholtz)], 0, 0, exp.d_helmholtz, 
#     numElements=[1], recombine=RECOMBINE)
# vol_helmholtz = ext_helmholtz[1][1]


'''
Defects
'''
base_submembrane = exp.create_hexagon_surface(factory, exp.d_helmholtz)

clusters = DefectCluster.make_all(exp.defects)
standalone_defects = copy.copy(exp.defects)
for cluster in clusters:
    for cluster_defect in cluster.defects:
        standalone_defects.remove(cluster_defect)


'''
Defects - standalone
'''
defect_base_tags = [
    factory.addDisk(
        defect.x, defect.y,
        exp.d_helmholtz, 
        defect.radius, defect.radius)
    for defect in exp.defects]

# Intersect defects with base hexagon
for defect_base_tag in defect_base_tags:
    factory.intersect([(2, base_submembrane)], [(2, defect_base_tag)], removeObject=False, removeTool=True)

if exp.defects:
    base_submembrane_no_defects = factory.cut(
        [(2, base_submembrane)], 
        [(2, defect_base_tag) for defect_base_tag in defect_base_tags],
        removeObject=True, removeTool=False)[0][0][1]

    defect_indices = [
        idx for idx, defect_base_tag in enumerate(defect_base_tags)
        if exp.defects[idx] in standalone_defects
    ]
    # if EXTRUDE_DEFECTS:
    #     defect_volumes = [
    #         factory.extrude([(2, defect_base_tags[idx])], 0, 0, 
    #             exp.d_submembrane + exp.d_membrane, 
    #             numElements=[1], recombine=RECOMBINE)[1][1]
    #         for idx in defect_indices]
else:
    base_submembrane_no_defects = base_submembrane
    defect_indices = []
    defect_volumes = []


'''
Defects - clusters
'''
cluster_volumes_all = []
cluster_indices_all = []
cluster_def_tags_unions = []

for cluster in clusters:
    cluster_def_tags = [
        defect_base_tags[exp.defects.index(defect)]
        for defect in cluster.defects]

    cluster_union = factory.fuse(
        [(2, cluster_def_tags[0])],
        [(2, cluster_def_tag) for cluster_def_tag in cluster_def_tags[1:]],
        removeObject=True, removeTool=True)

    cluster_def_tags_unions.append((cluster_def_tags, cluster_union))

    # if EXTRUDE_DEFECTS:
    #     cluster_tags = factory.extrude(cluster_union[0], 0, 0,
    #         exp.d_submembrane + exp.d_membrane,
    #         numElements=[1], recombine=RECOMBINE)

    #     cluster_volumes = [
    #         cluster_tag[1] for cluster_tag in cluster_tags 
    #         if cluster_tag[0] == 3]
    #     cluster_volumes_all.append(cluster_volumes)

    #     cluster_indices = [
    #         exp.defects.index(defect) for defect in cluster.defects]
    #     cluster_indices_all.append(cluster_indices)


factory.removeAllDuplicates()
factory.synchronize()


'''
Defect volumes
'''
if EXTRUDE_DEFECTS:
    defect_volumes = [
        factory.extrude([(2, defect_base_tags[idx])], 0, 0, 
            exp.d_submembrane + exp.d_membrane, 
            numElements=[1], recombine=RECOMBINE)[1][1]
        for idx in defect_indices]

    for cluster_def_tags, cluster_union in cluster_def_tags_unions:
        cluster_tags = factory.extrude(cluster_union[0], 0, 0,
            exp.d_submembrane + exp.d_membrane,
            numElements=[1], recombine=RECOMBINE)

        cluster_volumes = [
            cluster_tag[1] for cluster_tag in cluster_tags 
            if cluster_tag[0] == 3]
        cluster_volumes_all.append(cluster_volumes)

        cluster_indices = [
            exp.defects.index(defect) for defect in cluster.defects]
        cluster_indices_all.append(cluster_indices)
    



'''
Submembrane
'''
if EXTRUDE_SUBMEMBRANE:
    ext_submembrane = factory.extrude(
        [(2, base_submembrane_no_defects)], 0, 0, 
        exp.d_submembrane, numElements=[1], recombine=RECOMBINE)
    vol_submembrane = ext_submembrane[1][1]
    base_membrane = ext_submembrane[0][1]

    group_submembrane = model.addPhysicalGroup(3, [vol_submembrane])
    model.setPhysicalName(3, group_submembrane, 'submembrane')


'''
Membrane
'''
if EXTRUDE_MEMBRANE:
    ext_membrane = factory.extrude(
        [(2, base_membrane)], 0, 0, 
        exp.d_membrane, numElements=[1], recombine=RECOMBINE)
    vol_membrane = ext_membrane[1][1]
    base_solution = ext_membrane[0][1]

    group_membrane = model.addPhysicalGroup(3, [vol_membrane])
    model.setPhysicalName(3, group_membrane,    'membrane')


factory.synchronize()

'''
Solution
'''
if EXTRUDE_SOLUTION:
    ext_solution = factory.extrude(
        get_horizontal_surfaces(exp, exp.z_membrane_top),
        0, 0, exp.d_solution, 
        numElements=[SOLUTION_LAYERS], recombine=RECOMBINE)
    vols_solution = [vol[1] for vol in ext_solution if vol[0] == 3]

    group_solution = model.addPhysicalGroup(3, vols_solution)
    model.setPhysicalName(3, group_solution, 'solution')


# factory.removeAllDuplicates()
factory.synchronize()

'''
[NEW] Helmholtz
'''
if EXTRUDE_HELMHOLTZ:
    ext_helmholtz = factory.extrude(
        get_horizontal_surfaces(exp, exp.z_submembrane_bottom),
        0, 0, -1 * exp.d_helmholtz,
        numElements=[1], recombine=RECOMBINE)
    vols_helmholtz = [vol[1] for vol in ext_helmholtz if vol[0] == 3]

    # TODO: the following causes the mesh to fail to import with deal.ii:
    # if len(vols_helmholtz) > 1:
    #     vol_helmholtz = factory.fuse(
    #         [vols_helmholtz[0]], vols_helmholtz[1:],
    #         removeObject=True, removeTool=True)[0][0][1]
    # else:
    #     vol_helmholtz = vols_helmholtz[0]

    group_helmholtz = model.addPhysicalGroup(3, vols_helmholtz)
    model.setPhysicalName(3, group_helmholtz, 'helmholtz')


# factory.removeAllDuplicates()
# factory.synchronize()

'''
[NEW] Solution
'''
'''
EXTRUDE_SOLUTION:
    # heights = [
    #     exp.d_solution * 0.1,
    #     exp.d_solution * 0.2,
    #     exp.d_solution * 0.3,
    #     exp.d_solution * 0.4,]
    heights = [exp.d_solution]

    ext_solution_base = get_horizontal_surfaces(exp, exp.z_membrane_top)
    cumulative_height = exp.z_membrane_top

    vols_solution = []
    for height in heights:
        print(ext_solution_base)
        ext_solution = factory.extrude(
            ext_solution_base,
            0, 0, height,
            numElements=[1], recombine=RECOMBINE)
        vols_solution += [vol for vol in ext_solution if vol[0] == 3]
        cumulative_height += height
        factory.synchronize()
        ext_solution_base = get_horizontal_surfaces(exp, cumulative_height)

    # TODO: the following causes the mesh to fail to import with deal.ii:
    # if len(vols_solution) > 1:
    #     vol_solution = factory.fuse(
    #         [vols_solution[0]], vols_solution[1:],
    #         removeObject=True, removeTool=True)[0][0][1]
    # else:
    #     vol_solution = vols_solution[0]

    group_solution = model.addPhysicalGroup(3, vols_solution)
    model.setPhysicalName(3, group_solution,    'solution')
'''


# factory.removeAllDuplicates()
factory.synchronize()

# Layers
# if EXTRUDE_HELMHOLTZ:
    # group_helmholtz = model.addPhysicalGroup(3, [vol_helmholtz])
    # group_helmholtz = model.addPhysicalGroup(3, vols_helmholtz)
    # model.setPhysicalName(3, group_helmholtz,   'helmholtz')

# if EXTRUDE_SUBMEMBRANE:
#     group_submembrane = model.addPhysicalGroup(3, [vol_submembrane])
#     model.setPhysicalName(3, group_submembrane, 'submembrane')

# if EXTRUDE_MEMBRANE:
#     group_membrane = model.addPhysicalGroup(3, [vol_membrane])
#     model.setPhysicalName(3, group_membrane,    'membrane')

# if EXTRUDE_SOLUTION:
#     group_solution = model.addPhysicalGroup(3, [vol_solution])
#     model.setPhysicalName(3, group_solution,    'solution')
    

# Top surface
tags_top = get_horizontal_surfaces(exp, exp.z_solution_top)
group_top = model.addPhysicalGroup(2, 
    [dimtag[1] for dimtag in tags_top])
model.setPhysicalName(2, group_top, 'top')


# Bottom surface
tags_bottom = get_horizontal_surfaces(exp, exp.z_helmholtz_bottom)
group_bottom = model.addPhysicalGroup(2, 
    [dimtag[1] for dimtag in tags_bottom])
model.setPhysicalName(2, group_bottom, 'bottom')


# Side surfaces
tags_sides = []
for vertex_start, vertex_end in exp.hexagon.edges():
    x_start, y_start = vertex_start
    x_end, y_end = vertex_end
    x_min = min(x_start, x_end)
    y_min = min(y_start, y_end)
    z_min = exp.z_helmholtz_bottom
    x_max = max(x_start, x_end)
    y_max = max(y_start, y_end)
    z_max = exp.z_solution_top

    surfaces = model.getEntitiesInBoundingBox(
        x_min - EPSILON, y_min - EPSILON, z_min - EPSILON,
        x_max + EPSILON, y_max + EPSILON, z_max + EPSILON,
        dim=2,)

    for surface_dimtag in surfaces:
        dim, tag = surface_dimtag
        box_x_min, box_y_min, box_z_min, box_x_max, box_y_max, box_z_max = model.getBoundingBox(dim, tag)
        
        diff_x_min = abs(x_min - box_x_min)
        diff_y_min = abs(y_min - box_y_min)
        diff_x_max = abs(x_max - box_x_max)
        diff_y_max = abs(y_max - box_y_max)
        slope_xy_box = (box_x_max - box_x_min) / (box_y_max - box_y_min)

        if abs(y_max - y_min) < EPSILON:
            if diff_y_min < EPSILON and diff_y_max < EPSILON:
                tags_sides.append(tag)
        elif abs(x_max - x_min) < EPSILON:
            if diff_x_min < EPSILON and diff_x_max < EPSILON:
                tags_sides.append(tag)
        else:
            slope_xy_side = (x_max - x_min) / (y_max - y_min)
            if abs(slope_xy_side - slope_xy_box) < EPSILON:
                tags_sides.append(tag)

group_sides = model.addPhysicalGroup(2, tags_sides)
model.setPhysicalName(2, group_sides, 'sides')


if EXTRUDE_DEFECTS:
    # Standalone defects
    for defect_volume, defect_index in zip(defect_volumes, defect_indices):
        group_defect = model.addPhysicalGroup(3, [defect_volume])
        model.setPhysicalName(3, group_defect, f'defect{defect_index}')


    # Clustered defects
    for cluster_volumes, cluster_indices in zip(cluster_volumes_all, cluster_indices_all):
        group_cluster = model.addPhysicalGroup(3, cluster_volumes)
        cluster_label = f'defects{cluster_indices[0]}'
        for cluster_idx in cluster_indices[1:]:
            cluster_label += f'_{cluster_idx}'
        model.setPhysicalName(3, group_cluster, cluster_label)



# factory.removeAllDuplicates()
factory.synchronize()

'''
Mesh fields
'''
volumes_sub_mem_def = []
if EXTRUDE_SUBMEMBRANE:
    volumes_sub_mem_def.append(vol_submembrane)

if EXTRUDE_MEMBRANE:
    volumes_sub_mem_def.append(vol_membrane)

# volumes_sub_mem_def = [vol_submembrane, vol_membrane] + defect_volumes
if EXTRUDE_DEFECTS:
    volumes_sub_mem_def += defect_volumes
    for cluster_volumes in cluster_volumes_all:
        volumes_sub_mem_def += cluster_volumes

z_sub_mem_min = exp.d_helmholtz
z_sub_mem_max = exp.d_helmholtz + exp.d_submembrane + exp.d_membrane
z_sol_top     = z_sub_mem_max + exp.d_solution
z_helm_bottom = 0

field_defects = []
field_defect_tag = 5


mesh_field_ids = []
mesh_field_id = 1
'''
for defect in exp.defects:
    line_lc_min = DEF_LC_MIN(defect)
    line_lc_max = DEF_LC_MAX(defect)
    line_r_min = DEF_R_MIN(defect)
    line_r_max = DEF_R_MAX(defect)

    line_slope = (line_lc_max - line_lc_min) / (line_r_max - line_r_min)
    line_intercept = line_lc_min - line_slope * line_r_min

    f_sub_mem_linear   = f"Min({line_lc_max}, Max({line_lc_min}, {line_slope} * Sqrt((x-{defect.x})^2 + (y-{defect.y})^2) + ({line_intercept})))"
    # w_sub_mem   = f"(1 - ({w_sol} + {w_helm}))"
    w_sub_mem = 1

    model.mesh.field.add("MathEval", field_defect_tag)
    model.mesh.field.setString(field_defect_tag, "F", 
        # f"({w_helm} * {f_helm}) + ({w_sub_mem} * {f_sub_mem}) + ({w_sol} * {f_sol})")
        f"{w_sub_mem} * {f_sub_mem_linear}")
    field_defects.append(field_defect_tag)
    field_defect_tag += 1
'''


# model.mesh.setRecombine(2, base_submembrane_no_defects)
# for defect in exp.defects:
#     defect_surface_dimtags = model.getEntitiesInBoundingBox(
#         defect.x - defect.radius - EPSILON, 
#         defect.y - defect.radius - EPSILON, 
#         exp.z_submembrane_bottom - EPSILON,
#         defect.x + defect.radius + EPSILON,
#         defect.y + defect.radius + EPSILON,
#         exp.z_submembrane_bottom + EPSILON, 
#         dim=2)
#     for dimtag in defect_surface_dimtags:
#         model.mesh.setRecombine(2, dimtag[1])



# for idx, defect_base_tag in enumerate(defect_base_tags):
for defect in exp.defects:
    defect_curve_dimtags = model.getEntitiesInBoundingBox(
        defect.x - defect.radius - EPSILON, 
        defect.y - defect.radius - EPSILON, 
        exp.z_submembrane_bottom - EPSILON,
        defect.x + defect.radius + EPSILON,
        defect.y + defect.radius + EPSILON,
        exp.z_submembrane_bottom + EPSILON, 
        dim=1)
    defect_curves = [dimtag[1] for dimtag in defect_curve_dimtags]

    distance_id = mesh_field_id
    threshold_id = mesh_field_id + 1
    mesh_field_id += 2

    model.mesh.field.add("Distance", distance_id)
    model.mesh.field.setNumber(distance_id, "NNodesByEdge", 50)
    model.mesh.field.setNumbers(distance_id, "EdgesList", defect_curves)

    model.mesh.field.add("Threshold", threshold_id)
    model.mesh.field.setNumber(threshold_id, "IField", distance_id)
    model.mesh.field.setNumber(threshold_id, "LcMin", DEF_LC_MIN(defect))
    model.mesh.field.setNumber(threshold_id, "LcMax", DEF_LC_MAX(defect))
    model.mesh.field.setNumber(threshold_id, "DistMin", DEF_R_MIN(defect))
    model.mesh.field.setNumber(threshold_id, "DistMax", DEF_R_MAX(defect))
    model.mesh.field.setNumber(threshold_id, "StopAtDistMax", 1)
    model.mesh.field.setNumber(threshold_id, "Sigmoid", DEF_THRESHOLD_FIELD_SIGMOID)
    mesh_field_ids.append(threshold_id)





# field_test = 100
# model.mesh.field.add("MathEval", field_test)
# model.mesh.field.setString(field_test, "F", f'If (z > 5.46) 10 EndIf')
# field_defects = [field_test]


'''
field_sub_mem_def = max(field_defects) + 1
model.mesh.field.add("Min", field_sub_mem_def)
model.mesh.field.setNumbers(field_sub_mem_def, "FieldsList",
    # [2, 3] + field_defects)
    field_defects)


field_restrict = max(field_defects) + 2
model.mesh.field.add("Restrict", field_restrict)
model.mesh.field.setNumbers(field_restrict, "RegionsList", volumes_sub_mem_def)
    # [vol_submembrane, vol_membrane] + defect_tags)
model.mesh.field.setNumber(field_restrict, "IField", field_sub_mem_def)
'''

# if field_defects:
# field_all = max(field_defects) + 3
mesh_field_id_all = max(mesh_field_ids) + 1
model.mesh.field.add("Min", mesh_field_id_all)
model.mesh.field.setNumbers(
    mesh_field_id_all, "FieldsList", mesh_field_ids)

model.mesh.field.setAsBackgroundMesh(mesh_field_id_all)

factory.synchronize()
model.mesh.generate(3)

# gmsh.write("membrane_quad.bdf")
# gmsh.write("membrane_quad.msh")
gmsh.write(sys.argv[1])
gmsh.fltk.run()
gmsh.finalize()
