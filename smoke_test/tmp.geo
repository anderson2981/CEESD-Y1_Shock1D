size=0.001;
left_boundary_loc = 0.0;
right_boundary_loc = 0.3;
bottom_boundary_loc = 0.0;
top_boundary_loc = 0.02;
wall_boundary_loc = 0.32;
bl_ratio = 3;
interface_ratio = 2;
Point(1) = {left_boundary_loc,  bottom_boundary_loc, 0, size};
Point(2) = {right_boundary_loc, bottom_boundary_loc,  0, size};
Point(3) = {right_boundary_loc, top_boundary_loc,    0, size};
Point(4) = {left_boundary_loc,  top_boundary_loc,    0, size};
Point(5) = {wall_boundary_loc,  bottom_boundary_loc,    0, size};
Point(6) = {wall_boundary_loc,  top_boundary_loc,    0, size};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {3, 6};
Line(6) = {2, 5};
Line(7) = {5, 6};
Line Loop(1) = {-4, -3, -2, -1};
Line Loop(2) = {2, 5, -7, -6};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Physical Surface('fluid') = {1};
Physical Surface('solid') = {2};
Physical Curve('fluid_inflow') = {4};
Physical Curve('fluid_wall') = {1, 3};
Physical Curve('interface') = {2};
Physical Curve('solid_wall') = {5, 6, 7};
//
// Create distance field from curves, excludes cavity
Field[1] = Distance;
Field[1].CurvesList = {1,3};
Field[1].NumPointsPerCurve = 100000;

//Create threshold field that varrries element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = size / bl_ratio;
Field[2].SizeMax = size;
Field[2].DistMin = 0.0002;
Field[2].DistMax = 0.005;
Field[2].StopAtDistMax = 1;
//
// Create distance field from curves, excludes cavity
Field[4] = Distance;
Field[4].CurvesList = {2};
Field[4].NumPointsPerCurve = 100000;

//Create threshold field that varrries element size near boundaries
Field[5] = Threshold;
Field[5].InField = 4;
Field[5].SizeMin = size / interface_ratio;
Field[5].SizeMax = size;
Field[5].DistMin = 0.0002;
Field[5].DistMax = 0.005;
Field[5].StopAtDistMax = 1;
//  background mesh size
Field[3] = Box;
Field[3].XMin = 0.;
Field[3].XMax = 1.0;
Field[3].YMin = -1.0;
Field[3].YMax = 1.0;
Field[3].VIn = size;

// take the minimum of all defined meshing fields
Field[100] = Min;
Field[100].FieldsList = {2, 3, 5};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

Mesh.Algorithm = 5;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 100;
