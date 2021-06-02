This folder contains the following programs:

### obj_to_ccm
This program creates a serial mesh file format (labelled .ccm) from an input OBJ file. In turn, these .ccm files can be used as input for the subsequent programs. A list of .ccm meshes is provided in the `meshes/` folder. Note that the included OBJ parser supports the OBJ files provided in the OpenSubdiv repo, which sometimes includes (non-standard) semi-sharp crease tags.

### mesh_info
This program is useful to display properties of a .ccm mesh file.

### subd_cpu
This code provides a basic example to compute a subdivision in parallel on the CPU. It is compiled into two programs: `subd_cpu` and `bench_cpu`. By default, the former program subdivides a .ccm mesh and exports each subdivision level into several .obj files. The latter program runs the subdivision 100 times and displays timings. 
Typical usage is the following: 
```sh
subd_cpu pathToCcm.ccm maxSubdivisionDepth 1
```
where `maxSubdivisionDepth` is an integer value that controls the target subdivision level and 
the third argument is a flag to export the resulting subdivisions to .obj files (value should be 0 or 1).
 

### subd_gpu
This code provides a basic example to compute a subdivision in parallel on the GPU using OpenGL shaders. The shaders require hardware support for the GLSL extension `GL_NV_shader_atomic_float`. The code is compiled into two programs: `subd_gpu` and `bench_gpu`. By default, the former program subdivides a .ccm mesh and exports each subdivision level into several .obj files. The latter program runs the subdivision 100 times and displays timings. 
Typical usage is the following: 
```sh
subd_cpu pathToCcm.ccm maxSubdivisionDepth 1
```
where `maxSubdivisionDepth` is an integer value that controls the target subdivision level and 
the third argument is a flag to export the resulting subdivisions to .obj files (value should be 0 or 1).


