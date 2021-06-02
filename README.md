[![Build Status](https://travis-ci.com/jdupuy/HalfedgeCatmullClark.svg?branch=master)](https://travis-ci.com/jdupuy/HalfedgeCatmullClark)

This repository provides source code to reproduce some of the results of my paper ["A Halfedge Refinement Rule for Parallel Catmull-Clark Subdivision"](https://onrendering.com/).
The key contribution of this paper is to provide super simple algorithms to compute 
Catmull-Clark subdivision in parallel with support for semi-sharp creases. The algorithms are compiled in the C header-only library `CatmullClark.h`. In addition you will find a direct GLSL port of these algorithms in the 
`glsl/` folder. For various usage examples, see the `examples/` folder.

### License

Apart from the submodules folder, the code from this repository is released in public domain. You can do anything you want with them. You have no legal obligation to do anything else, although I appreciate attribution.

It is also licensed under the MIT open source license, if you have lawyers who are unhappy with public domain.

### Cloning

Clone the repository and all its submodules using the following command:
```sh
git clone --recursive git@github.com:jdupuy/HalfedgeCatmullClark.git
```

If you accidentally omitted the `--recursive` flag when cloning the repository you can retrieve the submodules like so:
```sh
git submodule update --init --recursive
