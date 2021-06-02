
This repository provides source code to reproduce some of the results of my paper ["A Halfedge Refinement Rule for Parallel Catmull-Clark Subdivision)"](https://onrendering.com/).

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
