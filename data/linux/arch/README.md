To build on Arch Linux, run

```bash
git clone https://github.com/kririae/dependencies_arch dependencies
# or just
git clone --recursive https://github.com/kririae/mitsuba
# or
git submodule update --init --recursive
```

Which provides OpenEXR 2.5 and GLEW MX, since in OpenEXR 3, `Imath` was moved into a independent library, see https://www.openexr.com/ for more information;
and in GLEW 2.x, GLEW MX was removed(though we could simply delete the corresponding code sections).

Yet, since the project depends on qt5, you must execute `data/linux/arch/env-archlinux.sh` to create the soft links.
Inspect the code for more information.

BTW, I modified all `sys.platform == 'linux2'` to `sys.platform.startwith('linux')` to provide compatibility.
